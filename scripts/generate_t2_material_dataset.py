"""Generate a material-aware T2 dataset with feature schema v2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neural_path_guiding.config import (
    load_yaml_config,
    resolve_project_path,
)
from neural_path_guiding.core.features import (
    FEATURE_DIMENSION,
)
from neural_path_guiding.core.targets import (
    contributions_to_target_distribution,
)
from neural_path_guiding.data.dataset import save_dataset
from neural_path_guiding.data.schema import (
    NeuralGuidingDataset,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.dataset_config import (
    build_settings_from_config,
)
from neural_path_guiding.renderers.mitsuba.dataset_generator import (
    build_metadata,
    sample_camera_ray,
    validate_dataset_settings,
)
from neural_path_guiding.renderers.mitsuba.material_features import (
    MATERIAL_FEATURE_MODE,
    MATERIAL_FEATURE_SCHEMA_VERSION,
    is_supported_diffuse_bsdf,
    surface_to_material_features,
)
from neural_path_guiding.renderers.mitsuba.teachers import (
    PRODUCT_INTEGRAND_TARGET,
    TeacherContext,
    create_teacher_runtime,
    estimate_bin_contributions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate T2 data using actual diffuse "
            "material reflectance."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/simple_diffuse_t2_material_pilot.yaml"
        ),
    )

    return parser.parse_args()


def generate_material_dataset(
    settings,
) -> None:
    """Generate T2 data while retaining the v1 NPZ array layout."""

    validate_dataset_settings(settings)

    if settings.teacher.target_type != PRODUCT_INTEGRAND_TARGET:
        raise ValueError(
            "The v2 generator requires product_integrand_v1."
        )

    rng = np.random.default_rng(settings.seed)

    runtime = create_teacher_runtime(
        teacher=settings.teacher,
        seed=settings.seed,
    )

    features_list = []
    targets_list = []
    contributions_list = []
    positions_list = []
    normals_list = []
    pixels_list = []

    attempts = 0
    rejected_misses = 0
    rejected_materials = 0
    rejected_backfaces = 0

    while len(features_list) < settings.num_shading_points:
        attempts += 1

        if attempts > settings.max_sampling_attempts:
            raise RuntimeError(
                "Insufficient supported diffuse interactions. "
                f"Collected {len(features_list)} / "
                f"{settings.num_shading_points}."
            )

        pixel_x, pixel_y, ray = sample_camera_ray(
            camera=settings.camera,
            rng=rng,
        )

        interaction = settings.scene.ray_intersect(ray)

        if not bool(interaction.is_valid()):
            rejected_misses += 1
            continue

        if not is_supported_diffuse_bsdf(
            interaction.bsdf()
        ):
            rejected_materials += 1
            continue

        if float(interaction.wi.z) <= 0.0:
            rejected_backfaces += 1
            continue

        outgoing = normalize_numpy_vector(
            -mitsuba_vector_to_numpy(ray.d)
        )

        features = surface_to_material_features(
            surface_interaction=interaction,
            outgoing_direction=outgoing,
            bounce_depth=0,
        )

        feature_vector = features.to_array()

        context = TeacherContext(
            scene=settings.scene,
            surface_interaction=interaction,
            normal=features.normal,
            outgoing_direction=outgoing,
        )

        contributions = estimate_bin_contributions(
            context=context,
            bins=settings.bins,
            teacher=settings.teacher,
            rng=rng,
            runtime=runtime,
        )

        target = contributions_to_target_distribution(
            contributions=contributions,
            smoothing=settings.teacher.smoothing,
        )

        features_list.append(feature_vector)
        targets_list.append(target)
        contributions_list.append(contributions)
        positions_list.append(features.position)
        normals_list.append(features.normal)
        pixels_list.append((pixel_x, pixel_y))

    metadata = build_metadata(settings)

    # Same number and order of features, new semantic contract.
    metadata["feature_schema_version"] = (
        MATERIAL_FEATURE_SCHEMA_VERSION
    )

    metadata["material_feature_mode"] = MATERIAL_FEATURE_MODE
    metadata["roughness_semantics"] = (
        "1.0 convention for pure Lambertian reflection"
    )
    metadata["material_support"] = "diffuse_reflection_only"
    metadata["sampling_depth"] = 0
    metadata["rejected_interactions"] = {
        "misses": rejected_misses,
        "unsupported_materials": rejected_materials,
        "backfaces": rejected_backfaces,
    }

    dataset = NeuralGuidingDataset(
        features=np.asarray(
            features_list,
            dtype=np.float32,
        ),
        targets=np.asarray(
            targets_list,
            dtype=np.float32,
        ),
        mean_contributions=np.asarray(
            contributions_list,
            dtype=np.float32,
        ),
        positions=np.asarray(
            positions_list,
            dtype=np.float32,
        ),
        normals=np.asarray(
            normals_list,
            dtype=np.float32,
        ),
        pixels=np.asarray(
            pixels_list,
            dtype=np.int32,
        ),
        metadata=metadata,
    )

    save_dataset(
        settings.output_path,
        dataset,
    )

    print("Material-aware T2 dataset generation completed.")
    print(f"Output: {settings.output_path.resolve()}")
    print(f"Samples: {dataset.num_samples}")
    print(f"Feature dimension: {FEATURE_DIMENSION}")
    print(f"Bins: {dataset.num_bins}")
    print(
        "Feature schema: "
        f"{MATERIAL_FEATURE_SCHEMA_VERSION}"
    )
    print(
        "Rejected interactions: "
        f"{metadata['rejected_interactions']}"
    )


def main() -> None:
    args = parse_arguments()

    config_path = resolve_project_path(
        args.config,
        PROJECT_ROOT,
    )

    config = load_yaml_config(
        config_path
    )

    settings = build_settings_from_config(
        config=config,
        config_path=config_path,
        project_root=PROJECT_ROOT,
    )

    generate_material_dataset(settings)


if __name__ == "__main__":
    main()