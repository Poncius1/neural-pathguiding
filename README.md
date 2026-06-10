@'
# Neural Path Guiding

Discrete neural directional sampling for variance reduction in physically based path tracing.

## Thesis topic

Muestreo direccional neural para reducción de varianza en Path Tracing físicamente basado.

## Structure

- `src/neural_path_guiding/core/`: renderer-independent sampling logic.
- `src/neural_path_guiding/training/`: PyTorch training code.
- `src/neural_path_guiding/renderers/mitsuba/`: Mitsuba-specific integration.
- `src/neural_path_guiding/evaluation/`: metrics and comparisons.
- `scripts/`: executable scripts.
- `tests/`: numerical and unit tests.
- `scenes/`: Mitsuba scenes.
- `data/`: generated datasets, ignored by Git.
- `outputs/`: rendered images and metrics, ignored by Git.
- `checkpoints/`: trained model weights, ignored by Git.
'@ | Set-Content README.md