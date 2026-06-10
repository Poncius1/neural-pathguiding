# Arquitectura del repositorio

## Propósito del documento

Este documento describe la estructura del repositorio `neural-pathguiding` y define qué tipo de código, documentación, configuraciones y resultados deben vivir en cada carpeta.

El objetivo principal del proyecto es construir un prototipo de investigación para **muestreo direccional neural discreto aplicado a Path Tracing físicamente basado**. Aunque el proyecto es de investigación, el repositorio debe mantenerse organizado, reproducible y fácil de leer.

La arquitectura del repositorio sigue una idea central:

> El código importante debe vivir en `src/`.
> Los scripts solo deben ejecutar tareas.
> Los datos, resultados y checkpoints no deben mezclarse con el código fuente.

---

## Estructura general

```text
neural-pathguiding/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── src/
│   └── neural_path_guiding/
│       ├── core/
│       ├── training/
│       ├── renderers/
│       │   └── mitsuba/
│       └── evaluation/
│
├── scripts/
├── tests/
├── configs/
├── scenes/
├── docs/
├── data/
├── outputs/
└── checkpoints/
```

---

## Principio de diseño

La regla más importante del repositorio es separar responsabilidades.

```text
scripts
   ↓
training ───────┐
renderers ──────┼──→ core
evaluation ─────┘
```

Esto significa:

* `core/` no debe depender de `training/`.
* `core/` no debe depender de Mitsuba.
* `core/` no debe depender de scripts.
* `training/` puede usar `core/`.
* `renderers/mitsuba/` puede usar `core/`.
* `evaluation/` puede usar `core/`.
* `scripts/` puede llamar funciones de cualquier módulo, pero no debe contener lógica compleja.

Esta separación ayuda a mantener el código entendible, testeable y fácil de modificar.

---

# Carpetas principales

## `src/`

La carpeta `src/` contiene el código fuente reusable del proyecto.

```text
src/
└── neural_path_guiding/
    ├── core/
    ├── training/
    ├── renderers/
    └── evaluation/
```

Todo código que represente lógica real del proyecto debe vivir aquí.

No se debe colocar lógica importante directamente en `scripts/`.

---

## `src/neural_path_guiding/core/`

Esta es la parte principal del proyecto.

Contiene la lógica matemática y algorítmica del método de muestreo direccional neural discreto.

Esta carpeta debe mantenerse lo más limpia posible. Idealmente, sus módulos deben ser pequeños, testeables y sin dependencias innecesarias.

Contenido esperado:

```text
core/
├── bins.py
├── sampling.py
├── pdf.py
├── features.py
├── mis.py
├── mlp_runtime.py
└── model_io.py
```

### `bins.py`

Define la discretización del hemisferio local.

Responsabilidades:

* Representar bins direccionales.
* Convertir una dirección local a un índice de bin.
* Obtener los límites angulares de un bin.
* Calcular el ángulo sólido de un bin.
* Samplear una dirección continua dentro de un bin.

Este será el primer módulo del proyecto.

### `sampling.py`

Contiene funciones de sampleo.

Responsabilidades:

* Samplear un bin a partir de una distribución discreta.
* Samplear una dirección dentro del bin seleccionado.
* Devolver dirección, índice de bin y PDF asociada.
* Mantener separada la lógica de muestreo de la lógica del renderer.

### `pdf.py`

Contiene funciones para evaluar PDFs.

Responsabilidades:

* Evaluar la PDF neural:

  ```text
  pθ(ω | x) = Pθ(b | x) / Ωb
  ```

* Validar que las probabilidades sean no negativas.

* Validar que las probabilidades sumen uno.

* Aplicar mezcla con distribución uniforme para evitar probabilidades cero.

### `features.py`

Define cómo se representan los atributos locales del punto de sombreado.

Responsabilidades:

* Definir una estructura para features del punto.
* Convertir features a vectores numéricos.
* Normalizar atributos como posición, dirección, roughness y profundidad de rebote.

Features iniciales esperadas:

```text
posición 3D
normal
dirección de salida
roughness
albedo/base color
profundidad de rebote
```

### `mis.py`

Contendrá funciones para Multiple Importance Sampling.

Responsabilidades futuras:

* Balance heuristic.
* Power heuristic.
* Evaluación de PDF mezcla.
* Combinación de BSDF sampling y neural sampling.

Este módulo no debe implementarse antes de validar correctamente `bins.py`, `sampling.py` y `pdf.py`.

### `mlp_runtime.py`

Contendrá la inferencia ligera de la MLP ya entrenada.

Responsabilidades:

* Cargar pesos exportados.
* Ejecutar forward pass simple.
* Convertir logits a probabilidades.
* Aplicar mezcla uniforme para soporte completo.

La lógica de entrenamiento no debe vivir aquí.

### `model_io.py`

Contendrá utilidades para cargar y guardar modelos exportados.

Responsabilidades:

* Cargar pesos en formato `.npz`.
* Validar shapes de matrices y biases.
* Guardar metadata del modelo.
* Separar checkpoints de entrenamiento de modelos listos para inferencia.

---

## `src/neural_path_guiding/training/`

Contiene el código de entrenamiento con PyTorch.

```text
training/
├── dataset.py
├── model.py
├── losses.py
├── train.py
└── export.py
```

### `dataset.py`

Responsabilidades:

* Cargar datasets generados offline.
* Leer features y targets por bin.
* Separar training/validation.
* Validar dimensiones del dataset.

### `model.py`

Responsabilidades:

* Definir la MLP de entrenamiento.
* Mantener clara la arquitectura.
* Permitir cambiar número de bins, número de capas y tamaño oculto.

Arquitectura inicial esperada:

```text
14 features → 64 hidden → 64 hidden → N bins
```

### `losses.py`

Responsabilidades:

* Implementar cross entropy con soft labels.
* Implementar KL divergence si se requiere.
* Mantener separada la función de pérdida del loop de entrenamiento.

### `train.py`

Responsabilidades:

* Ejecutar el loop de entrenamiento.
* Cargar configuración.
* Guardar checkpoints.
* Registrar métricas de entrenamiento.

### `export.py`

Responsabilidades:

* Convertir un checkpoint de PyTorch a un formato simple de inferencia.
* Exportar pesos a `.npz`.
* Guardar metadata necesaria para reproducir la arquitectura.

---

## `src/neural_path_guiding/renderers/`

Contiene integraciones con renderers concretos.

Por ahora solo se usará Mitsuba:

```text
renderers/
└── mitsuba/
    ├── scene_loader.py
    ├── baseline_renderer.py
    ├── dataset_generator.py
    ├── neural_integrator.py
    └── adapters.py
```

---

## `src/neural_path_guiding/renderers/mitsuba/`

Contiene código específico de Mitsuba.

### `scene_loader.py`

Responsabilidades:

* Cargar escenas desde `scenes/`.
* Configurar variantes de Mitsuba.
* Mantener centralizada la carga de escenas.

### `baseline_renderer.py`

Responsabilidades:

* Renderizar imágenes base.
* Ejecutar baselines de Path Tracing clásico.
* Guardar imágenes y métricas iniciales.

Baselines esperados:

```text
Path Tracing base
BSDF sampling
Light sampling / NEE
MIS clásico
```

### `dataset_generator.py`

Responsabilidades:

* Generar puntos de sombreado.
* Extraer features.
* Samplear direcciones por bin.
* Estimar contribuciones.
* Construir targets normalizados por bin.

### `neural_integrator.py`

Responsabilidades futuras:

* Integrar el sampling neural en el proceso de Path Tracing.
* Evaluar la PDF neural.
* Combinar neural sampling con otras estrategias mediante MIS.

Este archivo debe implementarse después de tener validado el núcleo matemático.

### `adapters.py`

Responsabilidades:

* Convertir estructuras de Mitsuba a estructuras del proyecto.
* Convertir `SurfaceInteraction` a features.
* Convertir direcciones entre espacio local y mundo.
* Evitar que `core/` dependa directamente de tipos de Mitsuba.

---

## `src/neural_path_guiding/evaluation/`

Contiene código para evaluar resultados.

```text
evaluation/
├── metrics.py
├── image_io.py
├── compare.py
└── reports.py
```

### `metrics.py`

Responsabilidades:

* MSE.
* RMSE.
* PSNR.
* Error relativo.
* Otras métricas futuras.

### `image_io.py`

Responsabilidades:

* Cargar imágenes.
* Guardar imágenes.
* Convertir formatos si es necesario.

### `compare.py`

Responsabilidades:

* Comparar renders contra ground truth.
* Comparar baselines contra sampling neural.
* Generar datos para tablas y plots.

### `reports.py`

Responsabilidades futuras:

* Generar archivos `.csv`.
* Generar resúmenes de experimentos.
* Preparar resultados para la tesis.

---

# Carpetas auxiliares

## `scripts/`

Contiene scripts ejecutables.

Los scripts no deben contener la lógica principal. Deben funcionar como puntos de entrada.


```text
scripts/
├── check_env.py
├── test_bins.py
├── render_baseline.py
├── generate_dataset.py
├── train_mlp.py
├── export_model.py
├── render_neural.py
└── evaluate.py
```

Regla:

> Si un script empieza a tener demasiada lógica, esa lógica debe moverse a `src/`.

### Scripts previstos

#### `check_env.py`

Verifica que el entorno esté funcionando:

* Python.
* NumPy.
* PyTorch.
* CUDA.
* Mitsuba.

#### `test_bins.py`

Ejecuta pruebas numéricas rápidas para validar los bins del hemisferio.

Más adelante estas pruebas deben migrar a `tests/`.

#### `render_baseline.py`

Renderiza una escena con un integrador base.

#### `generate_dataset.py`

Genera dataset offline para entrenamiento supervisado.

#### `train_mlp.py`

Entrena la MLP usando una configuración.

#### `export_model.py`

Exporta el modelo entrenado a un formato de inferencia.

#### `render_neural.py`

Renderiza usando el método de muestreo neural.

#### `evaluate.py`

Calcula métricas y compara resultados.

---

## `tests/`

Contiene pruebas automáticas.

Las pruebas son importantes porque este proyecto depende de PDFs y estimadores Monte Carlo. Un error pequeño en la PDF puede introducir sesgo.

Tests esperados:

```text
tests/
├── test_bins.py
├── test_pdf.py
├── test_sampling.py
├── test_mis.py
└── test_unbiased_estimators.py
```

Primeras pruebas necesarias:

* La suma de probabilidades debe ser uno.
* La PDF debe ser positiva.
* Una dirección sampleada dentro de un bin debe ser clasificada en ese bin.
* La integral de `1` sobre el hemisferio debe aproximar `2π`.
* La integral de `cosθ` sobre el hemisferio debe aproximar `π`.

---

## `configs/`

Contiene configuraciones de experimentos.

Ejemplo:

```text
configs/
├── cornell_baseline.yaml
├── cornell_dataset.yaml
├── cornell_train.yaml
├── cornell_neural_render.yaml
└── evaluation.yaml
```

El objetivo de `configs/` es evitar valores fijos dentro del código.

Ejemplo de parámetros que deben vivir en configs:

```text
scene path
samples per pixel
max depth
número de bins
arquitectura de la MLP
rutas de datasets
rutas de outputs
```

---

## `scenes/`

Contiene la implementacion de las escenas de Mitsuba.
Estructura:

```text
scenes/
├── cornell/
│   ├── cornell_box.xml
│   └── assets/
├── glossy/
└── README.md
```

Las escenas deben mantenerse separadas del código fuente.

---

## `docs/`

Documentación técnica.

```text
docs/
├── architecture.md
└── method_notes.md
├── dataset_generation.md
├── experiments.md
├── results.md
```

### `architecture.md`

Explica la organización del repositorio.

### `method_notes.md`

Explica el método matemático.

### `dataset_generation.md`

Documentará cómo se generan los datasets.

### `experiments.md`

Registrará experimentos, configuraciones y observaciones.

### `results.md`

Resumirá resultados útiles para la tesis.


---

## `data/`

Contiene datasets generados localmente.

Ejemplos:

```text
data/
├── cornell_train.npz
├── cornell_val.npz
└── metadata.json
```

Esta carpeta está ignorada por Git excepto por `.gitkeep`.


---

## `outputs/`

Contiene resultados generados.

Ejemplos:

```text
outputs/
├── renders/
├── crops/
├── metrics/
└── plots/
```

Esta carpeta está ignorada por Git excepto por `.gitkeep`.

---

## `checkpoints/`

Contiene pesos entrenados.

Ejemplos:

```text
checkpoints/
├── model_epoch_010.pt
├── model_best.pt
└── exported_model.npz
```

Esta carpeta está ignorada por Git excepto por `.gitkeep`.

---

# Flujo general del proyecto

El flujo esperado del proyecto será:

```text
1. Verificar entorno
2. Validar bins y PDFs
3. Renderizar baseline
4. Generar dataset offline
5. Entrenar MLP
6. Exportar modelo
7. Integrar sampling neural
8. Evaluar resultados
```

Comandos esperados:

```powershell
python scripts/check_env.py
python scripts/test_bins.py
python scripts/render_baseline.py --config configs/cornell_baseline.yaml
python scripts/generate_dataset.py --config configs/cornell_dataset.yaml
python scripts/train_mlp.py --config configs/cornell_train.yaml
python scripts/export_model.py --config configs/export_model.yaml
python scripts/render_neural.py --config configs/cornell_neural_render.yaml
python scripts/evaluate.py --config configs/evaluation.yaml
```

