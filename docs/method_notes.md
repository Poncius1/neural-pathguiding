# Notas del método matemático

## Propósito del documento

Este documento resume el método matemático del proyecto: **muestreo direccional neural discreto para reducción de varianza en Path Tracing físicamente basado**.

El objetivo es construir una herramienta auxiliar de muestreo. La red neuronal no reemplaza el integrador físico, no genera la imagen final y no funciona como denoiser. Su función es producir una distribución de probabilidad sobre direcciones del hemisferio local para guiar el muestreo de caminos.

---

# 1. Problema general

En Path Tracing se estima la ecuación de renderizado mediante integración Monte Carlo.

Para un punto de sombreado `x` y una dirección de salida `ωo`, la radiancia saliente puede expresarse como:

$$
L_o(x, \omega_o) =
\int_{\mathcal{H}^2}
L_i(x, \omega_i)
f_s(x, \omega_o, \omega_i)
\cos\theta_i
, d\omega_i
$$

donde:

* `x` es el punto de sombreado.
* `ωo` es la dirección de salida o vista.
* `ωi` es una dirección entrante del hemisferio.
* `Li(x, ωi)` es la radiancia incidente desde `ωi`.
* `fs(x, ωo, ωi)` es la BSDF.
* `cosθi` es el coseno entre `ωi` y la normal de la superficie.
* `dωi` es la medida de ángulo sólido.
* `H²` es el hemisferio local superior.

El integrando principal es:

$$
g(x, \omega_i) =
L_i(x, \omega_i)
f_s(x, \omega_o, \omega_i)
\cos\theta_i
$$

El objetivo del muestreo de importancia es elegir una PDF `p(ω | x)` que se parezca a la forma del integrando. Mientras mejor coincida la PDF con las regiones de alta contribución, menor será la varianza del estimador.

---

# 2. Estimador Monte Carlo

Si se samplea una dirección `ω` desde una PDF `p(ω | x)`, el estimador de una muestra es:

$$
\hat{L}_o(x, \omega_o) =
\frac{
L_i(x, \omega)
f_s(x, \omega_o, \omega)
\cos\theta
}{
p(\omega | x)
}
$$

Para `N` muestras independientes:

$$
\hat{L}*o(x, \omega_o) =
\frac{1}{N}
\sum*{k=1}^{N}
\frac{
L_i(x, \omega_k)
f_s(x, \omega_o, \omega_k)
\cos\theta_k
}{
p(\omega_k | x)
}
$$

El estimador es no sesgado si:

1. La PDF es positiva donde el integrando puede ser distinto de cero.
2. La PDF se evalúa correctamente en la misma medida usada por el integrando.
3. La contribución se divide entre la PDF real con la que se generó la muestra.

En este proyecto la medida será **ángulo sólido**.

---

# 3. Idea central del método

El método propone usar una MLP ligera para producir una distribución discreta sobre bins direccionales del hemisferio local.

Flujo conceptual:

```text
Punto de sombreado x
        ↓
features locales
        ↓
MLP
        ↓
probabilidades por bin
        ↓
selección de bin
        ↓
dirección continua dentro del bin
        ↓
evaluación Path Tracing normal
        ↓
PDF explícita
```

La red neuronal produce:

$$
P_\theta(b | x)
$$

donde:

* `b` es un bin direccional.
* `x` es el punto de sombreado.
* `θ` son los parámetros de la red.
* `Pθ(b | x)` es una probabilidad discreta.

Luego se samplea una dirección continua `ω` uniformemente dentro del bin seleccionado.

---

# 4. Discretización del hemisferio

El hemisferio local se representa usando:

$$
\mu = \cos\theta
$$

con:

$$
\mu \in [0, 1]
$$

y:

$$
\phi \in [0, 2\pi)
$$

Se divide el dominio en:

```text
n_mu × n_phi = N bins
```

Ejemplos iniciales:

```text
n_mu = 4, n_phi = 8  →  N = 32 bins
n_mu = 8, n_phi = 8  →  N = 64 bins
```

Para un bin con índices `(i, j)`:

$$
\mu_0 = \frac{i}{n_\mu}
$$

$$
\mu_1 = \frac{i + 1}{n_\mu}
$$

$$
\phi_0 = \frac{2\pi j}{n_\phi}
$$

$$
\phi_1 = \frac{2\pi (j + 1)}{n_\phi}
$$

---

# 5. Ángulo sólido del bin

La medida de ángulo sólido en coordenadas esféricas puede escribirse como:

$$
d\omega = d\phi , d\mu
$$

Por lo tanto, el ángulo sólido de un bin es:

$$
\Omega_b =
(\phi_1 - \phi_0)
(\mu_1 - \mu_0)
$$

Como la división es uniforme en `μ` y `φ`, todos los bins tienen el mismo ángulo sólido:

$$
\Omega_b =
\frac{2\pi}{N}
$$

Esta decisión simplifica la implementación inicial y reduce el riesgo de errores en la PDF.

---

# 6. Sampleo uniforme dentro de un bin

Una vez elegido un bin `b`, se generan dos números aleatorios:

$$
u_1, u_2 \in [0, 1)
$$

Luego:

$$
\mu =
\mu_0 + u_1(\mu_1 - \mu_0)
$$

$$
\phi =
\phi_0 + u_2(\phi_1 - \phi_0)
$$

La dirección local se reconstruye como:

$$
\sin\theta = \sqrt{1 - \mu^2}
$$

$$
\omega =
(
\cos\phi \sin\theta,
\sin\phi \sin\theta,
\mu
)
$$

Convención local:

```text
+Z = normal de la superficie
omega.z = cos(theta)
```

---

# 7. PDF neural explícita

La parte central del método es que la PDF neural sea evaluable.

Si la red predice una probabilidad discreta para cada bin:

$$
P_\theta(b | x)
$$

y la dirección continua se samplea uniformemente dentro del bin, entonces la PDF en ángulo sólido es:

$$
p_\theta(\omega | x) =
\frac{P_\theta(b | x)}{\Omega_b}
$$

donde `b` es el bin que contiene la dirección `ω`.

Como:

$$
\Omega_b = \frac{2\pi}{N}
$$

entonces:

$$
p_\theta(\omega | x) =
P_\theta(b | x)
\frac{N}{2\pi}
$$

Esta PDF es la que debe usarse en el denominador del estimador Monte Carlo.

---

# 8. Soporte completo

Para evitar sesgo, la PDF no debe ser cero en regiones donde el integrando pueda tener contribución.

La MLP produce probabilidades usando Softmax, pero por estabilidad se usará una mezcla con una distribución uniforme:

$$
P_{mix}(b | x) =
(1 - \alpha) P_\theta(b | x)
+
\alpha \frac{1}{N}
$$

donde:

```text
alpha = 0.02 o 0.05
```

Esto garantiza que:

$$
P_{mix}(b | x) > 0
$$

para todo bin `b`.

La PDF usada en render debe ser:

$$
p_{mix}(\omega | x) =
\frac{P_{mix}(b | x)}{\Omega_b}
$$

---

# 9. Features de entrada

La MLP recibirá atributos locales del punto de sombreado.

Features iniciales propuestas:

```text
posición 3D del punto:      3
normal de superficie:       3
dirección de salida:        3
roughness:                  1
albedo/base color:          3
profundidad de rebote:      1
```

Total inicial:

```text
14 features
```

Estas features pueden cambiar durante la investigación. La primera versión debe priorizar simplicidad y facilidad de depuración.

---

# 10. Salida de la MLP

La salida de la red será un vector de logits:

$$
z \in \mathbb{R}^{N}
$$

donde `N` es el número de bins direccionales.

Luego se aplica Softmax:

$$
P_\theta(b | x) =
\frac{
e^{z_b}
}{
\sum_j e^{z_j}
}
$$

Después se aplica mezcla uniforme:

$$
P_{mix}(b | x) =
(1 - \alpha) P_\theta(b | x)
+
\alpha \frac{1}{N}
$$

---

# 11. Arquitectura inicial de la MLP

Arquitectura base:

```text
Input → Linear → ReLU → Linear → ReLU → Linear → Softmax
```

Ejemplo inicial:

```text
14 features → 64 hidden → 64 hidden → 32 bins
```

o:

```text
14 features → 64 hidden → 64 hidden → 64 bins
```

Primero se trabajará con `32 bins` para reducir complejidad.

---

# 12. Target de entrenamiento

El target ideal no es solamente “dónde hay luz”, sino dónde hay alta contribución al integrando de renderizado.

El target conceptual es:

$$
T(x, \omega) =
L_i(x, \omega)
f_s(x, \omega_o, \omega)
\cos\theta
$$

Para entrenamiento supervisado offline:

1. Se generan muchos puntos de sombreado.
2. Para cada punto se samplean direcciones en el hemisferio.
3. Se estima la contribución de cada dirección.
4. Se acumulan contribuciones por bin.
5. Se normalizan las contribuciones para obtener una distribución objetivo.

Para cada bin `b`:

$$
C_b =
\sum_{k \in b}
T(x, \omega_k)
$$

Luego:

$$
P_{target}(b | x) =
\frac{C_b}{\sum_j C_j}
$$

Si la suma total es cero o muy pequeña, se puede descartar el punto o usar una distribución uniforme.

---

# 13. Dataset offline

Cada muestra del dataset debe contener:

```text
features: vector de entrada de la MLP
target: distribución objetivo sobre bins
metadata: información útil para reproducibilidad
```

Formato inicial esperado:

```text
features: [num_samples, feature_dim]
targets:  [num_samples, num_bins]
```

Metadata útil:

```text
scene
n_bins
n_mu
n_phi
max_depth
samples_per_bin
feature_names
generation_date
```

El dataset se guardará localmente en `data/` y no debe subirse a GitHub.

---

# 14. Función de pérdida

Como el target es una distribución discreta, la pérdida inicial será cross entropy con soft labels:

$$
\mathcal{L} =
-------------

\sum_b
P_{target}(b | x)
\log P_\theta(b | x)
$$

También puede interpretarse como minimizar la divergencia KL entre la distribución objetivo y la distribución predicha, ignorando el término constante del target.

---

# 15. Uso durante render

Durante el render, en cada punto de sombreado no especular:

1. Se extraen features del punto.
2. La MLP predice probabilidades por bin.
3. Se mezcla con uniforme.
4. Se samplea un bin.
5. Se samplea una dirección dentro del bin.
6. Se evalúa la PDF.
7. Se continúa el Path Tracing normalmente.

Pseudocódigo:

```text
features = extract_features(surface_interaction)

probs = mlp(features)
probs = mix_with_uniform(probs, alpha)

bin_id = sample_discrete(probs)
omega_local = sample_uniform_direction_in_bin(bin_id)

pdf = probs[bin_id] / solid_angle(bin_id)

contribution = Li * fs * cos_theta / pdf
```

---

# 16. MIS inicial

La primera integración con MIS debe mantenerse simple.

La combinación inicial recomendada es una mezcla entre:

```text
BSDF sampling
Neural directional sampling
```

Se elige una estrategia con probabilidad fija:

```text
q_bsdf = 0.5
q_neural = 0.5
```

Si se samplea una dirección `ω`, la PDF de la mezcla es:

$$
p_{mix}(\omega) =
q_{bsdf} p_{bsdf}(\omega)
+
q_{neural} p_{neural}(\omega)
$$

La contribución se evalúa como:

$$
\frac{
L_i f_s \cos\theta
}{
p_{mix}(\omega)
}
$$

Esto evita implementar MIS completo desde el primer prototipo y permite validar el comportamiento del método paso a paso.

---

# 17. Validación matemática

Antes de usar Mitsuba, deben validarse integrales simples sobre el hemisferio.

## Integral constante

La integral de `1` sobre el hemisferio es:

$$
\int_{\mathcal{H}^2} 1 , d\omega = 2\pi
$$

Estimador:

$$
\frac{1}{N}
\sum_k
\frac{1}{p(\omega_k)}
$$

Debe aproximar `2π`.

## Integral coseno

La integral de `cosθ` sobre el hemisferio es:

$$
\int_{\mathcal{H}^2} \cos\theta , d\omega = \pi
$$

Estimador:

$$
\frac{1}{N}
\sum_k
\frac{\cos\theta_k}{p(\omega_k)}
$$

Debe aproximar `π`.

Estas pruebas validan:

* sampleo correcto.
* PDF correcta.
* medida correcta.
* consistencia entre bin discreto y dirección continua.

---

# 18. Validación contra rendering

Después de validar el núcleo matemático, se validará en rendering.

Pruebas iniciales:

1. Escena simple difusa.
2. Ground truth con muchos samples por píxel.
3. Render con BSDF sampling.
4. Render con neural sampling.
5. Comparación de error.

Métricas:

```text
MSE
RMSE
PSNR
tiempo de render
samples por píxel
```

La primera meta no es que el método neural gane, sino demostrar que converge al mismo ground truth.

---

# 19. Condiciones para evitar sesgo

El método debe cumplir:

1. La PDF usada para samplear debe ser la misma que se usa para dividir la contribución.
2. La PDF debe estar expresada en ángulo sólido.
3. La PDF debe ser positiva donde el integrando pueda ser distinto de cero.
4. Las probabilidades discretas deben sumar uno.
5. La dirección sampleada debe pertenecer al bin seleccionado.
6. Si se usa MIS, todas las PDFs relevantes deben evaluarse en la misma medida.

Una red mala puede aumentar la varianza, pero no debe cambiar el valor esperado del estimador.

---

# 20. Alcance inicial

El primer prototipo debe limitarse a:

```text
escena simple
32 bins
MLP pequeña
materiales difusos o rough simples
profundidad de rebote baja
dataset offline supervisado
validación con ground truth
```

No se implementará al inicio:

```text
ReSTIR
tiempo real
Unreal Engine
Unity
volúmenes
caustics
materiales delta complejos
normalizing flows
denoising neural
```

---

# 21. Orden de implementación recomendado

```text
1. core/bins.py
2. tests/test_bins.py
3. core/pdf.py
4. tests/test_pdf.py
5. core/sampling.py
6. tests/test_sampling.py
7. core/features.py
8. renderers/mitsuba/baseline_renderer.py
9. renderers/mitsuba/dataset_generator.py
10. training/model.py
11. training/train.py
12. core/mlp_runtime.py
13. renderers/mitsuba/neural_integrator.py
14. evaluation/metrics.py
```

---

# 22. Criterio de éxito inicial

El primer bloque del proyecto será exitoso cuando:

* Los bins cubran correctamente el hemisferio.
* La PDF neural discreta sea evaluable.
* Las pruebas de integrales básicas pasen.
* La distribución mezclada con uniforme tenga soporte completo.
* El código esté documentado y testeado.

Después de eso se podrá avanzar hacia Mitsuba y generación de dataset.
