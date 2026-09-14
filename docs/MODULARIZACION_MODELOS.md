# Modularizacion de modelos

La frontera portable queda separada en `src/modeling/`:

- `extrapolation.py`: escala validada de camas muestreadas a bloque.
- `contracts.py`: interfaz minima `fit/predict` y normalizacion de predicciones.

Los modelos actuales conservan sus implementaciones y datasets en `src/models/`
para no cambiar sus resultados durante esta fase. El siguiente traslado puede
adaptar cada estimador al contrato `Regressor`, mientras el proyecto nuevo
provee solamente un frame con features numericas y la configuracion.

Reglas del contrato:

- El target de scoring es siempre el total del bloque, no el conteo de la muestra.
- La extrapolacion se calcula antes del entrenamiento/scoring y nunca se imputa
  silenciosamente.
- Las salidas son no negativas y conservan las claves `finca`, `bloque`,
  `fecha_origen`, `fecha_objetivo`, `semana_proyeccion` y `horizonte_dia`.
