# Reporte Fase 6 - GLM NB y Random Forest

## Dataset y causalidad

- Se construyo `dataset_supervisado_diario` con una fila por finca, bloque,
  origen y horizonte diario.
- El split temporal se define sobre todos los origenes y luego se restringe a
  ventanas evaluables, igual que el baseline M3.
- Historial de cortes, poda, clima y prediccion M3 usan informacion disponible
  hasta `t0`.
- La validacion contiene 714 dias.

## Resultados

| Experimento | WAPE | Estado |
|---|---:|---|
| RF diario pooled FENO | 27,64% | Challenger exploratorio |
| RF diario H1-H7 FENO | 27,54% | Challenger exploratorio |
| RF semanal FENO | 18,59% | Challenger exploratorio |
| RF residual sobre M3 | 66,95% | No promovido |
| GLM NB FENO | 343,02% | No promovido |
| GLM NB FENO_PODA_CLIMA | 71,27% | No promovido |

El RF mejora ampliamente a M3 en esta validacion, pero requiere validacion
interna, importancia por permutacion y control de estabilidad por finca,
bloque y periodo antes de promoción. El resultado semanal no debe compararse
directamente con WAPE diario sin reportar ambas escalas.

El GLM NB se incluye como control de conteo, pero sus predicciones presentan
inestabilidad con la escala y colinealidad actuales. No se usa como champion.
El residual sobre M3 tampoco mejora, por lo que M3 permanece como referencia
mecanistica obligatoria.

## Artefactos

- `dataset_supervisado_diario.parquet`
- `metrics_fase6_supervisado.csv`
- `importance_*.csv`
- `supervised_manifest.json`

Siguiente paso recomendado: Fase 7, modelos bayesianos, conservando M3 y RF
como referencias y revisando el GLM NB antes de la comparación final.
