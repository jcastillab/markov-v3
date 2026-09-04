# Reporte Fase 6 - GLM NB y Random Forest

## Dataset y causalidad

- Se construyo `dataset_supervisado_diario` con una fila por finca, bloque,
  origen y horizonte diario.
- El split temporal se define sobre todos los origenes y luego se restringe a
  ventanas evaluables, igual que el baseline M3.
- Historial de cortes, poda, clima y prediccion M3 usan informacion disponible
  hasta `t0`.
- La validacion contiene 1.295 dias.

## Resultados

| Experimento | WAPE | Estado |
|---|---:|---|
| RF diario pooled FENO | 31,74% | Challenger exploratorio |
| RF diario H1-H7 FENO | 31,81% | Challenger exploratorio |
| RF semanal FENO | 25,24% | Otra escala |
| RF residual sobre M3 | 42,55% | No promovido |
| GLM NB FENO | 31,60% | Control interpretable |
| GLM NB FENO_PODA_CLIMA | 92,68% | No promovido |

El RF mejora ampliamente a M3 en esta validacion, pero requiere validacion
interna, importancia por permutacion y control de estabilidad por finca,
bloque y periodo antes de promoción. El resultado semanal no debe compararse
directamente con WAPE diario sin reportar ambas escalas.

El GLM NB FENO es competitivo en esta validacion fija, pero las variantes con
poda o clima presentan inestabilidad. No se usa como champion.
El residual sobre M3 tampoco mejora, por lo que M3 permanece como referencia
mecanistica obligatoria.

## Artefactos

- `dataset_supervisado_diario.parquet`
- `metrics_fase6_supervisado.csv`
- `importance_*.csv`
- `supervised_manifest.json`

Siguiente paso recomendado: Fase 7, modelos bayesianos, conservando M3 y RF
como referencias y revisando el GLM NB antes de la comparación final.
