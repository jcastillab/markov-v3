# Reporte Fase 8 - Comparacion final

## Poblacion comun

El ranking primario usa `ROLLING_ORIGIN_COMMON` causal con 441 observaciones diarias comunes entre modelos.
Los experimentos retrospectivos P32 y las poblaciones o escalas distintas quedan excluidos.

## Ranking

| Modelo | WAPE | IC bootstrap 95% | Decision |
|---|---:|---:|---|
| RF_FENO_PODA_CLIMA_n200_d3_l5_s20_fsqrt_poisson | 30.57% | 27.35%-34.31% | champion_provisional |
| E00_M3_BASE_ROLLING | 57.59% | 49.65%-64.89% | baseline_obligatorio |

## Decision

`RF_FENO_PODA_CLIMA_n200_d3_l5_s20_fsqrt_poisson` es el champion provisional bajo rolling-origin causal.
M3 permanece como baseline obligatorio y referencia mecanistica.
La promocion operacional requiere un periodo futuro congelado independiente.

Artefactos: `ranking_final.csv`, `metrics_comparacion_final.csv`, `champion_manifest.json` y `selected_model_manifest.json`.
