# Reporte Fase 8 - Comparacion final

## Poblacion comun

El ranking primario usa `ROLLING_ORIGIN_COMMON` causal con 231 observaciones diarias comunes entre modelos.
Los experimentos retrospectivos P32 y las poblaciones o escalas distintas quedan excluidos.

## Ranking

| Modelo | WAPE | IC bootstrap 95% | Decision |
|---|---:|---:|---|
| RF_OPT_FENO_PODA_ROLLING | 23.34% | 19.73%-27.61% | champion_provisional |
| NB_JERARQUICO_ROLLING | 23.36% | 19.04%-28.61% | challenger |
| RF_OPT_FENO_ROLLING | 23.68% | 20.71%-27.50% | challenger |
| MODELO_SELECCIONADO_ROLLING | 24.18% | 20.27%-28.83% | challenger |
| RF_OPT_FENO_PODA_CLIMA_ROLLING | 24.51% | 20.55%-29.43% | challenger |
| RF_H1_H7_FENO_ROLLING | 26.49% | 21.21%-33.66% | challenger |
| RF_OPT_FENO_CLIMA_ROLLING | 26.52% | 22.04%-32.86% | challenger |
| GLM_NB_FENO_CLIMA_ROLLING | 27.56% | 21.99%-34.99% | challenger |
| GLM_NB_FENO_ROLLING | 27.77% | 21.71%-36.22% | challenger |
| GLM_NB_FENO_PODA_CLIMA_ROLLING | 29.04% | 23.00%-36.74% | challenger |
| GLM_NB_FENO_PODA_ROLLING | 29.44% | 23.11%-37.55% | challenger |
| NB_JERARQUICO_COVARIABLES_ROLLING | 46.09% | 26.72%-73.00% | challenger |
| RF_RESIDUAL_M3_FENO_ROLLING | 66.54% | 56.66%-75.08% | challenger |
| M3_DIRICHLET_MULTINOMIAL_ROLLING | 73.06% | 62.51%-82.05% | challenger |
| E01_M3_INGRESO_CALIBRADO_ROLLING | 74.04% | 62.79%-83.42% | challenger |
| E00_M3_BASE_ROLLING | 74.17% | 63.03%-83.40% | baseline_obligatorio |

## Exclusiones del rolling

- `P32_SEMIMARKOV`: RETROSPECTIVO_ORACLE_NO_CAUSAL.
- `M3_PODA_CLIMA`: coeficientes seleccionados en validacion fija; requiere nested rolling.

## Decision

`RF_OPT_FENO_PODA_ROLLING` es el champion provisional bajo rolling-origin causal.
M3 permanece como baseline obligatorio y referencia mecanistica.
La promocion operacional requiere un periodo futuro congelado independiente.

Artefactos: `ranking_final.csv`, `metrics_comparacion_final.csv`, `champion_manifest.json` y `selected_model_manifest.json`.
