# Reporte Fase 8 - Comparacion final

## Poblacion comun

El ranking primario usa `ROLLING_ORIGIN_COMMON` causal con 728 observaciones diarias comunes entre modelos.
Los experimentos retrospectivos P32 y las poblaciones o escalas distintas quedan excluidos.

## Ranking

| Modelo | WAPE | IC bootstrap 95% | Decision |
|---|---:|---:|---|
| RF_OPT_FENO_ROLLING | 25.12% | 23.42%-27.23% | champion_provisional |
| NB_JERARQUICO_ROLLING | 26.47% | 24.33%-28.88% | challenger |
| RF_H1_H7_FENO_ROLLING | 26.57% | 24.45%-29.12% | challenger |
| GLM_NB_FENO_ROLLING | 26.69% | 24.72%-28.94% | challenger |
| RF_OPT_FENO_CLIMA_ROLLING | 26.91% | 24.87%-29.17% | challenger |
| GLM_NB_FENO_CLIMA_ROLLING | 28.76% | 27.17%-30.53% | challenger |
| RF_OPT_FENO_PODA_CLIMA_ROLLING | 28.92% | 26.48%-31.62% | challenger |
| MODELO_SELECCIONADO_ROLLING | 28.92% | 26.48%-31.62% | challenger |
| GLM_NB_FENO_PODA_CLIMA_ROLLING | 29.83% | 26.55%-33.65% | challenger |
| GLM_NB_FENO_PODA_ROLLING | 31.97% | 27.14%-37.74% | challenger |
| RF_OPT_FENO_PODA_ROLLING | 32.05% | 29.23%-35.36% | challenger |
| RF_RESIDUAL_M3_FENO_ROLLING | 32.07% | 29.67%-34.88% | challenger |
| E00_M3_BASE_ROLLING | 33.70% | 31.00%-36.86% | baseline_obligatorio |
| E01_M3_INGRESO_CALIBRADO_ROLLING | 33.72% | 31.15%-36.55% | challenger |
| M3_DIRICHLET_MULTINOMIAL_ROLLING | 34.66% | 31.85%-37.97% | challenger |
| NB_JERARQUICO_COVARIABLES_ROLLING | 39.01% | 28.31%-52.07% | challenger |

## Exclusiones del rolling

- `P32_SEMIMARKOV`: RETROSPECTIVO_ORACLE_NO_CAUSAL.
- `M3_PODA_CLIMA`: coeficientes seleccionados en validacion fija; requiere nested rolling.

## Decision

`RF_OPT_FENO_ROLLING` es el champion provisional bajo rolling-origin causal.
M3 permanece como baseline obligatorio y referencia mecanistica.
La promocion operacional requiere un periodo futuro congelado independiente.

Artefactos: `ranking_final.csv`, `metrics_comparacion_final.csv`, `champion_manifest.json` y `selected_model_manifest.json`.
