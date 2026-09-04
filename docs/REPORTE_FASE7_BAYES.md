# Reporte Fase 7 - Modelos bayesianos

## Modelos

- `M3_DIRICHLET_MULTINOMIAL`: posterior conjugado por estado de origen,
  centrado en M3 causal y con draws de matrices de transición.
- `NB_JERARQUICO`: pooling Gamma-Poisson por finca, bloque y horizonte,
  encogido hacia la media global.
- No se uso PyMC: no esta instalado y los modelos conjugados cubren el primer
  bloque bayesiano sin añadir una dependencia innecesaria.

## Validacion

La poblacion de comparacion de esta fase contiene 1.295 dias.

| Modelo | WAPE | Cobertura 80% | Cobertura 95% | Estado |
|---|---:|---:|---:|---|
| M3 Dirichlet-Multinomial | 46,73% | 26,5% | 33,4% | No promovido |
| NB jerarquico | 35,79% | 5,0% | 7,4% | Challenger exploratorio |
| NB jerarquico con covariables | 47,56% | 34,7% | 50,3% | No promovido |

El NB jerarquico mejora el WAPE, pero sus intervalos tienen cobertura inferior
a la nominal. La parametrizacion de incertidumbre requiere calibracion antes
de usarla para decisiones operativas. El posterior Dirichlet conserva la
simplex de cada fila y aporta regularizacion, pero no mejora el baseline.

## Artefactos

- `dirichlet_posterior_summary.csv`
- `metrics_fase7_bayes.csv`
- `bayes_manifest.json`

Siguiente paso: Fase 8, comparacion final rolling origin, intervalos,
incertidumbre y seleccion formal de champion/challengers.
