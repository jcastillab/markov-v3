# Reporte Fase 8 - Comparacion final

## Poblacion comun

El ranking primario usa la validacion rolling-origin causal diaria con 714
observaciones. Se excluyen del ranking los experimentos P32 retrospectivos,
el clima restringido a 476 observaciones y el resultado semanal de 102
observaciones.

## Ranking

| Modelo | WAPE | IC bootstrap 95% | Decision |
|---|---:|---:|---|
| RF H1-H7 FENO | 27,54% | 25,71%-29,50% | Champion provisional |
| RF diario pooled FENO | 27,64% | 25,83%-29,56% | Challenger |
| NB jerarquico | 35,61% | 33,41%-37,78% | Challenger |
| M3 Dirichlet-Multinomial | 54,21% | n/d | Challenger |
| M3 BASE | 55,52% | n/d | Baseline obligatorio |

El champion es provisional porque la seleccion se realiza sobre el holdout
final disponible y no existe un tercer periodo temporal independiente. Antes
de promoción operacional se requiere un backtest adicional congelado y
validacion por finca, bloque y horizonte.

## Incertidumbre

Los intervalos bootstrap se calcularon sobre predicciones individuales con
1.000 replicaciones. Los modelos sin archivo de predicciones individual no
reciben un intervalo artificial; quedan marcados como `n/d`.

Los intervalos predictivos bayesianos de Fase 7 tienen cobertura inferior a la
nominal, por lo que Bayes no se promueve como proveedor de incertidumbre.

## Decision

- `RF_H1_H7_FENO` es el challenger con mejor desempeño puntual y queda como
  champion provisional de investigación.
- `M3` permanece como baseline obligatorio y referencia mecanistica.
- No se mezclan métricas `RETROSPECTIVE_ORACLE_NO_CAUSAL` con el ranking.
- Podas, clima, GLM NB y residual M3 quedan como challengers documentados.

Artefactos principales: `ranking_final.csv`, `metrics_comparacion_final.csv` y
`champion_manifest.json`.
