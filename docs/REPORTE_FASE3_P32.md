# Reporte Fase 3 - P32 y Semi Markov

## Poblacion y causalidad

- Observaciones P32 normalizadas: 1.297.
- Intervalos candidatos diarios: 1.097.
- Intervalos validos bajo el contrato vigente: 514.
- Validacion comun: 1.295 dias, la misma poblacion usada por M3 en Fase 2.
- Los experimentos P32 usan seguimiento de agosto de 2026 para ventanas
  anteriores; todos se etiquetan `RETROSPECTIVO_ORACLE_NO_CAUSAL`.

## QA de transiciones

| Resultado | Intervalos |
|---|---:|
| Validos | 514 |
| Codigo pendiente SP | 194 |
| Otros codigos pendientes | 9 |
| Transicion no canonica | 380 |

`SP` aislado no se transformo a `SP1`. Se conserva como codigo pendiente y se
excluye del ajuste hasta confirmacion agronomica.

## Resultados

| Experimento | WAPE | MAE | RMSE | Sesgo | Estado |
|---|---:|---:|---:|---:|---|
| E00 M3 BASE | 45,94% | 1.062,22 | 1.544,25 | -11,41% | Causal |
| E02 M3 P32 RAW | 39,69% | 917,87 | 1.337,14 | -24,11% | Retrospectivo |
| E03 M3 P32 CANONICO | 39,69% | 917,87 | 1.337,14 | -24,11% | Retrospectivo |
| E04 M3 P32 REG | 39,61% | 915,87 | 1.335,33 | -23,80% | Retrospectivo |
| E05 Semi Markov RC | 46,49% | 1.074,95 | 1.573,11 | -8,02% | Retrospectivo |
| E06 Semi Markov RC+SS | 41,97% | 970,40 | 1.514,94 | +7,42% | Retrospectivo |
| E07 Semi Markov RC+SS+AP | 44,38% | 1.026,19 | 1.604,07 | +8,53% | Retrospectivo |

## Interpretacion

1. P32 regularizado mejora el benchmark M3 en esta comparacion, pero no es
   evidencia operacional porque P32 fue levantado despues de las ventanas.
2. El suavizado hacia M3 produce una mejora pequena y reduce el sesgo.
3. La variante Semi Markov RC+SS es la mejor de esa familia, pero queda por
   detras de M3-P32 regularizado.
4. Ninguna variante retrospectiva se promociona.
5. La edad inicial ya no se concentra en cero: `x0` se reparte mediante
   `P(edad|estado)` estimada de exposiciones consecutivas. Aun no es un
   posterior de edad latente plenamente identificado; queda como siguiente
   mejora metodologica.

## Artefactos

- `transition_intervals_p32.parquet`
- `observaciones_p32.parquet`
- `qa_p32_transiciones.csv`
- `p32_age_distributions.csv`
- `p32_hazards.csv`
- `metrics_fase3_p32.csv`
- `p32_manifest.json`
