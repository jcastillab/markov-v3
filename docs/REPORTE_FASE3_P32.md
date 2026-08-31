# Reporte Fase 3 - P32 y Semi Markov

## Poblacion y causalidad

- Observaciones P32 normalizadas: 1.297.
- Intervalos candidatos diarios: 1.097.
- Intervalos validos bajo el contrato vigente: 514.
- Validation comun: 714 dias, la misma poblacion usada por M3 en Fase 2.
- Los experimentos P32 usan seguimiento de agosto de 2026 para ventanas
  anteriores; todos se etiquetan `RETROSPECTIVO_ORACLE_NO_CAUSAL`.

## QA de transiciones

| Resultado | Intervalos |
|---|---:|
| Validos | 514 |
| Codigo pendiente SP | 194 |
| Otros codigos pendientes | 9 |
| Transicion no canonica | 380 |

`SP` aislado no se transformo a `SP1`. Se conserva como
del ajuste hasta confirmacion agronomica.

## Resultados

| Experimento | WAPE | MAE | RMSE | Sesgo | Estado |
|---|---:|---:|---:|---:|---|
| E00 M3 BASE | 55,52% | 1.122,89 | 1.503,68 | -37,34% | Causal |
| E02 M3 P32 RAW | 34,80% | 703,84 | 960,88 | -26,99% | Retrospectivo |
| E03 M3 P32 CANONICO | 34,80% | 703,84 | 960,88 | -26,99% | Retrospectivo |
| E04 M3 P32 REG | 34,53% | 698,31 | 953,83 | -26,30% | Retrospectivo |
| E05 Semi Markov RC | 54,82% | 1.108,81 | 1.490,78 | -39,38% | Retrospectivo |
| E06 Semi Markov RC+SS | 54,77% | 1.107,85 | 1.493,48 | -30,29% | Retrospectivo |
| E07 Semi Markov RC+SS+AP | 37,96% | 767,80 | 1.096,42 | +4,35% | Retrospectivo |

## Interpretacion

1. P32 regularizado mejora el benchmark M3 en esta comparacion, pero no es
   evidencia operacional porque P32 fue levantado despues de las ventanas.
2. El suavizado hacia M3 produce una mejora pequena y reduce el sesgo.
3. La variante Semi Markov completa es la unica cercana a M3-P32, pero queda
   por detras de M3-P32_REG.
4. Las variantes Semi Markov parciales degradan fuertemente. No se promocionan.
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
