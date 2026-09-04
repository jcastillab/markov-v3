# Reporte Fase 4 - Podas

## Implementacion

- Se normalizaron las operaciones `CORTE` y `ALINEAMIENTO` usando `Cantidad`.
- `ESTIMADO` queda fuera de las features.
- Se construyeron features por finca, bloque y origen con lags prioritarios de
  42 a 84 dias, sumas de 7 y 14 dias, kernel 8-12 semanas, dias desde la
  ultima operacion y acumulados de 28, 56 y 84 dias.
- Todas las features de un origen usan fechas de poda no posteriores a `t0`.

## Validacion

La poblacion es la misma validacion causal de Fase 2: 1.295 dias evaluables.
El baseline se reproduce con WAPE 45,94%.

| Experimento | Mejor WAPE | Coeficiente | Estado |
|---|---:|---:|---|
| M3_BASE | 45,94% | 0,00 | Baseline causal |
| M3_PODA_INGRESO | 45,94% | 0,00 | No promovido |
| M3_PODA_TRANSICION | 45,94% | 0,00 | No promovido |
| M3_PODA_HIBRIDO | 45,94% | 0,00 | No promovido |

Los coeficientes positivos empeoran el WAPE. No se agrega masa: el ingreso
esta acotado y las columnas de transición se renormalizan conservando la
contabilidad del sistema.

## Artefactos

- `poda_features.parquet`
- `poda_lag_screening.csv`
- `metrics_fase4_podas.csv`

La señal de poda queda documentada como challenger no promovido. El siguiente
paso recomendado es Fase 5 (clima), manteniendo M3 como baseline obligatorio.
