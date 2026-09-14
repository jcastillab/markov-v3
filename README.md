# Markov Freedom v10

Pronostico de corte comercial de rosas Freedom (ALMER, LA PRADERA, SANTA HELENA).
Pipeline reproducible por fases, proyeccion operacional semanal idempotente y
dashboard unificado de evaluacion historica y pronostico futuro.

## Estructura

```text
config/pipeline.yaml       Configuracion central (rutas, fuentes, parametros)
data/raw/                  Fuentes historicas inmutables (xlsx)
data/external/scoring/     Entrada externa solo para evaluacion
data/vision/Resultados/    CSV de vision por semana (S36, S37, ...)
src/
  canonical.py             Capa canonica (ingestion y homologacion)
  models/                  Implementaciones internas (M3, RF, GLM, Bayes)
  modeling/                Paquete portable: contratos, bundles y adaptadores
  dashboard_streamlit.py   Dashboard
  dashboard_validation.py  Validacion de poblaciones y corridas
scripts/                   Orquestacion productiva
tests/                     pytest
docs/                      Contratos, arquitectura y guias
outputs/                   Generado (gitignored)
```

## Quickstart

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Pipeline historico (fases 0-8)

```powershell
.\.venv\Scripts\python.exe scripts\run_historical.py
```

### Entrenar y congelar modelos

```powershell
.\.venv\Scripts\python.exe scripts\train_models_cmd.py
```

### Proyectar la semana siguiente

```powershell
.\.venv\Scripts\python.exe scripts\run_operational.py
```

### Validar corrida e integridad

```powershell
.\.venv\Scripts\python.exe scripts\validate_run.py
```

### Todo de una vez

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\dashboard_streamlit.py
```

Pestanas:

- **Resumen semanal**: metricas y graficas de la validacion historica.
- **Nueva entrada**: semanas con corte real disponible (contrastable).
- **Pronostico futuro**: semana siguiente, mejor modelo por familia, con
  total global o detalle por bloque y filtros por modelo y finca.

## Modelos

Los mejores modelos causales de cada familia se congelan como bundles
content-addressed en `outputs/models/bundles`:

| Familia | Modelo operacional |
|---|---|
| M3 | `E00_M3_BASE_OPERATIONAL` |
| RF | `RF_OPT_FENO_OPERATIONAL` |
| RF_HORIZON | `RF_H1_H7_FENO_OPERATIONAL` |
| GLM_NB | `GLM_NB_FENO_OPERATIONAL` |
| BAYES_NB | `NB_JERARQUICO_OPERATIONAL` |
| BAYES_DIRICHLET | `M3_DIRICHLET_MULTINOMIAL_OPERATIONAL` |

Contrato portable en `src/modeling/` (adaptadores, bundle y registro) para
integrar a otro proyecto.

## Datos

Consulte `docs/GUIA_DATOS_ENTRADA.md` para saber donde vive cada dato, el
formato aceptado y como agregar una semana nueva.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Reglas

- M3 es el baseline obligatorio.
- Cero leakage: las features de un origen nunca usan informacion posterior.
- Los modelos retrospectivos (P32) se etiquetan y nunca se mezclan con los causales.
- Toda corrida operacional es inmutable y proyecta exactamente una semana.
- Reejecutar la misma semana reutiliza la corrida (idempotente).