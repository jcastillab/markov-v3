# AGENTS.md - markov_v3

Plataforma reproducible para pronostico de corte comercial de rosas Freedom
(ALMER, LA PRADERA, SANTA HELENA). Pipeline por fases + proyeccion operacional
semanal idempotente + dashboard unificado.

## Reglas de oro

1. **M3 es el baseline obligatorio.** Ningun challenger gana por complejidad;
   gana por WAPE en la misma poblacion de backtest causal (rolling origin).
2. **Cero leakage**: para origen `t0`, toda feature tiene timestamp `<= t0`.
   Modelos retrospectivos se etiquetan `RETROSPECTIVO_ORACLE_NO_CAUSAL`.
3. **Todo parametro vive en `config/pipeline.yaml`**, nunca hardcodeado.
4. **Commits al cerrar cada fase validada**, formato `fase-N: descripcion`.
5. Excel solo como entrada RAW y salida de auditoria; formato interno Parquet.
6. Codigos fenologicos pendientes nunca se homologan en silencio.
7. **Toda corrida operacional es inmutable, idempotente y proyecta una sola
   semana siguiente** (`ONLY_NEXT_ISO_WEEK`).

## Estructura

```text
config/pipeline.yaml       Configuracion central
data/raw/                  Fuentes historicas inmutables (xlsx)
data/external/scoring/     Entrada externa solo para evaluacion
data/vision/Resultados/    CSV de vision por semana (S36, S37, ...)
src/canonical.py           Capa canonica
src/models/                Implementaciones internas
src/modeling/              Paquete portable: contratos, bundles, adaptadores
src/dashboard_streamlit.py Dashboard
src/dashboard_validation.py Validacion de poblaciones y corridas
scripts/                   Orquestacion productiva
tests/                     pytest
docs/                      Guias y contratos
legacy/                    Codigo retirado
archive/                   Archivos manuales y ZIP
outputs/                   Generado (gitignored)
```

## Entorno y comandos

```powershell
# Entorno: .venv (Python 3.12). Instalar deps:
.\.venv\Scripts\pip.exe install -r requirements.txt

# Pipeline historico completo (fases 0-8):
.\.venv\Scripts\python.exe scripts\run_historical.py

# Entrenar y congelar modelos (mejor por familia):
.\.venv\Scripts\python.exe scripts\train_models_cmd.py

# Proyectar la semana siguiente (idempotente):
.\.venv\Scripts\python.exe scripts\run_operational.py

# Validar corrida e integridad:
.\.venv\Scripts\python.exe scripts\validate_run.py

# Todo de una vez:
.\.venv\Scripts\python.exe scripts\run_pipeline.py

# Dashboard:
.\.venv\Scripts\python.exe -m streamlit run src\dashboard_streamlit.py

# Tests
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Datos

Consulte `docs/GUIA_DATOS_ENTRADA.md`: ubicacion de cada archivo, formato
aceptado y como agregar una semana nueva.

## Modelos operacionales (bundles en outputs/models/bundles)

| Familia | Modelo |
|---|---|
| M3 | `E00_M3_BASE_OPERATIONAL` |
| RF | `RF_OPT_FENO_OPERATIONAL` |
| RF_HORIZON | `RF_H1_H7_FENO_OPERATIONAL` |
| GLM_NB | `GLM_NB_FENO_OPERATIONAL` |
| BAYES_NB | `NB_JERARQUICO_OPERATIONAL` |
| BAYES_DIRICHLET | `M3_DIRICHLET_MULTINOMIAL_OPERATIONAL` |

## Convenciones de codigo

- snake_case en columnas canonicas (`Finca->finca`, `Bloque->bloque`).
- Bloques siempre texto. Fincas: ALMER, LA PRADERA, SANTA HELENA.
- Ceros de corte son observaciones reales; no se eliminan.
- No forward-fill de conteos: transportar con `fecha_conteo_origen`.
- Metrica principal WAPE; nunca MAPE como principal.