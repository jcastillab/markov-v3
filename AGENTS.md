# AGENTS.md - markov_v3

Plataforma experimental reproducible para pronóstico de corte comercial de
rosas Freedom (ALMER, LA PRADERA, SANTA HELENA). Reconstrucción greenfield
guiada por `docs/PROMPT_MAESTRO_MODELOS_PRONOSTICO_ROSAS.md` (especificación
completa) y `docs/ARQUITECTURA_MODELO.md` (auditoría del sistema previo, cuyos
8 bugs documentados deben evitarse por diseño).

## Reglas de oro

1. **M3 es el baseline obligatorio.** Ningún challenger gana por complejidad;
   gana por WAPE en la misma población de backtest causal (rolling origin).
2. **Cero leakage**: para origen `t0`, toda feature tiene timestamp máximo
   `<= t0`. Modelos retrospectivos se etiquetan
   `RETROSPECTIVO_ORACLE_NO_CAUSAL` y sus métricas nunca se mezclan.
3. **Todo parámetro vive en `config/pipeline.yaml`**, nunca hardcodeado
   (vigencias M3, ingreso RC, taus, lags, bases GDD, seeds).
4. **Commits al cerrar cada fase validada** (prompt sección 5.11).
5. Excel solo como entrada RAW y salida de auditoría; formato interno Parquet.
6. Códigos fenológicos pendientes (SP aislado, SP2 1/2, R4C, S1, 1P, 1 1/2P,
   2P, R6) rompen la trayectoria o van a QA; **nunca** se homologan en
   silencio. Ver `docs/CONTRATOS_DATOS.md`.

## Estructura

```text
config/pipeline.yaml     Configuración central (única fuente de parámetros)
data/raw/                12 Excel fuente (inmutables, con hash SHA-256)
docs/                    Prompt maestro, arquitectura auditada, contratos
src/                     Código del pipeline (por fases, ver roadmap)
tests/                   pytest (matemáticos, integración, leakage)
outputs/                 Generado (gitignored): data_quality, datasets,
                         models, predictions, evaluation, analysis, reports
```

## Entorno y comandos

```powershell
# Entorno: .venv (Python 3.12). Instalar deps:
.\.venv\Scripts\pip.exe install -r requirements.txt

# Fase 0 (auditoría de fuentes):
.\.venv\Scripts\python.exe src\fase0_auditoria.py   # inventario + hashes
.\.venv\Scripts\python.exe src\fase0_contratos.py   # sondas de contratos

# Tests
.\.venv\Scripts\python.exe -m pytest tests -q

# Git: commits en español, formato "fase-N: descripcion"
```

## Roadmap de fases (prompt sección 34)

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Auditoría de fuentes, contratos, homologaciones | **CERRADA** |
| 1 | Capa canónica: dims, fact_bloque_dia, forecast_windows | **CERRADA** |
| 2 | M3 baseline + calibración causal ingreso RC | **CERRADA** |
| 3 | P32: M3_P32 + Semi Markov corregido (x0 por edad) | **CERRADA** |
| 4 | Podas: análisis lags + M3+poda | **CERRADA** |
| 5 | Clima: VPD/GDD/DLI correctos + M3+clima (PRADERA) | pendiente |
| 6 | GLM NB + Random Forest + residual M3 | pendiente |
| 7 | Bayes: Dirichlet-Multinomial, NB jerárquico | pendiente |
| 8 | Comparación final + reporte + champion/challengers | pendiente |

## Convenciones de código

- snake_case en columnas canónicas: `Finca->finca`, `Bloque->bloque`,
  `Cantidad (conteos)->corte_comercial_real`, `Cantidad (podas)->poda_cantidad`.
- Bloques siempre como texto. Fincas canónicas: ALMER, LA PRADERA,
  SANTA HELENA (ver mapa en `docs/CONTRATOS_DATOS.md`).
- Ceros de corte son observaciones reales; no se eliminan.
- No forward-fill de conteos: transportar con `fecha_conteo_origen` +
  `dias_desde_conteo`.
- Métrica principal WAPE; nunca MAPE como principal (días con corte bajo).
