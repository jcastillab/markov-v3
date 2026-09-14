# Guia de datos de entrada

Documento operativo que describe donde vive cada dato, que ingresar en cada
carpeta y el formato aceptado para que el pipeline corra.

## Estructura de datos

```text
data/
├── raw/                          # Fuentes historicas inmutables (entrenamiento)
│   ├── historico/                # Conteos, camas, plano y podas
│   ├── fenologia/                # Trayectorias fenologicas M3 y P32
│   ├── clima/                    # Estaciones meteorologicas horarias
│   └── auxiliares/               # Calendario y resumen de cortes
├── external/
│   └── scoring/                  # Archivo externo solo para evaluacion
└── vision/
    └── Resultados/
        └── SNN/                  # Una carpeta por semana (S36, S37, ...)
            └── <Finca>/
                └── <Bloque>/
                    └── *.csv     # Un CSV por cama muestreada
```

Todas las rutas se definen en `config/pipeline.yaml`:

- `paths.raw` -> `data/raw`
- `paths.external` -> `data/external`
- `paths.vision` -> `data/vision`

## 1. Historico (`data/raw/historico`)

| Archivo | Uso | Columnas obligatorias |
|---|---|---|
| `conteos_vs_cortes_multifinca.xlsx` | Corte real y conteos fenologicos | `Finca`, `Bloque`, `Fecha`, `Variedad`, `semana`, `Cantidad`, `conteo_RC`, `conteo_SS`, `conteo_AP`, `conteo_CO`, `conteo_total` |
| `camas_muestreadas_semana.xlsx` | Camas muestreadas por semana | `Semana` (`S<digitos>`), `Finca`, `Bloque`, `Cantidad_csv` |
| `plano_siembra.xlsx` | Camas activas por fecha | `Finca`, `Bloque`, `Cama`, `Variedad`, `Fecha Siembra`, `Fecha Erradicacion`, `Plantas`, `Area Sembrada`, `Estado` |
| `Podas 10.xlsx` | Eventos de poda | `Finca`, `Block`, `Fecha`, `Variedad`, `Destino`, `Cantidad` |

Reglas:

- Variedad solo `FREEDOM`.
- Clave unica `Finca + Bloque + Fecha` en conteos.
- Bloques siempre texto (`1`, `7`, `P11`).
- Fechas en formato dia/mes/ano (`dd/mm/yyyy`).
- `Cantidad` es el corte comercial real del dia.
- `conteo_CO` es corte inmediato visual; no es PRE_RC.

## 2. Fenologia (`data/raw/fenologia`)

| Archivo | Uso |
|---|---|
| `FENOLOGIAS ABRIL FREEDOM.xlsx` | Trayectorias M3 periodo ABRIL (hojas `Abril`, `Junio`) |
| `FENOLOGIAS JULIO FREEDOM.xlsx` | Trayectorias M3 periodo JULIO (hoja `JULIO`) |
| `Fenologias13.08Final-1.xlsx` | P32 (hojas `Garbanzo`, `Rayando 1`, `Separando S`, `Definiendo P`) |

Formato ancho: columnas `FINCA`, `FECHA` y una columna por tallo. Los codigos
pendientes (`SP aislado`, `SP2 1/2`, `R4C`, `S1`, `1P`, `1 1/2P`, `2P`, `R6`)
nunca se homologan en silencio; van a QA.

## 3. Clima (`data/raw/clima`)

| Archivo | Uso |
|---|---|
| `2025.xlsx` | Estacion horaria 2025 |
| `2026.xlsx` | Estacion horaria 2026 |

Columnas: `IdEstacion`, `Estacion`, `FechaHora`, variables meteorologicas.
La estacion operativa es la 746 para LA PRADERA. `FechaHora` puede venir como
texto o serial Excel; el contrato la parsea con `dayfirst=True`.

## 4. Auxiliares (`data/raw/auxiliares`)

| Archivo | Uso |
|---|---|
| `calendario.xlsx` | Referencia de calendario |
| `resumen_cortes_reales.xlsx` | Resumen de cortes para auditoria |

No participan en el entrenamiento principal.

## 5. Scoring externo (`data/external/scoring`)

| Archivo | Uso |
|---|---|
| `conteos_vs_cortes_multifinca.xlsx` | Solo evaluacion de "Nueva entrada" |

Mismas columnas que el historico. Nunca se usa para entrenar: solo aporta
features en el origen y reales para contrastar.

## 6. Vision artificial (`data/vision/Resultados`)

Estructura por semana:

```text
data/vision/Resultados/S37/<Finca>/<Bloque>/<video>.csv
```

Cada CSV:

- Una sola fila.
- Separador coma o punto y coma.
- UTF-8 o UTF-8 con BOM.

Columnas:

```text
semana, finca, bloque, fecha, conteo_RC, conteo_SS, conteo_AP, conteo_CO,
conteo_total
```

Con `vision.require_valid_video_metadata=true` ademas se exigen:

```text
duracion_s, fps, ancho, alto
```

Reglas:

- `conteo_total == conteo_RC + conteo_SS + conteo_AP + conteo_CO`.
- Conteos enteros no negativos.
- La fecha debe pertenecer a la semana ISO de la carpeta.
- `semana`, `finca` y `bloque` declarados deben coincidir con la ruta.
- Cada CSV representa una cama distinta (una cama = un video).

Fincas objetivo: `ALMER`, `LA PRADERA`, `SANTA HELENA`. Las carpetas con otras
fincas se ignoran.

## 7. Como agregar una semana nueva

1. Crear `data/vision/Resultados/S37/`.
2. Dentro, una carpeta por finca y una por bloque.
3. Depositar los CSV validos de la semana.
4. Ejecutar la proyeccion:

```powershell
.\.venv\Scripts\python.exe scripts\run_operational.py
```

5. La corrida generara `outputs/operational/<run_id>/` y `latest.json`.
6. Reejecutar es idempotente: si los datos no cambiaron, reutiliza la corrida.

## 8. Como agregar un archivo historico nuevo

1. Ubicar el archivo en la subcarpeta correcta de `data/raw`.
2. Validar columnas segun la tabla de su seccion.
3. Ejecutar la auditoria:

```powershell
.\.venv\Scripts\python.exe src\fase0_auditoria.py
.\.venv\Scripts\python.exe src\fase0_contratos.py
```

4. Regenerar datasets y modelos:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --skip-historical
```

## 9. Que nunca debe hacerse

- No mover o editar los xlsx de `data/raw` fuera del flujo: rompe hashes.
- No mezclar `data/external/scoring` con `data/raw`.
- No renombrar carpetas `SNN` ni cambiar la estructura de 4 niveles.
- No convertir bloques a numeros.
- No borrar codigos fenologicos pendientes.