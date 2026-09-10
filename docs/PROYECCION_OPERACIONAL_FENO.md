# Proyeccion operacional fenologica

## Alcance

El flujo consume exclusivamente los CSV producidos externamente por
Vision-artificial. No importa ni modifica codigo, modelos o configuracion de ese
proyecto. Cada CSV valido representa una cama diferente.

Los modelos operacionales son:

| Rol | Modelo | Referencia rolling |
|---|---|---:|
| Principal | RF_OPT_FENO_OPERATIONAL | RF_OPT_FENO_ROLLING, WAPE 23,68% |
| Sombra | RF_H1_H7_FENO_OPERATIONAL | RF_H1_H7_FENO_ROLLING, WAPE 26,49% |
| Baseline | E00_M3_BASE_OPERATIONAL | M3 obligatorio |

El grupo FENO no usa poda ni clima. Si usa conteos fenologicos, escala de
muestreo, M3 e historial causal de cortes. Si no hay al menos el numero de dias
configurado en `operational.minimum_history_days_28d`, los RF quedan como
`NO_DISPONIBLE`; el sistema no reemplaza el historial faltante por ceros.

## Entrenamiento

El entrenamiento y la prediccion son procesos separados. Primero se construyen
los datasets de Fase 1 y los intervalos M3 de Fase 2. Luego se congelan los RF:

```powershell
.\.venv\Scripts\python.exe src\fase1_pipeline.py
.\.venv\Scripts\python.exe src\fase2_m3.py
.\.venv\Scripts\python.exe src\entrenar_modelos_operacionales.py
```

Opcionalmente se puede limitar la ultima fecha objetivo conocida:

```powershell
.\.venv\Scripts\python.exe src\entrenar_modelos_operacionales.py --cutoff 2026-08-09
```

El artefacto y su manifest se escriben bajo `outputs/models/operational/`. El
manifest conserva el cutoff, las features en orden, los parametros y el hash
SHA-256 del artefacto. Cada version queda en un directorio content-addressed;
`active.json` solo indica cual version usa el scoring y no elimina versiones
anteriores.

## Prediccion semanal

Para procesar automaticamente la ultima carpeta `SNN` disponible:

```powershell
.\.venv\Scripts\python.exe src\proyeccion_vision.py
```

Para una semana concreta:

```powershell
.\.venv\Scripts\python.exe src\proyeccion_vision.py --semana S36
```

Cada combinacion de entrada, modelo y configuracion genera un `run_id` estable.
Una corrida existente no se sobrescribe. Los resultados quedan en:

```text
outputs/operational/<run_id>/
  comparacion_modelos.xlsx
  conteos_consolidados.parquet
  manifest.json
  predicciones_diarias.parquet
  predicciones_semanales.parquet
  qa.csv
  videos_validos.parquet
```

`comparacion_modelos.xlsx` deja vacias las columnas `modelo_elegido` y
`valor_reportado` para que la decision de reporte sea manual.

## Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\dashboard_streamlit.py
```

La seccion `Ultima proyeccion operacional` compara los tres modelos y permite
filtrar por finca y bloque. El dashboard solo lee artefactos; no entrena modelos
ni modifica predicciones.

## Causalidad

- Los modelos se entrenan solo con targets cuya fecha objetivo no supera el
  cutoff del manifest.
- Los rezagos de corte se construyen con fechas no posteriores al origen.
- Las semanas sin real conservan sus predicciones y quedan pendientes de
  evaluacion.
- Una correccion de datos crea otro `run_id`; nunca cambia una corrida anterior.
- El `run_id` incluye hashes de entrada visual, historial de cortes, plano,
  fuentes M3, artefacto, manifest, configuracion y codigo ejecutado.
- Finca, bloque y semana declarados deben coincidir con la ruta; una
  contradiccion rechaza el CSV.
- Cada corrida conserva el detalle de videos/camas aceptados y hashes de todas
  sus salidas para detectar modificaciones posteriores.
