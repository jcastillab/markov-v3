# Ejecucion portable

## Regla de datos

- `data/raw/` es el historico original usado para construir datasets y ajustar
  hiperparametros.
- `resultados acutuales/` es una poblacion externa de scoring. Su columna
  `Cantidad` no se usa para entrenar ni seleccionar modelos.
- `outputs/` se regenera y no se versiona en Git.

## Preparar el entorno

Usar Python 3.12 desde la raiz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Provisionar los archivos historicos requeridos en `data/raw/` antes de ejecutar
el pipeline. No se deben reemplazar durante una evaluacion externa.

## Construir artefactos historicos

```powershell
.\.venv\Scripts\python.exe src\fase1_pipeline.py
.\.venv\Scripts\python.exe src\fase2_m3.py
.\.venv\Scripts\python.exe src\fase3_p32.py
.\.venv\Scripts\python.exe src\fase4_podas.py
.\.venv\Scripts\python.exe src\fase5_clima.py
.\.venv\Scripts\python.exe src\fase6_supervisado.py
.\.venv\Scripts\python.exe src\evaluacion_hyperparametros.py
.\.venv\Scripts\python.exe src\fase7_bayes.py
.\.venv\Scripts\python.exe src\evaluacion_rolling.py
```

La seleccion de Random Forest usa `daily_wape` como objetivo principal. El
holdout rolling se conserva separado de la seleccion y de la entrada externa.

## Evaluar un archivo externo

```powershell
.\.venv\Scripts\python.exe src\evaluacion_entrada.py "resultados acutuales\conteos_vs_cortes_multifinca.xlsx"
```

Se generan en `outputs/evaluation/`:

- `predictions_entrada_diarias.csv`
- `predictions_entrada_semanales.csv`
- `evaluacion_entrada.xlsx`
- `external_run_manifest.json`

Las semanas parciales se comparan solo con los dias reales observados y quedan
marcadas como `PARCIAL`. Las semanas sin reales quedan como `NO EVALUABLE`.

## Abrir el dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\dashboard_streamlit.py
```

El dashboard separa validacion fija, holdout rolling y evaluacion externa. La
entrada externa permite filtrar por modelo, finca, bloque y semana, y muestra
semanas completas, parciales, aciertos, cercanas y no aciertos.
