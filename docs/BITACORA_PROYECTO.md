# Bitacora del proyecto Markov Freedom

## 1. Proposito y reglas metodologicas

El proyecto construye un sistema de pronostico de corte comercial para rosas
Freedom, con alcance en ALMER, LA PRADERA y SANTA HELENA. La referencia
mecanistica obligatoria es M3, un modelo de estados fenologicos y transiciones.

Reglas que guiaron el trabajo:

- La metrica primaria es WAPE; MAPE no se usa como metrica principal.
- La validacion es temporal y causal: las features disponibles para un origen
  `t0` no pueden usar informacion posterior a `t0`.
- El baseline M3 debe permanecer en todas las comparaciones.
- Los resultados retrospectivos u oracle no se mezclan con el ranking causal.
- Excel se usa como entrada RAW o salida de auditoria; el formato interno es
  Parquet.
- Los codigos fenologicos pendientes no se homologan silenciosamente.

## 2. Resumen ejecutivo

El pipeline completo se implemento en ocho fases. La evaluacion formal vigente
selecciona provisionalmente
`RF_FENO_PODA_CLIMA_n200_d3_l5_s20_fsqrt_poisson` como champion de
investigacion. En el holdout rolling-origin causal obtiene WAPE de 30,57 por
ciento sobre 441 observaciones comunes, frente a 57,59 por ciento de M3.

La busqueda adicional de hiperparametros evaluo 576 combinaciones de RF y dos
challengers. La seleccion se amplio a WAPE, MAE, RMSE, sesgo absoluto y R2,
en escalas diaria y semanal. El mejor RF por score compuesto fue:

```text
RF_FENO_PODA_CLIMA_n200_d3_l5_s20_fsqrt_poisson
WAPE diario de seleccion: 31,68 por ciento
R2 diario de seleccion: 0,4579
WAPE semanal de seleccion: 19,00 por ciento
R2 semanal de seleccion: 0,8577
```

El score compuesto da 70 por ciento de peso al alcance semanal y 30 por ciento
al diario. El periodo de seleccion precede al holdout final; la especificacion completa se persiste en
`selected_model_manifest.json`; rolling reconstruye la misma familia, features
e hiperparametros. La comparacion formal ya la incorpora con bootstrap, aunque
aun no existe un tercer periodo temporal independiente.

## 3. Bitacora por fase

### Fase 0 - Auditoria y contratos

Se inventariaron las fuentes Excel, hojas, columnas, tipos, fechas, nulos,
duplicados y hashes SHA-256. Se definieron los contratos de ingesta para
conteos, camas, clima, podas, planos y fenologia.

La fenologia historica Abril/Junio/Julio llega en formato ancho con 30 tallos
por fila. P32 llega como seguimiento individual en cuatro hojas. Se dejaron
explicitamente pendientes `SP` aislado, `SP2 1/2`, `R4C`, `S1`, `1P`, `1 1/2P`,
`2P` y `R6`.

Aciertos:

- Se conservaron los archivos RAW y se registraron hashes.
- Se identificaron fechas mixtas y problemas de tipos antes de modelar.
- Se evito convertir codigos dudosos por inferencia.

Errores o riesgos:

- Hay codigos fenologicos sin contrato agronomico cerrado.
- P32 no es contemporaneo con todas las ventanas historicas, por lo que no es
  valido como feature causal operacional.

Artefactos principales: `outputs/data_quality/inventario_fuentes.csv`,
`inventario_columnas.csv`, `hashes_fuentes.csv` y `hallazgos_fase0.md`.

### Fase 1 - Capa canonica y ventanas

Se construyeron dimensiones, `fact_bloque_dia`, conteos origen, camas,
operaciones de poda y ventanas de pronostico de siete dias. Cada fila de una
ventana contiene finca, bloque, origen, objetivo, horizonte y target diario.

Aciertos:

- Se separo la fecha del dato fenologico de la fecha objetivo.
- Se conservaron ceros reales de corte.
- Se centralizaron parametros en `config/pipeline.yaml`.

Riesgos:

- Las ventanas dependen de la cobertura real por finca y bloque.
- Las ventanas parciales requieren distinguir dias observados de dias aun sin
  real.

Artefactos: `fact_bloque_dia.parquet`, `forecast_windows.parquet`, dimensiones
Parquet y `qa_join_coverage.csv`.

### Fase 2 - Baseline M3

Se reconstruyo M3 con matrices de transicion por vigencia Abril y Julio. Se
probaron valores de ingreso RC y se mantuvieron variantes baseline y calibrada.
La validacion causal de Fase 2 contiene 1.295 dias; el baseline obtiene WAPE de
45,94 por ciento.

Aciertos:

- Se limito cada matriz a datos disponibles en el origen.
- Se fijo la orientacion de la matriz y se genero auditoria de transiciones.
- M3 quedo como referencia obligatoria.

Riesgos:

- La calibracion de ingreso RC no produjo una mejora suficiente para desplazar
  la referencia mecanistica.

Artefactos: `transition_intervals_tradicional.parquet`,
`metrics_fase2_m3.csv`, `m3_matrix_audit.csv` y `m3_manifest.json`.

### Fase 3 - P32 y Semi-Markov

Se normalizo el seguimiento P32 a estados raw, micro y macro, y se generaron
intervalos y distribuciones de edad. Se evaluaron M3-P32 y variantes
Semi-Markov.

Resultados documentados:

- 1.297 observaciones P32 normalizadas.
- 514 intervalos validos bajo contrato.
- M3-P32 regularizado: WAPE 39,61 por ciento, pero retrospectivo.
- Semi-Markov RC+SS: WAPE 41,97 por ciento, tambien retrospectivo.
- El baseline comparable obtiene 45,94 por ciento.

Aciertos:

- `SP` aislado no se transformo a `SP1`.
- Se corrigio la concentracion artificial de edad inicial en cero.
- Los resultados retrospectivos se etiquetaron y excluyeron del ranking.

Error o limitacion principal: P32 fue levantado despues de las ventanas, por lo
que su mejora no constituye evidencia operacional causal.

### Fase 4 - Podas

Se construyeron features de CORTE y ALINEAMIENTO con lags de 42 a 84 dias,
sumas, kernels de 8 a 12 semanas, dias desde la ultima operacion y acumulados.
Las operaciones ESTIMADO quedaron fuera.

Resultado: las variantes M3 con poda conservaron WAPE de 45,94 por ciento; los
coeficientes positivos empeoraron el ajuste. La poda quedo como challenger no
promovido.

Aciertos:

- Las features de poda se limitaron a fechas `<= t0`.
- Se preservo la contabilidad del sistema y no se agrego masa artificial.

### Fase 5 - Clima

Se agregaron datos horarios a diario para LA PRADERA, estacion 746. Se
calcularon lluvia, ET0, VPD, GDD y una estimacion declarada de DLI PAR.

Resultado exploratorio:

- Baseline local: WAPE 45,75 por ciento.
- Mejor challenger M3 clima: WAPE 44,27 por ciento con coeficiente 0,10.
- La mejora fue de 1,48 puntos porcentuales y no se promovio.

Aciertos:

- No se uso clima futuro.
- La radiacion estimada no se presento como medicion directa de invernadero.

Limitaciones:

- Solo existe cobertura para LA PRADERA.
- No hay medicion de microclima de invernadero.
- Se requiere validacion interna y estabilidad por bloque y periodo.

### Fase 6 - Modelos supervisados

Se creo `dataset_supervisado_diario.parquet`, una fila por finca, bloque,
origen y horizonte. Las features incluyen conteos y proporciones fenologicas,
prediccion M3, escala de camas e historial causal de cortes.

Resultados iniciales documentados:

- RF pooled FENO: WAPE 31,74 por ciento.
- RF H1-H7 FENO: WAPE 31,81 por ciento.
- RF semanal FENO: WAPE 25,24 por ciento, en otra escala.
- RF residual sobre M3: WAPE 42,55 por ciento; no promovido.
- GLM NB FENO: WAPE 31,60 por ciento; control interpretable competitivo.

Aciertos:

- Se mantuvo el split temporal causal.
- Se calcularon importancias por permutacion.
- Se separaron resultados diarios y semanales.

Error o limitacion: el RF mejora mucho a M3, pero inicialmente quedo como
exploratorio y requiere validacion adicional por finca, bloque y horizonte.

### Fase 7 - Modelos bayesianos

Se implementaron un Dirichlet-Multinomial conjugado centrado en M3 y un NB
jerarquico Gamma-Poisson con pooling por finca, bloque y horizonte. No se uso
PyMC para evitar una dependencia innecesaria en esta primera iteracion.

Resultados:

- M3 Dirichlet-Multinomial: WAPE 46,73 por ciento; no promovido.
- NB jerarquico: WAPE 35,79 por ciento; challenger exploratorio.
- Cobertura del NB: 5,0 por ciento al 80 por ciento y 7,4 por ciento al 95
  por ciento; inferior a la nominal.

Aciertos:

- Se conservaron las restricciones de simplex del posterior Dirichlet.
- Se reporto la cobertura en lugar de asumir incertidumbre calibrada.

Limitacion: los intervalos bayesianos no son confiables aun para decisiones
operativas y requieren calibracion.

### Fase 8 - Comparacion final

Se compararon M3 y el modelo supervisado seleccionado sobre la interseccion del
holdout rolling-origin causal de 441 observaciones diarias. Se excluyeron P32
retrospectivo y resultados con poblacion, split o escala distintos.

Resultado formal:

- `RF_FENO_PODA_CLIMA_n200_d3_l5_s20_fsqrt_poisson`: champion provisional.
- WAPE actual en `ranking_final.csv`: 30,57 por ciento.
- R2 actual: 0,4601.
- IC bootstrap por origen al 95 por ciento: 27,35 a 34,31 por ciento.
- M3: baseline obligatorio.

La decision es provisional porque el holdout disponible es el ultimo periodo y
no existe un tercer periodo temporal independiente.

### Trabajo posterior - hiperparametros y trazabilidad

La busqueda se corrigio para usar las rejillas de configuracion, paralelizar
combinaciones con Joblib y evitar sobreasignacion. En el equipo disponible se
usaron automaticamente 3 procesos y 1 hilo por modelo, dejando 1 CPU libre.

Se evaluaron 576 combinaciones RF y dos challengers. Se generaron ganadores
separados por cada metrica diaria y semanal, ademas de un `selection_score`
compuesto con 70 por ciento de peso semanal y 30 por ciento diario. El mejor
RF por score compuesto fue
`RF_FENO_PODA_CLIMA_n200_d3_l5_s20_fsqrt_poisson`, con WAPE diario 31,68 por
ciento, R2 diario 0,4579, WAPE semanal 19,00 por ciento y R2 semanal 0,8577 en
el periodo temporal de seleccion.

Problemas encontrados y corregidos:

- El script tenia rejillas hardcodeadas distintas de `pipeline.yaml`.
- Cada modelo usaba `n_jobs=-1`, lo que impedia controlar la carga al
  paralelizar experimentos.
- Las predicciones destacadas no incluian nombre, features ni hiperparametros.
- Se corrigio un fallo al cambiar el almacenamiento de predicciones a vectores
  durante la paralelizacion.
- La interfaz Streamlit no mostraba el ganador de hiperparametros; ahora lo
  incluye en validacion fija con ficha de parametros, metricas y ranking
  semanal.
- Rolling reutilizaba solo hiperparametros y forzaba features `FENO`; ahora
  consume el grupo de features exacto declarado por el manifiesto.
- El split fijo podia partir una fecha entre train y validacion; ahora conserva
  fechas completas y excluye targets de entrenamiento posteriores al cutoff.

## 4. Artefactos clave

- Datos canonicos: `outputs/datasets/*.parquet`.
- Dataset RF: `outputs/datasets/dataset_supervisado_diario.parquet`.
- Ranking formal: `outputs/evaluation/ranking_final.csv`.
- Busqueda: `outputs/evaluation/metrics_hyperparametros.csv`.
- Predicciones seleccionadas: `outputs/evaluation/predictions_best_*.csv`, con
  salidas separadas para cada metrica diaria, semanal y el score compuesto.
- Importancias: `outputs/models/importance_feno.csv` y archivos hermanos.
- Interfaz: `src/dashboard_streamlit.py`.
- Configuracion: `config/pipeline.yaml`.

## 5. Estado y siguientes pasos

Estado actual: pipeline reproducible de investigacion, con baseline, modelos
challenger, comparacion rolling causal y dashboard. El mejor RF por score
compuesto ya esta incorporado a la comparacion formal como champion
provisional, no operacional.

Siguientes pasos recomendados:

1. Ejecutar un tercer periodo temporal congelado.
2. Validar el RF por finca, bloque y horizonte.
3. Monitorear estabilidad y drift del RF seleccionado por finca y horizonte.
4. Revisar cobertura y estabilidad de los intervalos bayesianos.
5. Cerrar el contrato de codigos fenologicos pendientes.
6. Validar el clima con cobertura multi-finca o mantenerlo como challenger.
