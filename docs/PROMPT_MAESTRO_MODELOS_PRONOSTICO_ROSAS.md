# PROMPT MAESTRO PARA AGENTE DE DESARROLLO

## Proyecto de pronóstico de corte comercial de rosas Freedom

Actúa simultáneamente como:

1. Arquitecto de software senior especializado en pipelines de datos y sistemas reproducibles.
2. Data scientist senior especializado en series temporales, modelos de conteo, Random Forest, modelos bayesianos, Markov y Semi Markov.
3. Estadístico senior especializado en validación temporal, regularización, incertidumbre y diseño de experimentos.
4. Ingeniero de datos senior encargado de contratos de datos, calidad, trazabilidad y construcción de datasets causales.
5. Ingeniero agrónomo experto en fisiología vegetal, producción de rosa de corte bajo invernadero y relaciones entre temperatura, radiación, humedad, VPD, acumulación térmica, poda, brotación, desarrollo floral y corte comercial.

Trabaja inicialmente en **modo PLAN**. Antes de escribir o modificar código debes inspeccionar el repositorio y todas las fuentes indicadas, reconstruir el flujo vigente, validar los supuestos y entregar un plan técnico detallado. No implementes una alternativa hasta haber definido sus datos de entrada, target, causalidad, validación y criterio de comparación.

El objetivo no es reemplazar el trabajo previo sin evidencia. El objetivo es construir una plataforma experimental reproducible donde todos los modelos compitan contra el mismo baseline, la misma población evaluable y las mismas métricas.

---

# 1. Objetivo de negocio

Pronosticar cuántos tallos de rosa `FREEDOM` alcanzarán el corte comercial por finca, bloque y día, con agregación semanal de lunes a domingo.

El pronóstico debe apoyar:

- Planeación de mano de obra.
- Programación de corte y poscosecha.
- Planeación comercial.
- Logística y empaque.
- Seguimiento agronómico.
- Detección anticipada de desviaciones entre desarrollo fenológico esperado y corte real.

La variable final de negocio es el **corte comercial real**, medido en tallos.

El sistema debe producir como mínimo:

- Pronóstico diario para los 7 días de la semana objetivo.
- Pronóstico semanal como suma de los 7 días.
- Resultado por finca y bloque.
- Resultado agregado por finca.
- Error diario y semanal.
- Trazabilidad de qué modelo, parámetros, datos y artefactos generaron cada predicción.

---

# 2. Problema estadístico

El conteo fenológico semanal entrega una fotografía parcial del stock de tallos en distintos estados de desarrollo. A partir de esa fotografía se debe estimar la cantidad que progresará hasta `PC`, corte comercial, durante los días siguientes.

La dificultad central es que el corte futuro depende de varias capas:

1. Cantidad observada en cada estado fenológico.
2. Distribución interna de edad dentro de cada estado.
3. Probabilidades de permanencia y transición.
4. Ingresos futuros al estado RC.
5. Manejo agronómico, en especial podas y alineamientos.
6. Ambiente, en especial temperatura, radiación, humedad y demanda evaporativa.
7. Representatividad de las camas muestreadas respecto al bloque completo.
8. Variabilidad por finca, bloque, época y manejo.

Los modelos deben respetar esta estructura y evitar fuga temporal.

---

# 3. Alcance de fincas y variedad

Variedad principal:

`FREEDOM`

Fincas con conteos y cortes comerciales en la fuente principal:

- `ALMER`
- `PRADERA`, canonicalizar como `LA PRADERA` en la capa maestra.
- `SANTA HELENA`

Los nombres cambian entre archivos. Debe existir una dimensión o mapa explícito de homologación.

Ejemplos detectados:

- `PRADERA`
- `Pradera`
- `LA PRADERA`
- `LA PRADERA BQ P11`

No hagas joins por texto sin normalizar.

Los bloques deben manejarse como identificadores de texto. Existen bloques numéricos y códigos como `P11`, `P31`, etc.

---

# 4. Archivos disponibles

## 4.1 `conteos_vs_cortes_multifinca.xlsx`

Fuente principal de conteos fenológicos y cortes comerciales reales.

Columnas observadas:

```text
Finca
Bloque
Flor
Variedad
Color
Fecha
Cantidad
semana
Semana
conteo_RC
conteo_SS
conteo_AP
conteo_CO
conteo_total
```

Interpretación:

- `Cantidad` es corte comercial real diario del bloque.
- Los campos `conteo_*` aparecen en fechas de conteo y quedan vacíos en la mayoría de días.
- `semana` usa formato ISO numérico como `202628`.
- `Semana` usa formato como `S28`.

Cobertura observada:

```text
Filas de datos: 4.951
Rango de fechas: 2026-04-20 a 2026-08-09
Filas con algún conteo fenológico: 447
```

Por finca:

| Finca | Filas diarias | Filas con conteo | Bloques | Semanas con conteo |
| --- | ---: | ---: | ---: | ---: |
| ALMER | 558 | 65 | 5 | 13 |
| PRADERA | 3.662 | 334 | 33 | 15 |
| SANTA HELENA | 731 | 48 | 7 | 13 |

Rangos de semanas con conteo observados:

- ALMER: 202619 a 202632.
- PRADERA: 202617 a 202632.
- SANTA HELENA: 202618 a 202632.

Regla histórica usada en el proyecto: para una combinación `Finca + Bloque + Semana`, el último conteo válido de la semana actúa como conteo origen.

Valida esta regla contra el repositorio antes de reproducirla.

`conteo_CO` existe y no debe eliminarse sin definir primero su significado agronómico y operacional.

---

## 4.2 `camas_muestreadas_semana.xlsx`

Describe cuántas camas fueron muestreadas por finca, bloque y semana.

Columnas:

```text
Semana
Finca
Bloque
Cantidad_csv
```

Cobertura observada:

```text
Filas de datos: 1.844
```

Existe al menos una fila anómala al inicio:

```text
Resultados | S21 | El Cerezo | 16
```

Debe detectarse como fila inválida y excluirse mediante QA, no mediante una regla dependiente de posición.

Para las tres fincas objetivo, después de homologar nombres, se observan registros desde aproximadamente S17 o S19 hasta S32.

No asumas que todos los bloques con camas muestreadas pertenecen a Freedom. El join final debe estar condicionado por la población del modelo y por la fuente de conteos.

---

## 4.3 `plano_siembra.xlsx`

Fuente de camas totales, plantas, área y vigencia de siembra.

Columnas observadas:

```text
Finca
Sector
Flor
Variedad
Bloque
Cama
Fecha Siembra
Fecha Siembra Inicial
Fecha Erradicacion
Plantas
Plantas Inicial
Area Sembrada
Estado
Semana
Total General
```

Cobertura global observada:

```text
Filas: 89.715
Filas ROSE + FREEDOM: 32.932
```

Para Freedom se observaron las siguientes camas activas en las tres fincas objetivo:

| Finca | Bloques Freedom | Camas Freedom |
| --- | ---: | ---: |
| ALMER | 5 | 606 |
| LA PRADERA | 34 | 5.029 |
| SANTA HELENA | 7 | 876 |

Las fechas de siembra y erradicación deben usarse para determinar camas activas en la fecha de origen. No uses un total estático si la vigencia de una cama cambia.

Construye:

```text
camas_activas(finca, bloque, fecha)
camas_muestreadas(finca, bloque, semana)
factor_extrapolacion = camas_activas / camas_muestreadas
```

Regla histórica del proyecto:

```text
si camas_muestreadas > camas_activas:
    factor_extrapolacion = 1
    registrar alerta QA
```

No ocultes estos casos.

---

## 4.4 `FENOLOGIAS ABRIL FREEDOM(3).xlsx`

Fuente histórica principal para construir el M3 tradicional.

Hojas observadas:

- `Abril`
- `Junio`

Estructura:

```text
FINCA | DIA | FECHA | tallo_1 | tallo_2 | ... | tallo_30
```

Cada columna de tallo representa una trayectoria individual observada día a día.

La hoja Abril contiene seguimiento de 21 fincas, incluida LA PRADERA, ALMER y SANTA HELENA.

Los códigos observados incluyen, entre otros:

```text
G
DG
RC1
RC2
RC3
RC4
PE
S1S
S2S
S3S
S4S
S5S
SP1
SP1/2
SP1 1/2
SP2
PC
X
```

Existen vacíos intermedios y terminales. Deben tratarse según el contrato de gaps, censura e imputación definido para el baseline.

---

## 4.5 `FENOLOGIAS JULIO FREEDOM(2).xlsx`

Segunda referencia temporal para M3.

Hoja:

`JULIO`

Cobertura observada:

- `LA PRADERA BQ P11`, 17 días, 2026-07-16 a 2026-08-01.
- `ALMER`, 22 días, 2026-07-13 a 2026-08-03.
- `SANTA HELENA BL 7`, 18 días, 2026-07-22 a 2026-08-08.

Incluye códigos similares a abril y algunos códigos que requieren homologación o QA, por ejemplo `R4C` y `S1`.

El baseline histórico ha manejado matrices por vigencia Abril y Julio. Reproduce esa lógica de forma parametrizable, sin fechas hardcodeadas dentro del código.

---

## 4.6 `Fenologias13.08Final-1.xlsx`

Seguimiento fenológico individual reciente, diseñado para estudiar duración y distribución de transición por estado.

Hojas relevantes:

- `Garbanzo`
- `Rayando 1`
- `Separando S`
- `Definiendo P`
- `Graficas`, no usar como fuente primaria de entrenamiento.

Se observan 200 etiquetas únicas, 50 por cada uno de los cuatro grupos principales.

### Garbanzo

Fechas observadas:

2026-08-13 a 2026-08-20.

Incluye:

```text
Etiqueta
Fenologia
Tamaño (mm)
Cama
Fenologia por fecha
Tamaño por fecha
```

Esta hoja contiene una variable adicional de tamaño, que debe conservarse en la capa normalizada aunque inicialmente no entre al modelo.

### Rayando 1

Fechas observadas:

2026-08-13 a 2026-08-18.

### Separando S

Fechas observadas:

2026-08-13 a 2026-08-18.

### Definiendo P

Fechas observadas:

2026-08-14 a 2026-08-19.

Códigos crudos detectados:

```text
G
DG
RC1
RC2
RC3
RC4
RC5
S1S
S2S
S3S
S4S
S5S
SP
SP1
SP1 1/2
SP1/2
SP 1/2
SP 1 1/2
SP 2
SP2
PC
X
S1
1P
1 1/2P
2P
R6
SP 1
```

Existe una contradicción que el agente debe resolver explícitamente antes de entrenar:

- Una versión de la arquitectura heredada convierte `SP` a `SP1`.
- Una decisión posterior del proyecto establece que `SP` aislado **no debe clasificarse automáticamente como AP** y que la secuencia AP oficial es `SP1 -> SP1 1/2 -> SP2 -> PC`.

No hagas una homologación silenciosa. Produce primero una tabla `codigo_crudo -> codigo_canonico -> macroestado -> estado_QA` y marca los códigos pendientes.

La implementación actual reportó menos tallos usables que las 200 etiquetas crudas. Reconciliar este punto forma parte del QA.

---

## 4.7 `Podas 10.xlsx`

Fuente de operaciones de manejo.

Columnas observadas:

```text
Finca
Fecha
Destino
Flor
Variedad
Color
Cantidad
Block
Cantidad_proy
```

Cobertura total:

```text
Filas: 439.204
Filas FREEDOM: 72.751
Rango FREEDOM observado: 2026-01-01 a 2026-08-26
```

Destinos Freedom observados:

```text
CORTE
ALINEAMIENTO
ESTIMADO
```

Importante:

`Destino = CORTE` en este archivo representa una operación de poda o corte de manejo. No equivale al corte comercial target de `conteos_vs_cortes_multifinca.xlsx`.

Usa nombres separados desde la capa de datos:

```text
PODA_CORTE
PODA_ALINEAMIENTO
PODA_TOTAL
CORTE_COMERCIAL_REAL
```

Registros Freedom observados en las fincas objetivo:

| Finca | Filas | Bloques |
| --- | ---: | ---: |
| LA PRADERA | 8.639 | 34 |
| ALMER | 1.530 | 5 |
| SANTA HELENA | 1.768 | 7 |

En las tres fincas objetivo, `Cantidad_proy` coincide numéricamente con `Cantidad` en los registros revisados. Aun así, documenta la unidad de negocio antes de usar esta variable como masa de tallos.

---

## 4.8 `2025.xlsx` y `2026.xlsx`

Datos horarios de estaciones climáticas.

Columnas:

```text
Estacion
IdEstacion
FechaHora
Lluvia (mm)
Temperatura (°C)
TemperaturaMaxima (°C)
TemperaturaMinima (°C)
Humedad(%)
Vel_Viento (km/h)
Dir_Viento (°)
Evapotranspiracion (mm/h)
Punto_Rocio (°C)
UV (Index)
Radiacion_Solar (w/m2)
Presion Atmosferica (hPa)
GDD (°C)
```

Ambos archivos contienen 19 estaciones.

Para PRADERA la estación identificada en el proyecto es:

```text
IdEstacion = 746
Estacion = La Pradera - Sunshine
```

Cobertura observada de la estación:

- 2025: 8.784 registros horarios en el archivo.
- 2026: 5.207 registros hasta 2026-08-05.

Los archivos presentan formatos mixtos de `FechaHora`, incluyendo texto y serial Excel. Normaliza a `datetime` de forma determinista.

Existe solapamiento entre archivos alrededor del cambio de año. Deduplica por:

```text
IdEstacion + FechaHora
```

No uses clima de otras estaciones para ALMER o SANTA HELENA sin una tabla de asignación agronómica validada. El primer alcance del modelo climático debe ser PRADERA.

---

## 4.9 `calendario.xlsx`

Dimensión temporal auxiliar.

Columnas:

```text
Fecha
semana
Mes_Año
Año
Sem Max
```

Sirve para estandarizar semana ISO y evitar diferencias entre funciones de calendario.

---

## 4.10 `resumen_cortes_reales(1).xlsx`

Fuente auxiliar de control y resultados históricos.

Contiene campos como:

```text
Semana
Finca
Modelo
Linea 100%
Linea 80/95%
Conteo Est
Conteo IA
Cortes reales
%cobertura
Mediana_videos
#Verde
#Amarillo
#Rojo
%IA
```

No reemplaza silenciosamente al target diario de `conteos_vs_cortes_multifinca.xlsx`. Úsalo para conciliación o validación agregada.

---

## 4.11 Documentación técnica

Lee completa y contrasta:

```text
ARQUITECTURA_MODELO(1).md
Informe_Consultoria_Modelo_Rosas_OpenCode.md
```

Estos documentos contienen hallazgos metodológicos y deuda técnica que deben entrar al plan.

---

# 5. Conocimiento acumulado del proyecto que debe preservarse

El desarrollo previo ya dejó aprendizajes importantes. No los descartes.

## 5.1 Baseline M3 obligatorio

M3 debe mantenerse como benchmark principal. Ningún modelo nuevo se considera mejor por tener mayor complejidad. Debe mejorar métricas sobre la misma población de backtest.

## 5.2 Estados operacionales

Cadena macro esperada:

```text
PRE_RC -> RC -> SS -> AP -> PC
```

Pérdida:

```text
RC -> L
SS -> L
AP -> L
```

Estados transitorios del motor principal:

```text
RC
SS
AP
```

Estados terminales:

```text
PC = corte comercial
L = pérdida
```

Homologaciones históricas deben revisarse contra la decisión vigente.

Referencia macro inicial:

```text
G, DG -> PRE_RC
RC1, RC2, RC3, RC4, RC5, PE -> RC
S1S, S2S, S3S, S4S, S5S -> SS
SP1, SP1 1/2, SP2 -> AP
PC -> PC
X -> L
```

No clasifiques `SP` aislado hasta resolver el contrato.

## 5.3 Problema actual de x0

El Semi Markov vigente ha colocado todo el conteo de RC, SS y AP en edad cero.

Eso elimina gran parte de la ventaja de un modelo dependiente de duración.

La nueva arquitectura debe separar:

```text
stock por macroestado
stock por microestado, si la fuente lo permite
P(edad | estado o microestado)
stock por estado y edad
```

## 5.4 Edad latente

El seguimiento individual reciente contiene muchos intervalos donde no se observa la entrada al estado. La edad inicial queda latente.

La arquitectura auditada detectó un error lógico en el cálculo de probabilidades competitivas para la inferencia de edad. No copies código heredado sin validar algebraicamente el posterior.

Debes incluir tests manuales de likelihood y normalización.

## 5.5 AP es el estado más sensible

En comparaciones previas, reemplazar AP M3 por AP P32 degradó el WAPE.

Consecuencia:

- Mantén AP M3 como referencia.
- Evalúa AP P32 como challenger.
- No promociones AP P32 sin soporte suficiente.
- Conserva microestados `SP1`, `SP1 1/2` y `SP2` para estudiar cercanía a PC.

## 5.6 Ingreso RC

El pipeline heredado llegó a fijar el ingreso diario en 20% del RC inicial.

Experimentos posteriores indicaron que valores alrededor de 10% a 15% daban mejor WAPE en casos revisados.

Por tanto:

`alpha_ingreso_RC` debe ser parámetro configurable y calibrado causalmente.

Evalúa al menos:

```text
0.00
0.05
0.10
0.12
0.15
0.20
```

Selecciona alpha usando datos de entrenamiento o inner validation. No uses el holdout final para escogerlo.

No segmentes alpha por bloque hasta demostrar soporte suficiente.

## 5.7 Ceros

Los días con corte comercial igual a cero son observaciones reales. No deben eliminarse de los promedios ni del dataset supervisado.

## 5.8 Granularidad de aprendizaje

No trates cada tallo del estudio fenológico como una observación independiente para evaluar el error de corte de bloque.

Los tallos sirven para aprender dinámica de transición.

La unidad de evaluación de producción debe mantenerse en finca, bloque, fecha y horizonte.

## 5.9 Backtesting

La validación debe ser temporal y causal.

Usa `rolling origin` o `walk forward`.

Para cada fecha de origen `t0`, un modelo operativo solo accede a información disponible hasta `t0`.

Cualquier modelo que use información posterior debe etiquetarse como:

`RETROSPECTIVO_ORACLE_NO_CAUSAL`

No mezcles sus métricas con las operacionales.

## 5.10 Extrapolación

La proyección muestral debe escalarse a bloque mediante camas activas y camas muestreadas, salvo que un modelo directo aprenda explícitamente en escala bloque.

Cada experimento debe declarar su escala:

```text
MUESTRA
BLOQUE
TASA_POR_CAMA
```

## 5.11 Git

Al cerrar cada bloque de trabajo validado:

1. Ejecutar tests.
2. Revisar `git diff`.
3. Crear commit descriptivo.
4. No acumular múltiples bloques validados sin commit.

---

# 6. Arquitectura de datos requerida

No construyas cada modelo leyendo Excel de forma independiente.

Crea una capa canónica común.

Flujo objetivo:

```text
RAW
  -> QA
  -> NORMALIZACION
  -> DIMENSIONES
  -> FEATURES DIARIAS
  -> DATASETS DE MODELO
  -> ENTRENAMIENTO
  -> BACKTEST
  -> COMPARACION
  -> EXPORTACION
```

## 6.1 Dimensiones mínimas

### `dim_finca`

```text
finca_id
finca_canonica
nombre_fuente
fuente
```

### `dim_bloque`

```text
finca_id
bloque_id
bloque_raw
```

### `dim_fecha`

```text
fecha
anio
mes
semana_iso
dia_semana
dia_anio
inicio_semana
fin_semana
```

### `dim_cama_vigencia`

```text
finca_id
bloque_id
cama_id
variedad
fecha_siembra
fecha_erradicacion
plantas
area_sembrada
estado
```

---

# 7. Tabla maestra diaria

Construye `fact_bloque_dia` con una fila por:

```text
Finca + Bloque + Fecha
```

Campos mínimos:

```text
finca
bloque
fecha
semana_iso
corte_comercial_real
conteo_RC
conteo_SS
conteo_AP
conteo_CO
conteo_total
es_fecha_conteo
camas_activas
camas_muestreadas
factor_extrapolacion
poda_alineamiento
poda_corte
poda_total
```

Para PRADERA añade clima diario.

No forward fill de conteos como si fueran nuevas observaciones. Si necesitas transportar el último conteo hasta una fecha objetivo, guarda:

```text
fecha_conteo_origen
dias_desde_conteo
```

---

# 8. Tabla de ventanas de pronóstico

Construye una tabla central `forecast_windows`.

Clave:

```text
finca
bloque
fecha_origen
fecha_objetivo
horizonte_dia
```

Una ventana válida debe tener 7 fechas objetivo consecutivas de lunes a domingo según el contrato vigente.

Campos:

```text
finca
bloque
fecha_origen
semana_origen
semana_objetivo
fecha_objetivo
horizonte_dia
conteo_RC_t0
conteo_SS_t0
conteo_AP_t0
conteo_CO_t0
conteo_total_t0
camas_muestreadas_t0
camas_activas_t0
factor_extrapolacion_t0
corte_real_dia
corte_real_semana
ventana_evaluable
motivo_no_evaluable
```

Esta tabla debe ser compartida por todos los modelos.

---

# 9. Dataset supervisado para Random Forest y modelos directos

Crear `dataset_supervisado_diario` con una fila por:

```text
Finca + Bloque + Fecha_origen + Horizonte_dia
```

Target:

```text
y = corte comercial real del bloque en fecha_objetivo
```

Campos de identidad:

```text
finca
bloque
fecha_origen
fecha_objetivo
horizonte_dia
semana_objetivo
```

## 9.1 Features de conteo

```text
RC_t0
SS_t0
AP_t0
CO_t0
TOTAL_t0
p_RC
p_SS
p_AP
p_CO
RC_AP_ratio
SS_AP_ratio
log1p_RC
log1p_SS
log1p_AP
```

Usa divisiones seguras.

## 9.2 Features de escala

```text
camas_activas
camas_muestreadas
factor_extrapolacion
cobertura_muestreo = camas_muestreadas / camas_activas
plantas_activas
area_activa
```

Solo agrega plantas y área si el join temporal es fiable.

## 9.3 Historial de cortes disponible en t0

Crear únicamente con fechas `<= fecha_origen`:

```text
corte_lag_1d
corte_lag_2d
corte_lag_3d
corte_lag_7d
corte_lag_14d
corte_sum_3d
corte_sum_7d
corte_sum_14d
corte_mean_7d
corte_mean_28d
```

No uses cortes dentro de la semana objetivo como feature.

## 9.4 Tendencia de conteos

Si existe un conteo anterior del mismo bloque:

```text
dias_desde_conteo_anterior
delta_RC
delta_SS
delta_AP
tasa_delta_RC_dia
tasa_delta_SS_dia
tasa_delta_AP_dia
```

Mantén un indicador de ausencia cuando no exista historial.

## 9.5 Calendario

```text
dia_semana_objetivo
semana_iso
mes
dia_anio
sin_doy
cos_doy
```

No uses un entero de semana como sustituto de tendencia temporal sin evaluar extrapolación.

---

# 10. Formulaciones de target para modelos directos

Compara tres escalas.

## A. Conteo de bloque

```text
y_block = corte_real_bloque
```

## B. Tasa por cama activa

```text
y_rate = corte_real_bloque / camas_activas
pred_bloque = pred_rate * camas_activas
```

## C. Corte equivalente a escala muestral

```text
y_sample_equiv = corte_real_bloque / factor_extrapolacion
pred_bloque = pred_sample * factor_extrapolacion
```

La selección debe surgir del backtest, no de preferencia conceptual.

Todos los modelos se comparan finalmente en la misma escala de bloque.

---

# 11. Modelo 1, M3 tradicional

Este es el baseline obligatorio.

## 11.1 Fuentes

- `FENOLOGIAS ABRIL FREEDOM(3).xlsx`
- `FENOLOGIAS JULIO FREEDOM(2).xlsx`

## 11.2 Unidad de aprendizaje

Un intervalo diario consecutivo de una trayectoria individual:

```text
estado_t -> estado_t+1
```

No construyas transición sobre gaps mayores a un día.

## 11.3 Macroestados

```text
RC
SS
AP
PC
L
```

`PRE_RC` sirve para entender entrada a RC, pero el motor operacional principal inicia desde RC, SS y AP.

## 11.4 Contrato matemático

Sea:

```text
x_t = [RC_t, SS_t, AP_t]^T
```

`Q` contiene masa que permanece dentro de estados transitorios.

`r` contiene probabilidad de corte comercial.

`p` contiene probabilidad de pérdida.

Para cada estado origen `j`:

```text
sum_i Q[i,j] + r[j] + p[j] = 1
```

Propagación:

```text
x_(t+1) = Q x_t + ingreso_RC_t * e_RC
PC_(t+1) = r^T x_t
L_(t+1) = p^T x_t
```

Verifica la convención de orientación contra el código de referencia antes de implementar.

## 11.5 M3 por exposiciones y grupos

El proyecto previo usa M3 como una matriz agregada que respeta exposiciones o grupos de duración, no como un promedio ingenuo de probabilidades.

Reproduce exactamente el método `M3_EXPOSICIONES_GRUPO` del baseline existente antes de modificarlo.

Genera auditoría con:

```text
finca
periodo
estado_origen
estado_destino
n_exposiciones
n_eventos
probabilidad
procedencia
```

## 11.6 Vigencias

Las matrices Abril y Julio representan contextos temporales distintos.

Externaliza en configuración:

```text
fecha_inicio_vigencia
fecha_fin_vigencia
fuente_fenologia
```

No fijes `01/07/2026` dentro del código.

---

# 12. Modelo 2, M3 + podas y alineamientos

Objetivo:

Incorporar la señal de manejo que inicia o sincroniza nuevos flujos vegetativos y reproductivos.

## 12.1 Hipótesis agronómica

Una poda o alineamiento no genera corte comercial inmediato. Su efecto aparece después de una latencia fisiológica asociada a brotación, elongación, diferenciación floral, desarrollo del botón y maduración hasta punto de corte.

Por esto el análisis debe estudiar rezagos y distribución de respuesta, no correlación contemporánea.

## 12.2 Features de poda

Por finca, bloque y fecha de origen:

```text
poda_alineamiento_lag_42d
poda_alineamiento_lag_49d
poda_alineamiento_lag_56d
poda_alineamiento_lag_63d
poda_alineamiento_lag_70d
poda_alineamiento_lag_77d
poda_alineamiento_lag_84d
poda_corte_lag_42d ...
poda_total_lag_42d ...
```

El trabajo previo encontró interés en 8 a 12 semanas. Mantén esa zona como ventana prioritaria y amplía el screening de forma controlada, por ejemplo 4 a 14 semanas.

Además calcula:

```text
poda_sum_7d alrededor de cada lag
poda_sum_14d alrededor de cada lag
poda_kernel_8_12w
dias_desde_ultima_poda
dias_desde_ultimo_alineamiento
poda_acumulada_28d
poda_acumulada_56d
poda_acumulada_84d
```

## 12.3 Análisis de temporalidad

No uses únicamente Pearson entre dos series crudas.

Realiza:

1. Gráficas de señal de poda y corte por bloque.
2. Cross correlation por rezago.
3. Pearson y Spearman como screening.
4. Correlación después de remover tendencia y estacionalidad.
5. Distributed lag regression.
6. Validación de estabilidad del lag por finca y bloque.
7. Evaluación de si el lag cambia según época.

Evita pseudo replicación cuando la misma señal de finca se repita sobre múltiples bloques.

## 12.4 Integración con M3

Compara al menos:

```text
M3_BASE
M3_PODA_INGRESO
M3_PODA_TRANSICION
M3_PODA_HIBRIDO
```

### M3_PODA_INGRESO

La poda modifica el ingreso diario a RC.

### M3_PODA_TRANSICION

La poda modifica una o más probabilidades de transición mediante una función acotada y renormalizada.

### M3_PODA_HIBRIDO

La poda modifica ingreso y transición solo si inner validation demuestra ganancia estable.

Nunca sumes masa sin conservar la contabilidad del sistema.

---

# 13. Modelo 3, M3 + clima

Primer alcance climático:

`LA PRADERA`, estación 746.

## 13.1 Procesamiento horario a diario

Construye una tabla `clima_diario`.

Variables directas:

```text
temp_mean_C
temp_min_C
temp_max_C
temp_range_C
rh_mean_pct
rh_min_pct
rh_max_pct
rain_mm
rain_hours
wind_mean_kmh
wind_max_kmh
et0_mm
dewpoint_mean_C
uv_mean
uv_max
radiation_mean_W_m2
radiation_max_W_m2
pressure_mean_hPa
```

No promedies lluvia para representar acumulación diaria. Usa suma.

## 13.2 VPD

Calcula VPD a resolución horaria cuando temperatura y HR estén disponibles.

Presión de vapor de saturación:

```text
es(T) = 0.6108 * exp(17.27*T / (T + 237.3))
```

VPD:

```text
VPD_kPa = es(T) * (1 - RH/100)
```

Luego agrega:

```text
vpd_mean_kPa
vpd_max_kPa
vpd_min_kPa
vpd_daylight_mean_kPa
hours_vpd_gt_1
hours_vpd_gt_1_5
hours_vpd_lt_0_3
```

Los umbrales son features de exploración, no límites fisiológicos universales.

## 13.3 GDD

El proyecto heredado usa base 5 °C.

Calcula como mínimo:

```text
GDD5 = max(((Tmax + Tmin)/2) - 5, 0)
```

Incluye análisis de sensibilidad para una base cercana a 5.2 °C, reportada en literatura de desarrollo de brotes de rosa, y otras bases cercanas si el backtest lo justifica.

Features:

```text
gdd5_daily
gdd5_sum_3d
gdd5_sum_7d
gdd5_sum_14d
gdd5_sum_21d
gdd5_sum_28d
gdd5_sum_42d
gdd5_sum_56d
```

## 13.4 Radiación y DLI

La columna `Radiacion_Solar (w/m2)` mide energía radiante, no PPFD.

Calcula primero energía solar diaria:

```text
solar_MJ_m2_d = sum(radiation_W_m2 * segundos_intervalo) / 1e6
```

No llames DLI en `mol m-2 d-1` a esa variable.

Para estimar DLI fotosintético desde radiación de onda corta debes declarar una conversión explícita a PAR o PPFD. Si no existe sensor cuántico, conserva ambas variables:

```text
solar_MJ_m2_d
DLI_PAR_est_mol_m2_d
```

Documenta el factor de conversión y su incertidumbre.

No confundas radiación exterior con radiación recibida por el cultivo bajo cubierta. Si no existe transmitancia de invernadero, etiqueta la variable como proxy exterior.

## 13.5 Acumulados climáticos

Para cada feature diaria `x`, genera ventanas causales que terminen en `t0`:

```text
x_mean_1d
x_mean_3d
x_mean_7d
x_mean_14d
x_mean_21d
x_mean_28d
x_mean_42d
x_mean_56d
```

Para variables aditivas:

```text
rain_sum_3d ...
gdd_sum_3d ...
DLI_sum_3d ...
et0_sum_3d ...
```

También evalúa anomalías respecto a una referencia móvil:

```text
x_7d_minus_28d_mean
x_zscore_28d
```

## 13.6 Features fisiológicas combinadas

Evalúa interacciones con fundamento agronómico:

```text
temp_x_DLI
GDD_x_DLI
VPD_x_DLI
rain_x_VPD
rh_x_temp
```

No introduzcas todas las interacciones a la vez. Usa hipótesis y validación temporal.

## 13.7 Clima desde poda

Construye un experimento especialmente importante:

```text
GDD acumulado desde última poda o alineamiento hasta t0
DLI acumulado desde última poda o alineamiento hasta t0
lluvia acumulada desde última poda o alineamiento hasta t0
VPD medio desde última poda o alineamiento hasta t0
```

Esta representación conecta mejor la fisiología del ciclo del tallo con el manejo que un lag calendario fijo.

## 13.8 Clima futuro y causalidad

Con los archivos disponibles tienes clima observado, no un pronóstico meteorológico futuro.

Por tanto:

### Operacional causal

Solo features climáticas con fecha `<= t0`.

### Oracle retrospectivo

Se admite un experimento que use clima real de `t0+1 ... t0+7` únicamente como techo diagnóstico.

Debe etiquetarse:

```text
NO_CAUSAL_ORACLE
```

No compite como modelo de producción.

Una futura versión operacional que use clima futuro requerirá integrar un proveedor de pronóstico meteorológico y congelar la versión disponible en t0.

## 13.9 Temporalidad clima -> corte

Evalúa lags mediante:

1. Cross correlation.
2. Distributed lag models.
3. Correlación parcial controlando estacionalidad.
4. Modelos por estado fenológico.
5. Estabilidad temporal del efecto.

No interpretes correlación como causalidad fisiológica.

---

# 14. Fundamento agronómico que debe guiar el análisis climático

Trata estas relaciones como hipótesis a validar, no como coeficientes fijos.

## Temperatura

La tasa de desarrollo de brotes de rosa responde a temperatura y acumulación térmica. La literatura de rosa de corte ha mostrado utilidad de unidades térmicas para seguir desarrollo desde brotación hasta botón visible y cosecha.

Por eso GDD merece prioridad sobre temperatura media aislada.

## Radiación y DLI

La luz aporta energía para fotosíntesis y modifica tasa de crecimiento y desarrollo. El efecto depende de la etapa y de la interacción con temperatura.

Por eso evalúa:

```text
DLI previo al conteo
DLI acumulado desde poda
interacción DLI x temperatura
```

## Humedad y VPD

VPD resume el gradiente de demanda evaporativa mejor que HR aislada. En rosa bajo invernadero influye en transpiración y conducta estomática.

La estación es exterior y el cultivo está bajo cubierta. Esta diferencia reduce la interpretación causal directa.

Registra explícitamente:

```text
clima_fuente = EXTERIOR_ESTACION
microclima_invernadero_disponible = False
```

## Lluvia

La lluvia exterior no equivale a suministro hídrico del cultivo bajo cubierta, donde el riego es manejado.

No interpretes `Lluvia_mm` como agua recibida por la raíz.

Úsala como proxy de condiciones atmosféricas asociadas, nubosidad, humedad y régimen sinóptico, y valida su aporte incremental después de temperatura, radiación y VPD.

## Poda

La poda define el inicio de nuevos pulsos de desarrollo. Su efecto debe analizarse en tiempo biológico, no únicamente como una suma de tallos podados.

La combinación `poda + GDD acumulado` es una hipótesis prioritaria.

---

# 15. Modelo 4, M3 con distribuciones de transición de `Fenologias13.08Final-1.xlsx`

Objetivo:

Crear una variante Markov estática comparable con M3 tradicional, pero cuyas probabilidades provengan del seguimiento P32 reciente.

Nombre sugerido:

```text
M3_P32_DIST
```

No confundir con Semi Markov.

## 15.1 Dataset de transición

Construye `transition_intervals_p32` con una fila por tallo y par de fechas consecutivas.

Campos:

```text
grupo
etiqueta
cama
fecha_t
fecha_t1
estado_raw_t
estado_raw_t1
estado_micro_t
estado_micro_t1
estado_macro_t
estado_macro_t1
tamano_mm_t
tamano_mm_t1
delta_dias
evento
valido
motivo_exclusion
```

Solo `delta_dias == 1` entra a la transición diaria estándar.

## 15.2 Conteos

Para cada estado origen:

```text
N(i,j) = número de intervalos i -> j
```

Probabilidad:

```text
P(i,j) = N(i,j) / sum_j N(i,j)
```

## 15.3 Variantes obligatorias

```text
M3_P32_RAW
M3_P32_CANONICO
M3_P32_REG
```

### RAW

Usa todos los códigos resueltos.

### CANONICO

Excluye o separa transiciones no compatibles con la cadena fenológica.

### REG

Aplica suavizado Dirichlet o shrinkage hacia M3 tradicional.

Ejemplo conceptual:

```text
P_post(j|i) = (N_ij + tau * P_M3(j|i)) / (N_i + tau)
```

Tau debe seleccionarse en entrenamiento o sensibilidad documentada.

## 15.4 Ablación

Compara híbridos:

```text
P32 solo en RC + M3 en SS/AP
P32 en RC/SS + M3 en AP
P32 en RC/SS/AP
```

Esto es importante porque AP P32 ya mostró degradación en experimentos anteriores.

---

# 16. Modelo 5, Semi Markov corregido

Aunque el usuario solicita varios M3 y nuevos modelos, conserva un Semi Markov como línea central de investigación.

## 16.1 Estado y edad

Representa stock como:

```text
masa[estado, edad]
```

No inicies todo en edad cero.

## 16.2 Distribución inicial de edad

Compara:

```text
PI0
PI_OCC
P(edad | microestado)
P(edad | estado, finca, periodo)
posterior de edad latente
```

## 16.3 Hazards

Para estado `s`, edad `d` y evento `k`:

```text
h(s,d,k)
```

Eventos:

```text
STAY
ADVANCE
CUT
LOSS
OTHER, si se conserva
```

Las probabilidades competitivas deben sumar 1.

## 16.4 Tests matemáticos mínimos

1. Toda distribución posterior de edad suma 1.
2. Todos los hazards están entre 0 y 1.
3. La suma de eventos por estado y edad es 1.
4. La supervivencia no aumenta con edad.
5. La masa total se conserva salvo PC y L.
6. Un gap no genera transición diaria estándar.
7. Un episodio censurado no inventa evento de salida.
8. Un código pendiente rompe la trayectoria o entra como QA, no como estado inventado.
9. La likelihood manual de un caso pequeño coincide con el código.
10. El fallback a M3 conserva procedencia.

---

# 17. Modelo 6, Random Forest

Construye Random Forest como modelo directo de pronóstico, no como sustitución ciega del mecanismo fenológico.

Objetivo:

Aprender la relación:

```text
estado del bloque en t0 + historia causal + manejo + clima -> corte futuro
```

## 17.1 Variantes

### RF_DIARIO_POOLED

Una sola Random Forest con `horizonte_dia` como feature.

### RF_H1_H7

Siete modelos independientes, uno por horizonte.

### RF_SEMANAL

Predice directamente la suma de corte de lunes a domingo.

Compara las tres.

## 17.2 Features base

Usa el `dataset_supervisado_diario` descrito antes.

Conjuntos de features por ablación:

```text
RF_FENO
RF_FENO_PODA
RF_FENO_CLIMA
RF_FENO_PODA_CLIMA
```

No evalúes solo el modelo completo. Necesitamos saber qué bloque de variables agrega señal.

## 17.3 Control de complejidad

La cantidad de semanas históricas es limitada.

Por eso prioriza:

```text
max_depth acotado
min_samples_leaf mayor que 1
min_samples_split regularizado
max_features controlado
bootstrap
```

No elijas hiperparámetros por el holdout final.

Usa búsqueda reducida con validación temporal.

## 17.4 Importancia

Reporta:

```text
permutation importance en validation
importancia por grupo de features
PDP o ALE para variables principales, si el tamaño de muestra lo soporta
```

No interpretes `feature_importances_` por impureza como evidencia causal.

## 17.5 Identificadores

No entregues `bloque` como entero ordinal sin pensar la codificación.

Compara:

- One hot para finca.
- One hot para bloque cuando exista soporte.
- Modelo sin bloque para probar generalización.

Nunca uses una codificación numérica que implique que bloque 26 está más cerca de 25 que de 4.

---

# 18. Modelo 7, modelos bayesianos

La línea bayesiana es prioritaria porque el problema tiene poca muestra en algunas capas y existen conocimientos previos útiles provenientes de M3.

Implementa por etapas.

## 18.1 Bayes A, M3 Dirichlet Multinomial

Para cada estado origen, modela la fila de probabilidades con prior Dirichlet.

```text
P_s ~ Dirichlet(alpha_s)
N_s ~ Multinomial(P_s)
```

Usa M3 tradicional como centro del prior cuando corresponda.

Ventajas esperadas:

- Regularización natural.
- Intervalos posteriores.
- Menor sensibilidad a celdas con pocos eventos.
- Predicción posterior en vez de una matriz puntual.

Salida:

```text
media posterior
mediana posterior
intervalo 80%
intervalo 95%
```

Propaga draws de matrices para obtener distribución de corte semanal.

## 18.2 Bayes B, Semi Markov jerárquico

Para cada estado y edad:

```text
P(evento | estado, edad)
```

Usa pooling parcial entre edades y, si los datos lo permiten, entre fincas o periodos.

M3 debe actuar como prior informativo débil o moderado, no como verdad fija.

## 18.3 Bayes C, regresión jerárquica de conteo

Modelo directo sobre corte comercial.

Distribución sugerida para evaluar:

```text
Negative Binomial
```

Estructura conceptual:

```text
y[b,t,h] ~ NegBin(mu[b,t,h], dispersion)

log(mu[b,t,h]) =
    offset_escala
    + intercepto_global
    + efecto_finca
    + efecto_bloque
    + efecto_horizonte
    + beta_RC * f(RC_t0)
    + beta_SS * f(SS_t0)
    + beta_AP * f(AP_t0)
    + beta_poda * poda_features
    + beta_clima * clima_features
    + estacionalidad
```

Usa efectos jerárquicos para compartir información sin forzar igualdad entre bloques.

## 18.4 Bayes D, estado latente para x0

Explora un modelo donde la distribución de edad dentro de cada macroestado sea latente:

```text
pi_s ~ Dirichlet(prior_s)
```

El pronóstico integra la incertidumbre de `pi_s` en lugar de fijar toda la masa en edad cero.

Esta variante conecta directamente con el mayor problema actual del Semi Markov.

## 18.5 Evaluación bayesiana

Además de WAPE de la media o mediana posterior, reporta:

```text
cobertura_intervalo_80
cobertura_intervalo_95
ancho_medio_intervalo
log predictive density, si está disponible
CRPS, si la implementación lo soporta
```

Revisa convergencia:

```text
R-hat
ESS
trace plots
posterior predictive checks
```

No promociones un modelo con convergencia deficiente aunque el WAPE puntual sea bueno.

---

# 19. Baseline directo adicional recomendado

Del proyecto anterior `ecuaciones conteo rosa` se aprendió que conviene mantener un modelo directo interpretable antes de modelos de árbol o Bayes complejos.

Incluye como control:

```text
GLM Negative Binomial regularizado
```

o una aproximación equivalente disponible en el stack.

Forma conceptual:

```text
log(mu) = beta0 + beta_X X + gamma_h + interacciones X*h
```

Este modelo sirve para detectar si Random Forest agrega ganancia real o simplemente captura una tendencia que un modelo de conteo sencillo ya explica.

---

# 20. Integración entre modelos mecanísticos y modelos directos

Crea features derivadas de M3 para RF y Bayes, sin reemplazar el benchmark.

Ejemplos:

```text
M3_pred_h1 ... M3_pred_h7
M3_pred_week
M3_stock_RC_h1
M3_stock_SS_h1
M3_stock_AP_h1
M3_expected_days_to_PC
```

Modelos híbridos:

```text
RF_RESIDUAL_M3
BAYES_RESIDUAL_M3
```

Ejemplo:

```text
residual = corte_real - pred_M3
modelo_nuevo predice residual
pred_final = pred_M3 + residual_predicho
```

Esto permite conservar estructura biológica y aprender sesgos sistemáticos.

Evalúa el híbrido con el mismo rolling origin.

---

# 21. Análisis de correlación y temporalidad

Antes de incorporar clima o podas al motor, construye una fase independiente de análisis temporal.

## 21.1 Unidad de análisis

Realiza análisis por:

```text
finca
bloque
finca agregada
```

No mezcles niveles sin ponderación.

## 21.2 Podas

Para cada lag `L`:

```text
corr(poda[t-L], corte[t])
```

Evalúa:

```text
Pearson
Spearman
cross correlation
regresión distributed lag
```

## 21.3 Clima

Para cada feature climática y ventana `W`:

```text
feature_clima[t-W:t]
```

Estudia relación con:

```text
corte[t]
transición RC->SS
transición SS->AP
transición AP->PC
```

No busques únicamente relación clima -> corte. El clima actúa sobre el desarrollo previo del tallo.

## 21.4 Control de multiplicidad

Si pruebas decenas de lags y variables, documenta el universo de tests.

Usa:

- Corrección por múltiples comparaciones para análisis inferencial.
- Validación temporal para decidir qué feature entra al modelo.

No selecciones una variable por el p value mínimo de una grilla grande.

---

# 22. Prevención de leakage

Regla central:

Para una predicción con origen `t0`, una feature es causal si su timestamp máximo es `<= t0`.

Implementa un validador automático:

```text
assert feature_max_timestamp <= fecha_origen
```

Casos críticos:

1. Clima observado posterior a t0.
2. Cortes reales de la semana objetivo.
3. Matrices fenológicas levantadas después de la fecha de backtest.
4. Estadísticos globales calculados usando train + validation.
5. Imputaciones que miran hacia el futuro.
6. Normalización entrenada con todo el dataset.
7. Selección de hiperparámetros usando holdout final.

Cada salida debe incluir:

```text
evaluacion_causal = True/False
fecha_max_dato_modelo
fecha_origen
```

---

# 23. Backtesting requerido

## 23.1 Rolling origin

Ordena semanas de origen cronológicamente.

Para cada corte temporal:

```text
TRAIN = semanas anteriores
VALIDATION = siguiente bloque temporal
```

No hagas random split.

## 23.2 Misma población

Todos los modelos deben compararse sobre la intersección de ventanas donde todos tienen predicción válida, además de reportar su cobertura propia.

Genera dos tablas:

```text
metricas_poblacion_comun
metricas_cobertura_propia
```

## 23.3 Niveles de métrica

Calcula:

```text
diario bloque
semanal bloque
semanal finca
global
```

## 23.4 Métricas

Principal:

```text
WAPE = sum(abs(pred-real)) / sum(abs(real))
```

Adicionales:

```text
MAE
RMSE
Bias_pct
Acierto_global = 1 - WAPE
Cobertura
```

No uses MAPE como métrica principal porque hay días con corte bajo o cero.

Para incertidumbre:

```text
bootstrap agrupado por bloque y semana
```

Reporta intervalos del WAPE si el tamaño de muestra lo permite.

---

# 24. Experimentos obligatorios

Construye una matriz de experimentos con identificadores estables.

Como mínimo:

```text
E00_M3_BASE
E01_M3_INGRESO_CALIBRADO
E02_M3_P32_RAW
E03_M3_P32_CANONICO
E04_M3_P32_REG
E05_SEMIMARKOV_P32_RC
E06_SEMIMARKOV_P32_RC_SS
E07_SEMIMARKOV_P32_ALL
E08_M3_PODA
E09_M3_CLIMA_CAUSAL
E10_M3_PODA_CLIMA
E11_RF_FENO
E12_RF_FENO_PODA
E13_RF_FENO_CLIMA
E14_RF_FULL
E15_RF_RESIDUAL_M3
E16_BAYES_M3_DIRICHLET
E17_BAYES_NB_HIER
E18_BAYES_SEMIMARKOV
E19_GLM_NB_BASE
```

Cada experimento debe registrar:

```text
experiment_id
model_family
features_version
data_version
train_start
train_end
validation_start
validation_end
causal
hyperparameters
seed
source_hashes
code_commit
```

---

# 25. Ablaciones obligatorias

No reportes únicamente el mejor modelo.

Realiza ablaciones para responder:

1. ¿Cuánto mejora pasar de M3 fijo a ingreso RC calibrado?
2. ¿Cuánto agrega P32 sobre M3?
3. ¿RC P32 agrega más que SS P32?
4. ¿AP P32 sigue degradando?
5. ¿Podas agregan señal después de controlar fenología?
6. ¿Clima agrega señal después de controlar fenología?
7. ¿GDD agrega más que temperatura media?
8. ¿VPD agrega más que HR?
9. ¿DLI agrega más que radiación media?
10. ¿Poda + GDD supera poda con lag calendario?
11. ¿Random Forest supera GLM NB?
12. ¿RF mejora al usar la predicción M3 como feature o residual?
13. ¿Bayes mejora WAPE o entrega principalmente mejor incertidumbre?

---

# 26. QA de datos obligatorio

Antes del modelado genera un reporte con:

## Identidad

```text
n_filas
n_fincas
n_bloques
n_fechas
n_semanas
```

## Duplicados

```text
Finca + Bloque + Fecha
Finca + Bloque + Semana en conteos
IdEstacion + FechaHora
Grupo + Etiqueta + Fecha en P32
```

## Cobertura

```text
camas activas
camas muestreadas
bloques sin muestreo
bloques sin conteo
semanas incompletas
```

## Fenología

```text
frecuencia_codigo_raw
frecuencia_codigo_canonico
codigos_pendientes
transiciones_no_canonicas
gaps
censuras
```

## Clima

```text
horas esperadas por dia
horas observadas
missing por variable
duplicados
saltos temporales
outliers físicos
```

## Podas

```text
Destinos únicos
Cantidad negativa
Cantidad cero
Cantidad_proy vs Cantidad
bloques no conciliados con plano
```

---

# 27. Contratos de nombres

Nunca dependas de nombres inconsistentes de Excel dentro de los modelos.

Crea nombres snake_case canónicos.

Ejemplos:

```text
Finca -> finca
Bloque / Block -> bloque
Fecha -> fecha
Cantidad -> corte_comercial_real en la fuente de cortes
Cantidad -> poda_cantidad en la fuente de podas
Cantidad_proy -> poda_cantidad_proy
Temperatura (°C) -> temperatura_c
Humedad(%) -> humedad_pct
Radiacion_Solar (w/m2) -> radiacion_w_m2
```

El significado debe depender de la fuente, no del nombre original aislado.

---

# 28. Diseño de módulos esperado

Adapta al repositorio existente, pero mantén separación de responsabilidades.

Arquitectura sugerida:

```text
src/
  config/
  ingestion/
    conteos_cortes.py
    camas_muestreadas.py
    plano_siembra.py
    fenologia_tradicional.py
    fenologia_p32.py
    podas.py
    clima.py
    calendario.py
  validation/
    schemas.py
    qa.py
    temporal_leakage.py
  normalization/
    fincas.py
    bloques.py
    fenologia.py
  features/
    conteos.py
    podas.py
    clima.py
    ventanas.py
  models/
    m3.py
    m3_p32.py
    semimarkov.py
    random_forest.py
    glm_nb.py
    bayes_m3.py
    bayes_nb.py
  evaluation/
    rolling_origin.py
    metrics.py
    comparison.py
  reporting/
    audit_tables.py
    plots.py
  tests/
```

No dupliques lógica ya correcta del repositorio. Refactoriza cuando exista deuda evidente.

---

# 29. Configuración

Todo parámetro experimental debe vivir en configuración.

Ejemplo:

```yaml
project:
  variety: FREEDOM
  target_farms:
    - ALMER
    - LA PRADERA
    - SANTA HELENA

forecast:
  horizon_days: 7
  week_start: MONDAY

m3:
  ingress_grid: [0.00, 0.05, 0.10, 0.12, 0.15, 0.20]
  periods:
    - name: ABRIL
      source: FENOLOGIAS ABRIL FREEDOM(3).xlsx
    - name: JULIO
      source: FENOLOGIAS JULIO FREEDOM(2).xlsx

climate:
  station_by_farm:
    LA PRADERA: 746
  gdd_bases: [5.0, 5.2]
  causal_windows_days: [1, 3, 7, 14, 21, 28, 42, 56]

pruning:
  lag_days_min: 28
  lag_days_max: 98
  priority_lags_days: [56, 63, 70, 77, 84]

random_forest:
  random_state: 42

bayes:
  seed: 42
```

El formato final depende de la arquitectura del repo, pero los valores no deben quedar dispersos en código.

---

# 30. Resultados y artefactos esperados

Cada corrida debe producir como mínimo:

```text
outputs/
  data_quality/
    qa_summary.csv
    qa_fenologia_codes.csv
    qa_climate_completeness.csv
    qa_join_coverage.csv

  datasets/
    fact_bloque_dia.parquet
    forecast_windows.parquet
    dataset_supervisado_diario.parquet
    transition_intervals_tradicional.parquet
    transition_intervals_p32.parquet

  models/
    model_manifest.json
    fitted_parameters.*

  predictions/
    predictions_daily.csv
    predictions_weekly.csv

  evaluation/
    metrics_by_experiment.csv
    metrics_common_population.csv
    metrics_by_farm.csv
    metrics_by_block.csv
    rolling_origin_results.csv

  analysis/
    poda_lag_analysis.csv
    climate_lag_analysis.csv
    feature_ablation.csv

  reports/
    REPORTE_COMPARACION_MODELOS.md
```

Si Parquet no forma parte del stack actual, usa un formato tabular eficiente ya soportado. No hagas de Excel el formato interno del pipeline.

Excel queda para auditoría y presentación.

---

# 31. Reporte de comparación

`REPORTE_COMPARACION_MODELOS.md` debe contener:

1. Población de evaluación.
2. Versiones de datos.
3. Modelos comparados.
4. WAPE, MAE, RMSE y sesgo.
5. Métricas por finca.
6. Métricas por semana.
7. Cobertura.
8. Ranking causal.
9. Ranking retrospectivo separado.
10. Análisis de errores grandes.
11. Ablaciones.
12. Impacto de podas.
13. Impacto de clima.
14. Importancia de features RF.
15. Posterior e incertidumbre Bayes.
16. Limitaciones.
17. Recomendación de champion y challengers.

No declares ganador únicamente por WAPE global. Revisa estabilidad, sesgo, cobertura y causalidad.

---

# 32. Criterios de promoción de un modelo

Un challenger entra a consideración de champion si cumple todos:

```text
WAPE validation menor que baseline
sesgo controlado
mejora en más de una ventana temporal
cobertura comparable
sin leakage
reproducible
tests aprobados
procedencia completa
```

Para modelos bayesianos añade:

```text
convergencia adecuada
posterior predictive checks aceptables
intervalos con cobertura razonable
```

No promociones un modelo por una semana excepcional.

---

# 33. Riesgos que el plan debe tratar explícitamente

1. Pocas semanas de historia.
2. P32 corto y levantado después de varias semanas del backtest.
3. Edad inicial desconocida en los conteos semanales.
4. Ingreso RC incierto.
5. AP con poco soporte y efecto directo sobre PC.
6. Clima exterior distinto del microclima bajo cubierta.
7. Lluvia exterior sin equivalencia directa con riego.
8. Podas con posible cambio de significado operacional por `Destino`.
9. Extrapolación que amplifica error de muestra.
10. Fincas con distinta escala productiva.
11. Bloques con poca historia.
12. Correlaciones espurias por estacionalidad.
13. Leakage en variables futuras.
14. Selección de features por entrenamiento.
15. Códigos fenológicos inconsistentes.
16. Diferencias de nombres de finca y bloque entre fuentes.
17. Seriales Excel y timestamps mixtos.
18. Solapamiento entre 2025.xlsx y 2026.xlsx.
19. DLI mal definido si se confunden joules con fotones PAR.
20. Random Forest con sobreajuste por baja muestra.
21. Modelos bayesianos jerárquicos demasiado complejos para el soporte disponible.

---

# 34. Orden recomendado de desarrollo

## Fase 0, auditoría y contratos

- Inventariar archivos.
- Validar esquemas.
- Homologar fincas, bloques, fechas y estados.
- Crear hashes de fuentes.
- Resolver contrato `SP`.
- Documentar significado de `conteo_CO`.
- Documentar unidad de podas.

Commit al cerrar la fase.

## Fase 1, capa canónica de datos

- Construir `fact_bloque_dia`.
- Construir camas activas y factor.
- Construir ventanas de forecast.
- Crear tests de joins y cardinalidad.

Commit al cerrar la fase.

## Fase 2, reproducir M3

- Reproducir baseline actual.
- Validar identidad contra salidas previas.
- Parametrizar ingreso RC.
- Ejecutar grid causal.

Commit al cerrar la fase.

## Fase 3, P32

- Normalizar seguimiento.
- Crear transiciones.
- Construir M3_P32 RAW, CANONICO y REG.
- Construir Semi Markov corregido.
- Resolver edad y x0.

Commit al cerrar la fase.

## Fase 4, podas

- Analizar lags.
- Crear features.
- Probar M3 + podas.
- Probar poda + GDD desde evento cuando clima esté listo.

Commit al cerrar la fase.

## Fase 5, clima

- Construir serie diaria estación 746.
- Calcular VPD, GDD, radiación y DLI correctamente.
- Validar causalidad.
- Analizar lags.
- Probar M3 + clima.

Commit al cerrar la fase.

## Fase 6, Random Forest y GLM

- Construir dataset supervisado.
- Ejecutar RF por ablaciones.
- Ejecutar GLM NB como control.
- Probar residual sobre M3.

Commit al cerrar la fase.

## Fase 7, Bayes

- M3 Dirichlet Multinomial.
- Regresión NB jerárquica.
- Semi Markov bayesiano si el soporte lo permite.

Commit al cerrar la fase.

## Fase 8, comparación final

- Rolling origin común.
- Métricas.
- Incertidumbre.
- Reporte final.
- Champion y challengers.

Commit al cerrar la fase.

---

# 35. Qué debe entregar el agente en MODO PLAN antes de implementar

Entrega un documento estructurado con:

## A. Diagnóstico del repositorio

```text
qué existe
qué sirve
qué está obsoleto
qué se reutiliza
qué se corrige
```

## B. Inventario real de datos

Para cada archivo:

```text
hojas
columnas
granularidad
rango temporal
n_filas
nulos
duplicados
claves
problemas de tipos
```

## C. Diccionario de homologación

Especialmente:

```text
fincas
bloques
microestados
macroestados
códigos pendientes
```

## D. Diseño de datasets

Incluye esquema exacto de:

```text
fact_bloque_dia
forecast_windows
dataset_supervisado_diario
transition_intervals_tradicional
transition_intervals_p32
clima_diario
poda_features
```

## E. Diseño matemático

Para cada modelo:

```text
inputs
target
ecuaciones
parámetros
supuestos
escala
salida
```

## F. Estrategia de causalidad

Indica exactamente qué timestamp máximo admite cada feature.

## G. Backtest

Define folds temporales concretos según las semanas disponibles.

## H. Plan de implementación

Lista archivos a crear o modificar.

## I. Tests

Lista tests unitarios, matemáticos, de integración y de leakage.

## J. Matriz de experimentos

Incluye todos los modelos y ablaciones.

## K. Riesgos y decisiones abiertas

No pidas al usuario que decida asuntos que el repositorio o los datos ya resuelven. Primero inspecciona y propone la opción técnicamente mejor.

---

# 36. Preguntas técnicas que el plan debe responder

1. ¿Cuál es la definición exacta vigente de M3 en el código y cómo se reproduce desde Abril y Julio?
2. ¿Qué regla vigente selecciona el conteo origen por semana?
3. ¿Cuál es el desfase exacto entre fecha de conteo y lunes de la semana objetivo?
4. ¿Qué representa `conteo_CO`?
5. ¿Qué códigos fenológicos quedan sin homologar?
6. ¿Cómo se resolverá `SP` aislado?
7. ¿Qué cantidad de P32 queda válida después del QA?
8. ¿Cómo se construirá `P(edad | estado)` para x0?
9. ¿Qué alpha de ingreso RC gana bajo inner validation causal?
10. ¿Qué lag de poda es estable y no resultado de estacionalidad?
11. ¿Qué variable climática agrega señal incremental después de fenología?
12. ¿GDD supera temperatura media?
13. ¿VPD supera HR?
14. ¿DLI agrega señal después de radiación?
15. ¿Poda + GDD desde evento supera lag calendario?
16. ¿RF diario pooled supera siete RF independientes?
17. ¿RF directo bloque supera RF tasa por cama?
18. ¿RF residual sobre M3 supera RF puro?
19. ¿Bayes M3 reduce inestabilidad de probabilidades P32?
20. ¿Bayes NB mejora calibración e incertidumbre aunque el WAPE sea similar?
21. ¿Qué modelo es champion causal y cuáles quedan como challengers?

---

# 37. Criterio científico

No fuerces una explicación fisiológica después de ver el resultado.

Antes de cada experimento declara:

```text
hipótesis
mecanismo esperado
ventana temporal esperada
variable observable
métrica de confirmación
```

Después compara evidencia a favor y en contra.

Ejemplo:

```text
Hipótesis:
mayor acumulación térmica después de una poda acelera el avance fenológico y reduce el tiempo hasta corte.

Feature:
GDD5 acumulado desde última poda hasta t0.

Validación:
mejora de likelihood de transición o reducción de WAPE en rolling origin.
```

---

# 38. Referencias agronómicas orientativas

Usa literatura como fundamento de hipótesis, no como sustituto del ajuste local.

Referencias útiles para revisar:

1. Pasian y Lieth, 1994, predicción del desarrollo de brotes de rosa mediante temperatura y unidades térmicas.
2. Estudios de Scientia Horticulturae sobre tasa de desarrollo de rosa bajo temperatura e irradiancia.
3. Mattson y Lieth, Acta Horticulturae, desarrollo anual de brotes de rosa y unidades térmicas.
4. Literatura de DLI en horticultura protegida. DLI se expresa en `mol de fotones PAR m-2 d-1`.
5. Estudios de transpiración y conductancia de canopia de rosa bajo distintos niveles de humedad y VPD.

El cultivo está bajo cubierta. Cualquier señal de estación exterior debe interpretarse como proxy hasta disponer de microclima interno.

---

# 39. Principio final de comparación

La pregunta central del proyecto no es:

```text
¿Cuál modelo es más sofisticado?
```

La pregunta es:

```text
¿Qué representación de la biología, el manejo y la incertidumbre mejora de forma causal y reproducible el pronóstico de corte comercial frente a M3?
```

Mantén siempre M3 visible en las tablas de comparación.

No ocultes experimentos negativos.

Una variante que empeora también aporta información sobre qué señal no está bien representada.

---

# 40. Primera respuesta esperada del agente

Tu primera respuesta, todavía en MODO PLAN, debe contener exactamente estas secciones:

```text
1. Diagnóstico inicial del repositorio
2. Inventario y calidad de las fuentes
3. Contratos de datos propuestos
4. Reconstrucción del baseline M3
5. Diseño M3 + podas
6. Diseño M3 + clima
7. Diseño M3 P32 por distribuciones
8. Diseño Semi Markov corregido
9. Diseño Random Forest
10. Diseño bayesiano
11. Diseño del dataframe maestro
12. Estrategia de backtesting causal
13. Matriz de experimentos
14. Tests requeridos
15. Arquitectura de módulos
16. Orden de implementación y commits
17. Riesgos abiertos
18. Decisiones que deben resolverse con evidencia de los datos
```

No escribas código productivo en esa primera respuesta.

Sí puedes incluir pseudocódigo, ecuaciones, esquemas y ejemplos de dataframe.

Después del plan, implementa por fases y valida cada bloque antes de continuar.
