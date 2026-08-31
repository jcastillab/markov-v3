# ARQUITECTURA_MODELO.md

## Auditoría técnica y metodológica del sistema de pronóstico Semi Markov Freedom

**Repositorio auditado:** `modelo_unificado_semimarkov_20260827(2)`  
**Rama encontrada:** `feature/modelo-unificado`  
**Último commit encontrado:** `14bf644 feat: generar excel interactivo de escenarios de ingreso`  
**Estado del árbol Git:** existen archivos modificados, eliminados y no versionados posteriores al último commit.  
**Suite ejecutada:** 56 pruebas, 56 aprobadas.  
**Validación de compilación:** el paquete y los tests compilan.  
**Control `git diff --check`:** detecta espacios finales en archivos modificados.  
**Criterio de esta auditoría:** se diferencia entre pipeline vigente, componentes experimentales, benchmarks y proyectos heredados de referencia.

## Hallazgos ejecutivos de auditoría

La arquitectura actual contiene una base estadística útil y una trazabilidad superior a versiones anteriores, pero todavía corresponde a un sistema de investigación aplicada, no a una solución de producción cerrada. El baseline M3 es el único modelo que el propio código clasifica como operacional y causal. Los challengers de Fase 16 dependen de fenología P32 levantada en agosto de 2026 y se evalúan retrospectivamente sobre semanas anteriores.

Los hallazgos de mayor impacto son los siguientes:

1. **Error lógico en la probabilidad usada para inferir edad latente.** En `modelo_unificado/fase2.py:289-296`, para eventos distintos de pérdida se calcula `loss + (1-loss)*P(evento|no pérdida)`. La probabilidad competitiva correcta para ese evento debe ser `(1-loss)*P(evento|no pérdida)`. La implementación actual hace que las probabilidades de eventos competidores excedan masa probabilística y altera el posterior de edad inicial.
2. **Ajuste climático con edad latente mal representada.** `modelo_unificado/fase16_clima.py:93-123` concatena intervalos de edad conocida y latente, pero los latentes conservan `bin_edad=None`. El baseline por estado y bin no tiene esa clave, por lo que esas filas parten de probabilidades cercanas a uniformes y se tratan como observaciones completas sin ponderación por `q_posterior`.
3. **Selección de variable climática por error de entrenamiento.** `modelo_unificado/fase16_clima.py:123` escoge el candidato con menor `nll_train`, aunque el código calcula `nll_validation`. Este criterio favorece sobreajuste. Los resultados actuales muestran un ejemplo claro: para SS→AP se selecciona humedad por entrenamiento, mientras otras variables presentan menor NLL de validación.
4. **El ingreso RC sigue fijo en 20% dentro de Fase 16.** `modelo_unificado/fase16.py:320-323` fija `0.20 * RC0` diariamente. El desarrollo posterior de auditoría ya explora 10% a 15% en `excel_auditoria.py:17`. La arquitectura productiva y la arquitectura de evaluación de escenarios no están sincronizadas.
5. **La inicialización semi Markov vigente concentra todo x0 en edad 0.** `modelo_unificado/fase16_evaluacion.py:10-14` coloca RC, SS y AP del conteo en bin `0`. Esto descarta la distribución de edad interna del stock observado y limita el beneficio real de un modelo dependiente de duración.
6. **Riesgo de escala en podas.** La variable `Cantidad_proy` no declara una unidad biológica verificable. La propia auditoría del repositorio lo reconoce en `fase16_auditoria.py:35-47`. El coeficiente se calibra con residual a escala bloque y luego entra al motor en escala muestral antes de volver a multiplicarse por el factor de extrapolación.
7. **Desalineación temporal de podas durante el lead.** En `fase16.py:339-343`, todos los días anteriores al primer día objetivo reutilizan `feature_day=0`. La trayectoria previa al horizonte recibe varias veces la señal correspondiente al primer día objetivo, no la señal histórica correspondiente a cada día simulado.
8. **Trazabilidad de fallback incorrecta.** `fase2.py:559-580` sobrescribe `fallback=False` para todas las filas de procedencia, incluso cuando `riesgos_competitivos_finales` marcó `M3_FALLBACK`.
9. **Fórmula de cola PI_OCC inconsistente.** `semimarkov.py:53-68` usa la supervivencia de edad 2 directamente como masa inicial de la cola `3+`, sin multiplicar primero por la permanencia de edad 2. Esto sobrepondera la cola cuando `PI_OCC` se usa.
10. **Fase 16 no reconstruye las fuentes operacionales desde datos crudos.** Consume `Fact_Conteos`, `Fact_Operacion_Diaria`, `Fact_Operacion_Semanal`, matrices y clima desde un Excel generado por el proyecto heredado. La trazabilidad completa depende de otra ejecución y otro código.

# 1. Resumen ejecutivo

## Objetivo general del proyecto

Pronosticar tallos de rosa `FREEDOM` que alcanzarán corte comercial durante un horizonte operativo diario y semanal, a partir de conteos fenológicos, dinámica de transición entre estados, duración dentro del estado, señales de poda, clima y factores de expansión de muestra a bloque.

## Problema de negocio que resuelve

El negocio necesita anticipar volumen de corte para planear mano de obra, empaque, logística, ventas y seguimiento agronómico. El conteo semanal entrega una fotografía parcial del cultivo. El sistema convierte esa fotografía en una trayectoria diaria esperada hasta corte.

## Variable objetivo

La variable final es:

`Corte comercial`, medido como número de tallos que alcanzan `PC` por bloque y día.

A nivel semanal:

`Corte semanal = suma de cortes diarios de lunes a domingo`.

En Fase 16 el campo de comparación es `CORTE_COMERCIAL_REAL` frente a `PC_dia_extrapolado`.

## Alcance actual del sistema

El repositorio contiene tres capas históricas:

* Un proyecto Markov multifinca heredado para `ALMER`, `PRADERA` y `SANTA HELENA`, que construye el data mart de Power BI, matrices M3, extrapolación y un challenger hazard climático.
* `implementacion_nueva`, que concentra la línea Semi Markov P32 y la comparación A a E.
* Herramientas de auditoría posteriores, entre ellas Markov P32, escenarios de ingreso RC y Excel detallado paso a paso.

La ruta más reciente de modelado consolidado es `fase16_cli`. Su configuración activa está centrada en `PRADERA`.

## Estado de madurez de la solución

**Madurez estimada: prototipo de investigación avanzado, con baseline operacional y challengers experimentales.**

Fortalezas:

* Conservación explícita de masa.
* Separación de estados terminales.
* Auditorías de calendario, cardinalidad, procedencia y composición.
* División temporal TRAIN y VALIDATION.
* Tests sobre reglas fenológicas, Fase 16, Markov P32 y Excel.
* Baseline M3 congelado y comparable.

Limitaciones para producción:

* Árbol Git sin cierre limpio.
* Sin paquete reproducible de dependencias para `implementacion_nueva`.
* Configuración incompleta y varios parámetros críticos hardcodeados.
* Challengers P32 no causales para el backtest histórico actual.
* Fase 16 depende de un data mart generado por un proyecto heredado.
* No existe registro formal de versión de datos, modelo, parámetros y artefactos.
* Hay errores lógicos que afectan edad latente, clima y trazabilidad.

## Resultado actual de Fase 16

Métricas presentes en `outputs/fase_16/resultado_final_fase16.csv`:

| Modelo | WAPE | Sesgo % | Estado metodológico |
| --- | ---: | ---: | --- |
| A | 40.27% | +11.56% | Operacional causal |
| B1 | 36.23% | -6.98% | Diagnóstico retrospectivo |
| B2 | 37.05% | -2.93% | Diagnóstico retrospectivo |
| B3 | 52.00% | +10.03% | Diagnóstico retrospectivo |
| C_MEJOR | 38.92% | -0.79% | Experimental retrospectivo |
| D1 | 79.55% | -79.33% | Experimental, transición climática no promovida |
| D2 | 79.55% | -79.33% | Benchmark futuro sin diferencia efectiva actual |
| E | 79.55% | -79.33% | Experimental, hereda fallo climático |

El mejor WAPE observado es B1. Ese resultado no constituye evidencia operacional causal porque su hazard P32 estaba disponible después de parte de las ventanas evaluadas.

# 2. Estructura del repositorio

## Árbol resumido

```text
raiz/
    README_INICIO.md
    INVENTARIO_REFERENCIAS.md
    PROMPT_OPENCODE.md
    opencode.json

    implementacion_nueva/
        README.md
        configuracion_pipeline.json
        configuracion_fase11.json
        configuracion_fase13.json
        configuracion_fase15.json
        configuracion_fase16.json

        modelo_unificado/
            config.py
            ingestion.py
            normalizacion.py
            trayectorias.py
            qa.py
            fase2.py
            m3.py
            semimarkov.py
            fase11.py
            fase13.py
            fase14.py
            fase15.py
            fase16.py
            fase16_evaluacion.py
            fase16_podas.py
            fase16_clima.py
            markov_p32.py
            dataio.py
            adapters.py
            alineamientos.py
            clima.py
            integracion.py
            backtest.py
            inventario.py
            *_cli.py
            excel_auditoria.py
            excel_detalle_modelos.py

        docs/
        tests/
        outputs/

    referencias/
        01_markov_hazard/
            modelo_markov/
            datos/
            scripts/
            tests/
            salidas/

        02_markov_alineamientos/
            modelo_markov/
            datos/
            Podas 10.xlsx

        03_contexto_propuesta/
        04_fenologia_duraciones_nueva/
            Fenologias13.08Final-1.xlsx
```

## Responsabilidad de cada bloque

### `implementacion_nueva/modelo_unificado`

Es el núcleo de investigación actual.

| Módulo | Responsabilidad | Estado |
| --- | --- | --- |
| `config.py` | Diccionario fenológico y procedencia P32 | Vigente |
| `ingestion.py` | Lectura determinista del XLSX de fenología individual | Vigente |
| `normalizacion.py` | Alias, microestado, macroestado y metadatos | Vigente |
| `trayectorias.py` | Episodios, salidas, censura, gaps | Vigente para QA |
| `qa.py` | Resúmenes de códigos, gaps y transiciones | Diagnóstico |
| `fase2.py` | Intervalos, edad conocida/latente, rho, posterior, hazards Semi Markov | Núcleo estadístico vigente |
| `m3.py` | Matrices oficiales M3, proveedor por vigencia, proyección Markov | Vigente |
| `semimarkov.py` | Motor por estado y edad, conservación de masa | Vigente |
| `fase16.py` | Orquestador A, B1, B2, B3, C, D1, D2, E | Entry point vigente |
| `fase16_evaluacion.py` | Motor diario común y métricas | Vigente |
| `fase16_podas.py` | Candidatos C1 a C5 y calibración ridge | Experimental vigente |
| `fase16_clima.py` | Modificadores logit de hazard por clima | Experimental vigente |
| `markov_p32.py` | Benchmark Markov empírico P32 | Benchmark retrospectivo |
| `excel_auditoria.py` | Escenarios 10% a 15% de ingreso RC | Auditoría vigente |
| `excel_detalle_modelos.py` | Trazabilidad de una semana | Auditoría, archivo no versionado en el ZIP |
| `fase16_auditoria.py` | Auditoría de escala y cardinalidad | Diagnóstico, contiene referencias obsoletas |
| `fase11.py` | Normalización de podas y análisis de correlación | Parcialmente reutilizado por Fase 16 |
| `fase13.py`, `fase14.py`, `fase15.py` | Etapas históricas de desarrollo | Legacy intermedio |
| `pipeline_cli.py` | Pipeline A a E original con parámetros provisionales | Obsoleto para corrida final |
| `adapters.py`, `alineamientos.py`, `clima.py`, `integracion.py`, `backtest.py` | Primera arquitectura genérica de challengers | Fuera del camino Fase 16 |
| `dataio.py` | Lector propio CSV/XLSX | Vigente |
| `inventario.py` | Inventario de fuentes | Diagnóstico |

### `referencias/01_markov_hazard`

Proyecto multifinca anterior. Es importante porque genera el archivo `modelo_datos_power_bi_multifinca.xlsx`, que Fase 16 usa como fuente materializada.

Responsabilidades:

* Leer fenologías históricas Abril y Julio.
* Construir M3 por finca y periodo.
* Preparar conteos y cortes.
* Calcular camas muestreadas y camas activas.
* Extrapolar muestra a bloque.
* Imputar bloques no muestreados a nivel finca.
* Construir clima diario desde estación horaria.
* Ajustar un hazard discreto multinomial con 8 variables.
* Exportar hechos y dimensiones para Power BI.

### `referencias/02_markov_alineamientos`

Proyecto heredado centrado en señales de poda y alineamiento. Su archivo `Podas 10.xlsx` sigue siendo insumo activo de Fase 16.

### `referencias/04_fenologia_duraciones_nueva`

Contiene el seguimiento P32 por tallo usado para construir duración y hazard Semi Markov.

## Dependencias entre módulos vigentes

```text
fase16_cli
    ↓
fase16
    ├→ dataio
    ├→ ingestion → normalizacion → config
    ├→ fase2 → normalizacion / trayectorias conceptuales
    ├→ m3
    ├→ semimarkov
    ├→ fase16_evaluacion → semimarkov
    ├→ fase16_podas
    ├→ fase16_clima → semimarkov
    ├→ fase11.normalizar_podas
    └→ fase14._date/_num/_yes
```

La dependencia de Fase 16 sobre helpers privados de Fase 14 es deuda técnica. Los helpers de tipos y parsing deben vivir en un módulo común.

# 3. Flujo completo del pipeline

## Flujo operativo real de la versión nueva

```text
Datos origen crudos
    ↓
Pipeline heredado Markov / Power BI
    ↓
Data mart Excel modelo_datos_power_bi_multifinca.xlsx
    ↓
Fact_Conteos + Fact_Operacion_Diaria + Fact_Operacion_Semanal
Matriz_Q + Vectores_r_p + Fact_Clima_Diaria
    ↓
Lectura dataio
    ↓
Selección de conteo semanal
    ↓
Construcción de ventanas operacionales de 7 días
    ↓
TRAIN / VALIDATION por semana
    ↓
Fenología P32 individual
    ↓
Ingesta → normalización → intervalos t→t+1
    ↓
Edad conocida / edad latente
    ↓
Hazards P32 suavizados con prior M3
    ↓
A / B1 / B2 / B3
    ↓
Podas C1-C5
    ↓
Clima D1 / D2
    ↓
Composición E
    ↓
Pronóstico diario muestral
    ↓
Factor de extrapolación
    ↓
Pronóstico diario de bloque
    ↓
Suma semanal
    ↓
WAPE / MAE / RMSE / sesgo
    ↓
CSV de auditoría y Excel
```

## Etapas

### Datos origen

Existen conteos, cortes, cama muestreada, plano de siembra, clima horario, podas y fenología individual. Sin embargo, Fase 16 no relee todos los originales.

### Ingesta

Hay dos mecanismos:

* `dataio.py` para tablas CSV/XLSX materializadas.
* `ingestion.py` para el XLSX P32, leído directamente desde XML dentro del ZIP XLSX.

### Validación

Se validan:

* Ventanas con `Orden_dia` 1 a 7.
* Fechas objetivo consecutivas.
* Conteos seleccionados.
* Sumas probabilísticas del motor.
* Conservación de masa.
* Códigos fenológicos y gaps mediante módulos QA.

Faltan contratos de esquema globales y validación tipada de cada fuente antes de entrar al pipeline.

### Transformación

Se normalizan finca, bloque, fecha, estados fenológicos, podas y matrices.

### Construcción de variables

Se generan macroestados, edad, bin de edad, eventos, exposiciones, posterior de edad, supervivencia, hazards, señales de poda y features climáticas rezagadas.

### Modelado

Se ejecutan ocho variantes A a E y benchmarks Markov P32.

### Pronóstico

El motor simula día a día desde el conteo hasta el final de la semana objetivo. Los ingresos entran a RC después de las transiciones del día.

### Evaluación

Se usa la misma población de ventanas de 7 días para todos los modelos. Fase 16 separa semanas en 60% TRAIN y 40% VALIDATION.

### Exportación

CSV para auditoría y XLSX para análisis interactivo.

# 4. Fuentes de datos

## Fuentes activas de Fase 16

| Fuente | Origen | Variables principales | Granularidad | Frecuencia | Uso |
| --- | --- | --- | --- | --- | --- |
| `modelo_datos_power_bi_multifinca.xlsx / Fact_Conteos` | Pipeline heredado | Finca, Bloque, Fecha, conteo_RC, conteo_SS, conteo_AP, camas activas, camas muestreadas, factor | Conteo por finca, bloque y fecha | Semanal, con más de un conteo ocasional | x0 y factor |
| `Fact_Operacion_Diaria` | Pipeline heredado | Fecha_conteo, Fecha, Orden_dia, Corte_real_bloque, Ventana_evaluable | Bloque, origen, día objetivo | Diario dentro de horizonte semanal | Calendario y target real |
| `Fact_Operacion_Semanal` | Pipeline heredado | Fecha_conteo, semana, corte real semanal | Bloque, origen, semana | Semanal | Conciliación del real diario |
| `Matriz_Q` | Pipeline heredado | origen, destino, probabilidad, finca, periodo | Estado origen/destino por finca y periodo | Por vigencia | M3 |
| `Vectores_r_p` | Pipeline heredado | permanencia transitoria, corte, pérdida | Estado por finca y periodo | Por vigencia | Completar Q+r+p |
| `Fact_Clima_Diaria` | Estación La Pradera Sunshine | temperatura, humedad, lluvia, radiación, VPD, ET0, DLI, GDD | Finca y día | Diario | D1, D2, E |
| `Fenologias13.08Final-1.xlsx` | Ensayo P32 | etiqueta, grupo, cama, fecha y estado observado | Tallo y día | Diario durante seguimiento | Duración y transiciones P32 |
| `Podas 10.xlsx` | Operación agrícola | Finca, Fecha, Destino, Variedad, Block, Cantidad_proy | Evento por bloque y fecha | Evento/diario | C1 a C5 y E |

## Conteos fenológicos

Fuente activa de Fase 16: `Fact_Conteos`.

Cobertura observada en el archivo auditado:

* 447 filas de datos.
* Fechas 25/04/2026 a 04/08/2026.
* ALMER: 65 filas.
* PRADERA: 334 filas.
* SANTA HELENA: 48 filas.

Regla de origen heredada: el último conteo de cada `Finca + Bloque + Semana` es el conteo seleccionado.

## Cortes reales

Fase 16 usa `Fact_Operacion_Diaria.Corte_real_bloque` y exige ventana evaluable de siete días.

El proyecto heredado construye los cortes desde `conteos_vs_cortes_multifinca.xlsx`, cuya granularidad cruda es `Finca + Bloque + Fecha` y contiene `Cantidad` junto con conteos RC, SS y AP.

`resumen_cortes_reales.xlsx` queda como fuente corporativa de control, no como sustitución silenciosa del real operacional.

## Datos climáticos

`Fact_Clima_Diaria` contiene 582 filas de datos, PRADERA, desde 01/01/2025 hasta 05/08/2026.

Variables:

* Temperatura media, máxima y mínima, °C.
* Humedad media, %.
* Lluvia, mm.
* Radiación media, W/m².
* VPD medio y máximo, kPa.
* ET0, mm.
* DLI, mol/m²/día.
* GDD base 5 °C.
* Indicadores de completitud climática.

## Camas muestreadas

Fuente original: `camas_muestreadas_semana.xlsx`.

Columnas:

`Semana, Finca, Bloque, Cantidad_csv`.

Fase 16 no lee ese archivo. Hereda `Camas_muestreadas` desde `Fact_Conteos`.

## Camas totales

Fuente original: `plano_siembra.xlsx`.

Variables principales:

`Finca, Sector, Flor, Variedad, Bloque, Cama, Fecha Siembra, Fecha Erradicacion, Plantas, Area Sembrada, Estado`.

El pipeline heredado cuenta camas activas a la fecha del conteo. Fase 16 hereda `Camas_totales_activas` y `Factor_extrapolacion` ya materializados.

## Ensayos fenológicos individuales

`Fenologias13.08Final-1.xlsx` contiene hojas:

* Garbanzo.
* Rayando 1.
* Separando S.
* Definiendo P.
* Graficas.

La primera fila contiene fechas de seguimiento. Cada tallo se identifica por `Etiqueta`. La fuente también contiene `Cama`, y en Garbanzo contiene `Tamaño (mm)`.

**Hallazgo:** `ingestion.py` conserva `id_tallo`, grupo, fecha y estado, pero descarta `Cama` y `Tamaño`. Esto impide conectar directamente el hazard P32 con bloque, cama o covariables de tamaño.

## Tablas de homologación

En el pipeline nuevo no existe una tabla externa de homologación. El contrato está hardcodeado en `config.py` mediante `MACRO_POR_MICRO` y `CANONICAL_ALIASES`.

Esto facilita reproducibilidad dentro del código, pero obliga a desplegar código para corregir una homologación y dificulta versionar reglas de negocio por fecha.

# 5. Modelo de datos

## Entidades principales

### Finca

Identifica unidad productiva. La versión nueva filtra una finca por configuración. El valor actual es `PRADERA`.

### Bloque

Unidad primaria para conteo, pronóstico, poda, extrapolación y evaluación.

### Conteo

Fotografía del stock fenológico en una fecha.

Campos centrales:

`Finca, Bloque, Fecha, conteo_RC, conteo_SS, conteo_AP, Factor_extrapolacion`.

### Ventana de pronóstico

Clave lógica:

`Finca + Bloque + Fecha_conteo`.

Debe producir exactamente siete días objetivo.

### Día objetivo

Clave lógica de salida:

`Modelo + Finca + Bloque + Fecha_origen + Dia_horizonte`.

### Tallo fenológico P32

Clave conceptual:

`Grupo_seguimiento + Id_tallo + Fecha`.

### Episodio de estado

ID construido como:

`grupo | tallo | estado | fecha_inicio_segmento`.

### Intervalo de transición

Representa únicamente `t → t+1`.

### Matriz M3

Clave conceptual:

`Finca + Periodo_modelo + Estado_origen + Estado_destino`.

### Clima diario

Clave:

`Finca + Fecha`.

### Poda diaria

Clave normalizada:

`Finca + Bloque + Fecha`.

## Relaciones

```text
Finca
  1 ─── N Bloque

Bloque
  1 ─── N Conteo
  1 ─── N PodaDia
  1 ─── N VentanaPronostico

Conteo
  1 ─── 1 VentanaPronostico

VentanaPronostico
  1 ─── 7 PronosticoDia por modelo

Finca + Periodo
  1 ─── N MatrizM3

P32 Tallo
  1 ─── N ObservacionFenologica
  1 ─── N Intervalo

Intervalos P32
  N ─── 1 Hazard por Estado + BinEdad

ClimaDia
  N ─── N Ventana mediante ventana temporal rezagada
```

## Riesgos de claves

* Fase 16 filtra finca antes de usar claves internas `(bloque, fecha)`. Esa clave no es globalmente única si se elimina el filtro.
* P32 no incorpora `Cama` en la observación normalizada.
* No existe validación explícita de duplicado para `Grupo + Tallo + Fecha` antes de estimar intervalos.
* Las claves del data mart se heredan del pipeline anterior sin versionado de dataset en Fase 16.

# 6. Reglas de negocio

## Homologación de estados

Macroestados actuales:

| Microestado | Macroestado |
| --- | --- |
| G, DG | PRE_RC |
| RC1, RC2, RC3, RC4, RC5, PE | RC |
| S1S, S2S, S3S, S4S, S5S | SS |
| SP1, SP1 1/2, SP2 | AP |
| PC | PC |
| X | L |

Alias:

| Entrada | Canonical |
| --- | --- |
| SP | SP1 |
| SP 1/2 | SP1 1/2 |
| SP 1 1/2 | SP1 1/2 |
| SP 2 | SP2 |
| SP1/2 | SP1 1/2 |

Códigos pendientes declarados en Fase 2:

`1 1/2P, 1P, 2P, 5S, R6, S1, SP 1`.

## Estados válidos

Estados transitorios del motor:

`RC, SS, AP`.

Preestado utilizado para conocer entrada a RC:

`PRE_RC`.

## Estados terminales

* `PC`: corte comercial.
* `L`: pérdida.

En Markov P32 ambos son absorbentes.

## Restricción fenológica canonical

Cadena esperada:

`PRE_RC → RC → SS → AP → PC`.

Pérdida `L` está permitida desde RC, SS y AP.

En Fase 2, `RC → AP` se captura como `OTHER`. Otras transiciones no canónicas se excluyen como `TRANSICION_NO_CLASIFICADA`.

## Tratamiento de pérdidas

Fase 2 estima una probabilidad de pérdida por estado, no por edad:

`loss_s = (D_loss + tau_loss * prior_loss) / (N + tau_loss)`.

Luego condiciona los eventos no pérdida por bin de edad y los transforma a riesgo competitivo multiplicando por `(1 - loss_s)`.

## Manejo de nulos y códigos desconocidos

Fenología:

* Código no reconocido → `estado_macro=None`.
* No se atraviesan gaps ni estados desconocidos.
* Esos intervalos se excluyen de hazards.

Operación:

* `_num` heredado de Fase 14 convierte vacío a `0.0`.
* Esta regla es riesgosa para targets reales si una fila marcada como evaluable contiene un vacío inesperado.

## Imputación

### Fenología individual

La nueva ruta no imputa días faltantes. Un gap distinto de un día genera censura de intervalo.

El proyecto heredado sí contiene reglas de relleno de vacíos aislados en fenologías históricas.

### Edad latente

Si no se observa la entrada canonical al estado, la edad inicial se trata como latente entre 0 y `age_max=6`, con un prior `rho` y posterior por likelihood.

### Bloques no muestreados

El pipeline heredado imputa pronóstico operacional por tasa ponderada por camas de bloques muestreados. Fase 16 no ejecuta esta capa.

## Extrapolación

Regla heredada de bloque:

`factor_teorico = camas_activas / camas_muestreadas`.

Si `camas_muestreadas > camas_activas`, se fija factor 1 y se audita.

Fase 16 toma el factor ya calculado desde `Fact_Conteos`.

## Construcción de cohortes

Semi Markov mantiene masa por `estado + bin de edad`.

Bins:

`0, 1, 2, 3+`.

Permanecer incrementa edad. Avanzar entra al siguiente estado en edad 0. La cola `3+` permanece en `3+`.

## Construcción de x0

Fase 16 toma el conteo seleccionado y construye:

`x0 = {RC: conteo_RC, SS: conteo_SS, AP: conteo_AP}`.

Luego ubica todo el stock de cada estado en edad 0.

## Restricciones fenológicas

* PC surge únicamente desde AP en Semi Markov canonical.
* L se trata como salida competitiva.
* OTHER se admite únicamente en RC.
* No se construyen intervalos sobre gaps.
* La masa siempre debe cerrar a uno en hazards y conservarse durante la propagación.

# 7. Ingeniería de variables

## Variables fenológicas

| Variable | Fórmula / construcción | Significado | Unidad | Código |
| --- | --- | --- | --- | --- |
| `estado_micro` | Alias canonical | Estado observado | Categoría | `normalizacion.py` |
| `estado_macro` | Mapa micro→macro | Etapa operacional | Categoría | `config.py` |
| `evento` | Comparación estado t y t+1 | STAY, avance, corte, pérdida, other | Categoría | `fase2.py` |
| `edad_original` | Días desde entrada observada al estado | Tiempo en estado | día | `fase2.py` |
| `bin_edad` | 0, 1, 2, 3+ | Duración discretizada | día/bin | `fase2.py` |
| `rho_observacion` | Prior sobre edad inicial latente | Incertidumbre inicial | probabilidad | `fase2.py` |
| `q_posterior` | rho × likelihood normalizado | Edad inicial posterior | probabilidad | `fase2.py` |
| `N_s_total` | exposición conocida + ponderada latente | Soporte estadístico | exposición equivalente | `fase2.py` |
| `D_s_k` | eventos ponderados | Conteo por evento | evento equivalente | `fase2.py` |
| `S_s` | producto de permanencias previas | Supervivencia dentro del estado | probabilidad | `fase2.py` |
| `h_evento` | riesgo competitivo | Probabilidad de salida/evento | probabilidad/día | `fase2.py` |

## Variables climáticas del proyecto heredado hazard

El hazard multinomial de `referencias/01_markov_hazard/modelo_markov/clima.py` usa:

1. `Edad_estado_dias`.
2. `Edad_estado_dias_cuadrado`.
3. `GDD5_acum_estado`.
4. `VPD_acum_estado_kPa_dia`.
5. `DLI_acum_estado_mol_m2`.
6. `Temperatura_media_3d`.
7. `Humedad_media_3d`.
8. `Lluvia_acum_3d_mm`.

### VPD

Presión de vapor de saturación:

`es = 0.6108 * exp(17.27*T / (T + 237.3))`.

VPD:

`VPD = es * (1 - HR/100)`.

Unidad: kPa.

### DLI

`DLI = suma(radiación positiva horaria) * 3600 / 1,000,000`.

Unidad exportada: mol/m²/día bajo la conversión usada por el proyecto.

### GDD5

`GDD5 = max(((Tmax + Tmin)/2) - 5, 0)`.

Unidad: °C día.

## Variables climáticas de Fase 16

Fase 16 ya no usa las ocho variables del hazard heredado. Construye candidatos simples de ventana:

| Nombre | Columna | Ventana | Lag |
| --- | --- | ---: | ---: |
| radiación | Radiacion_media_W_m2 | 28 días | 84 días |
| radiación | Radiacion_media_W_m2 | 28 días | 28 días |
| temperatura | Temperatura_media_C | 28 días | 28 días |
| humedad | Humedad_media_pct | 28 días | 28 días |
| lluvia | Lluvia_mm | 28 días | 28 días |

La función `feature()` calcula **media**, incluso para lluvia. Esto difiere de Fase 11 y del hazard heredado, donde lluvia se acumula mediante suma.

## Variables de duración

* `edad_original`.
* `bin_edad`.
* `S_s`.
* `h_total_s`.
* `h_advance_cut`.
* `h_loss`.
* `h_other`.

## Variables de extrapolación

* `Camas_totales_activas`.
* `Camas_muestreadas`.
* `Factor_extrapolacion_teorico`.
* `Factor_extrapolacion`.
* `Factor_ajustado`.
* `Cobertura_camas_pct` en el pipeline heredado.

## Variables de poda

`fase11.normalizar_podas` genera:

* `PODA_ALINEAMIENTO`.
* `PODA_CORTE`.
* `PODA_TOTAL`.
* `CORTE_PODA` como alias de poda, no corte comercial.

Fase 16 construye señales a lags 8 a 12 semanas y kernel promedio para C5.

# 8. Arquitectura estadística

## Modelo A, Markov M3

### Entradas

* x0 RC, SS, AP.
* Matriz Q oficial por finca y periodo.
* Vector de corte `r`.
* Vector de pérdida `p`.
* Ingreso RC diario.

### Salidas

* Stock diario RC, SS, AP.
* Corte PC diario.
* Pérdida L.
* Corte extrapolado a bloque.

### Parámetros

Matrices por periodo ABRIL/JULIO. Cambio de vigencia alrededor del 01/07/2026.

### Supuesto principal

La probabilidad del siguiente estado depende del estado actual, no de cuánto tiempo lleva el tallo en ese estado.

### Limitación

No representa duración explícita. Si la distribución de edad del conteo cambia entre semanas, la misma masa recibe la misma probabilidad.

## Modelo Semi Markov B

### B1

Hazard P32 en RC. SS y AP permanecen M3.

### B2

Hazard P32 en RC y SS. AP permanece M3.

### B3

Hazard P32 en RC, SS y AP.

### Entradas

* Conteo RC, SS, AP.
* Hazards por estado y bin de edad.
* Ingreso RC diario.
* Prior M3 para regularización.

### Salidas

Mismas salidas físicas del Markov, con memoria de duración.

### Parámetros

* Bins 0,1,2,3+.
* `tau=1` en Fase 16 para riesgos no pérdida.
* `tau_loss=10` por default, ya que Fase 16 no lo sobrescribe.
* `prior_other=0.05`.
* `m3_loss_floor=0.01`.
* `m3_category_floor=0.01`.

### Supuestos

* Duración discretizada en cuatro bins.
* Pérdida constante por estado.
* Edad latente restringida a 0..6.
* Hazard P32 transferible a todas las ventanas de PRADERA.

### Limitaciones

* x0 arranca en edad 0.
* Muestra P32 corta, 13 a 20 de agosto de 2026.
* Gran parte de exposiciones tiene edad latente.
* Transferencia temporal retrospectiva.

## Modelo C, Semi Markov + podas

Candidatos:

* C1: alineamiento lag 8 semanas.
* C2: corte de poda lag 8.
* C3: poda total lag 8.
* C4: alineamiento + corte lag 8.
* C5: poda total promedio entre lags 8 y 12.

Coeficientes no negativos mediante ridge coordinado.

La variante actual seleccionada es C1, con `alpha ≈ 8.49`.

Limitación central: unidad de `Cantidad_proy` no documentada y escala de calibración no alineada de forma estricta con el motor muestral.

## Modelo D1, modificador climático causal

Parte del M3 y ajusta logits de transición por una feature climática rezagada.

Transiciones ajustadas:

* RC→SS.
* SS→AP.
* AP→PC.

El modifier selecciona un `beta` por transición.

Resultado actual: WAPE cercano a 79.55%, claramente peor que A y B.

## Modelo D2, clima futuro retrospectivo

Permite ventanas climáticas que terminen después del origen. Con la configuración actual, los lags son tan grandes que D1 y D2 terminan usando exactamente la misma información. Por eso entregan las mismas métricas.

D2, en su forma actual, no aporta un benchmark futuro informativo.

## Modelo E

Composición:

`E = transiciones D1 + ingresos C seleccionado`.

El código audita identidad algebraica de sus componentes. El mal desempeño de D1 domina E.

## Markov P32

Benchmark empírico de transición diaria sobre las observaciones P32.

Variantes:

* `MARKOV_P32_RAW`.
* `MARKOV_P32_CANONICO`.
* `MARKOV_P32_REG`, suavizado alpha=0.5.
* Híbridos RC P32 + SS/AP M3.

Sirve para separar el beneficio de memoria de duración frente al beneficio de actualizar probabilidades de transición con P32.

## Hazard discreto multinomial heredado

El proyecto `referencias/01_markov_hazard` implementa un modelo distinto al modifier climático de Fase 16.

Forma general:

`P(Y=k|x) = softmax(eta_k)`.

`eta_k = beta_0k + beta_k' z`.

Clases:

* RC: STAY, ADVANCE, LOSS.
* SS: STAY, ADVANCE, LOSS.
* AP: STAY, CUT, LOSS.

Usa regularización L2 y optimización tipo Adam implementada manualmente.

Este modelo produce `Coeficientes_Hazard` y `Matriz_Clima_Efectiva` en el data mart heredado, pero Fase 16 **no consume esos coeficientes**. Ajusta su propio modifier simplificado.

# 9. Construcción de transiciones

## M3 oficial

La convención histórica es por columnas:

`Q[destino, origen]`.

Para cada estado origen:

`sum(Q[:,origen]) + r[origen] + p[origen] = 1`.

`provider_from_official_rows()` reconstruye la matriz a partir de `Matriz_Q` y `Vectores_r_p`.

## Markov P32

Se observan pares diarios consecutivos.

Conteo:

`N(i,j) = número de tallos con estado i en t y j en t+1`.

Probabilidad:

`P(i,j) = N(i,j) / sum_j N(i,j)`.

Estados PC y L se vuelven absorbentes.

La versión canonical excluye transiciones que violan la cadena fenológica.

## Semi Markov

Para cada intervalo se registra:

`estado, edad, evento`.

Exposición:

`N(s,d)`.

Evento:

`D(s,d,k)`.

Primero se estima pérdida por estado. Luego la distribución condicional de eventos no pérdida se suaviza hacia M3:

`p_final(k|no pérdida,s,d) = (D_k + tau * p_M3_k) / (N_no_perdida + tau)`.

Riesgo competitivo:

`h_k = (1 - loss_s) * p_final_k`.

Para pérdida:

`h_L = loss_s`.

## Incorporación de edad

Edad conocida existe si la entrada al estado fue observada desde su predecesor canonical en el día inmediatamente anterior.

Ejemplo:

`RC t-1 → SS t` implica edad SS=0 en t.

Si el seguimiento inicia dentro de SS, la edad se trata como latente.

## Estados absorbentes

PC y L no reingresan al stock activo.

# 10. Construcción de x0

## Fuente

`Fact_Conteos`.

Fase 16 filtra la finca y toma filas con `Conteo_seleccionado`.

## Regla de conteo seleccionado

La lógica heredada marca el último conteo de cada bloque y semana.

## Cálculo

```text
x0_RC = conteo_RC
x0_SS = conteo_SS
x0_AP = conteo_AP
```

El motor Fase 16 crea:

```text
(RC,0) = x0_RC
(SS,0) = x0_SS
(AP,0) = x0_AP
resto de bins = 0
```

## Imputación

Fase 16 no imputa x0 faltante. La ventana entra únicamente si existe clave de conteo seleccionada.

El pipeline heredado contiene lógica operacional para bloques no muestreados, pero esa lógica estima corte por tasa de finca, no construye un x0 Markov real para el bloque.

## Supuestos

* El conteo representa masa muestral.
* RC, SS y AP son mutuamente excluyentes.
* La fecha del conteo es t0.
* El factor de extrapolación representa la relación muestra/bloque.
* El motor Semi Markov trata todo el stock observado como recién entrado al estado.

## Casos donde falla

* Conteo ausente.
* Conteo parcial.
* Estado no homologado.
* Fecha sin ventana operacional completa.
* Factor faltante o inconsistente.
* Conteo con mezcla interna de edades que no se representa.

## Riesgos de sesgo

El principal es el último punto. Un conteo AP compuesto por tallos con 0, 2 y 5 días en AP no es equivalente a colocar todos en AP edad 0. La tasa de corte temprana queda sesgada hacia abajo si el hazard de corte aumenta con edad.

# 11. Integración climática

## Arquitectura heredada

El hazard climático original reconstruye clima diario desde estación horaria y usa variables biológicamente interpretables acumuladas desde la entrada al estado y ventanas de 3 días.

Durante pronóstico operacional causal, si la fecha climática es posterior al origen, usa un proxy basado en media de 14 días previos al conteo.

## Arquitectura Fase 16

Fase 16 usa features de 28 días con lag 28 u 84 días. Cada transición recibe un único beta seleccionado.

Ajuste:

1. Obtener hazard base por estado y bin.
2. Transformar probabilidades a logits.
3. Sumar `beta * z` al logit de la transición objetivo.
4. Aplicar softmax para renormalizar todos los eventos del estado/bin.

`z = (x - media_train) / std_train`.

## Ventanas temporales

D1 bloquea una feature si su ventana termina después de t0.

D2 la permite.

Con lags 28 y 84, los targets de una semana posterior al conteo siguen consultando ventanas que terminan antes de t0. Por ello D1 y D2 son iguales en la ejecución auditada.

## Problemas metodológicos

### Selección por TRAIN

La variable elegida minimiza `nll_train`. Debe seleccionarse con inner validation temporal o validación cruzada temporal.

### Edad latente

Los intervalos latentes conservan `bin_edad=None`, pero se introducen directamente al ajuste climático.

### Baseline contaminado respecto a split climático

Los hazards base se estiman usando toda la fenología P32 antes de dividir fechas para el ajuste climático. Por tanto, la validación interna del beta no tiene un baseline completamente independiente.

### Dependencia dentro del tallo

Varios intervalos provienen del mismo tallo. El ajuste los trata como observaciones independientes. El tamaño muestral efectivo es menor que el número de intervalos.

### Transferencia temporal

Los coeficientes se obtienen con P32 agosto 2026 y se aplican retrospectivamente a ventanas de meses previos.

# 12. Pronóstico

## Paso a paso desde un conteo semanal

### Paso 1. Seleccionar origen

Para `Finca + Bloque + Semana`, se toma el conteo seleccionado.

### Paso 2. Construir stock

Se leen RC, SS y AP del conteo.

### Paso 3. Resolver periodo M3

Antes de julio se usa ABRIL. Desde 01/07/2026 se usa JULIO.

### Paso 4. Elegir hazards

Según modelo:

* A: M3 en RC, SS y AP.
* B1: P32 RC, M3 SS/AP.
* B2: P32 RC/SS, M3 AP.
* B3: P32 RC/SS/AP.
* C: B2 + ingreso por poda.
* D1: M3 modificado por clima.
* D2: M3 modificado por clima con permiso futuro.
* E: D1 + ingreso C.

### Paso 5. Calcular lead

`lead = fecha_inicio_objetivo - fecha_conteo`.

### Paso 6. Simular desde t0+1

Cada día:

1. Tomar masa inicial por estado y edad.
2. Aplicar permanencia.
3. Aplicar avances RC→SS y SS→AP.
4. Aplicar corte AP→PC.
5. Aplicar pérdida L.
6. Aplicar OTHER desde RC.
7. Incrementar edad de permanentes.
8. Colocar avances en edad 0 del nuevo estado.
9. Añadir ingreso RC a `(RC,0)`.
10. Verificar conservación de masa.

### Paso 7. Alinear semana objetivo

`target_project()` simula `lead + 7` días y toma siete filas desde índice `lead - 1`.

Este ajuste fue introducido para corregir un off by one de fases anteriores.

### Paso 8. Corte muestral

`PC_dia_muestral = salida AP→PC`.

### Paso 9. Extrapolar a bloque

`PC_dia_extrapolado = PC_dia_muestral * Factor_extrapolacion`.

### Paso 10. Agregar semana

`Pronostico_semanal = suma(PC_dia_extrapolado, 7 días)`.

# 13. Extrapolación

## Muestra a bloque

Regla heredada:

`F = N / n`.

Donde:

* `N`: camas activas del bloque a t0.
* `n`: camas muestreadas.

Pronóstico:

`y_bloque = y_muestra * F`.

## Caso especial n > N

Si camas muestreadas supera camas activas:

`F = 1`.

El evento queda marcado como `MUESTREO_SUPERA_CAMAS_ACTIVAS`.

## Cambios de camas dentro del horizonte

El proyecto heredado calcula camas activas al inicio y fin de la semana objetivo para auditoría, pero conserva el factor anclado a la fecha del conteo.

## Bloques no muestreados

Método heredado:

`r_finca = suma(proyección bloques muestreados) / suma(camas activas de bloques muestreados)`.

Luego:

`proyección_bloque_no_muestreado = r_finca * camas_activas_bloque`.

Estas filas son operacionales y se excluyen de métricas Markov directas.

## Alcance en Fase 16

Fase 16 no reconstruye ni aplica esa segunda etapa. Evalúa ventanas con conteo seleccionado y factor ya materializado.

# 14. Evaluación

## WAPE

`WAPE = Σ|ŷ - y| / Σ|y|`.

Es la métrica principal.

Ventaja: pondera por volumen y evita que bloques pequeños dominen el resultado.

## MAE

`MAE = promedio(|ŷ - y|)`.

Unidad: tallos.

## RMSE

`RMSE = sqrt(promedio((ŷ - y)^2))`.

Penaliza errores grandes.

## Sesgo

`Sesgo_pct = Σ(ŷ - y) / Σ|y|`.

Positivo: sobrepronóstico neto.  
Negativo: subpronóstico neto.

El código también reporta `sesgo_absoluto = |Σ error| / Σ|y|`.

## Cobertura

Fase 16 retorna `cobertura=1` siempre que el conjunto de filas no esté vacío. Por tanto, esa métrica no mide proporción de ventanas disponibles respecto al universo original. La cobertura real se filtra antes mediante `Ventana_evaluable`.

Recomendación: reportar `n_ventanas_evaluables / n_ventanas_candidatas` por separado.

## MAPE

`MAPE = promedio(|ŷ - y| / |y|)` para filas con `y != 0`.

**No está implementado como métrica agregada en Fase 16.** El Excel sí calcula errores relativos para visualización diaria/semanal.

MAPE no debe ser métrica primaria aquí porque días o bloques con corte bajo amplifican el error relativo.

## Acierto

El núcleo define:

`Acierto = 1 - WAPE`.

Este acierto global no debe confundirse con semáforos diarios basados en error relativo.

## Unidad de evaluación

Fase 16 calcula métricas finales desde observaciones diarias, pero reporta `n_ventanas` como número de horizontes únicos `Finca + Bloque + Fecha_origen`.

## Horizonte

Siete días objetivo.

# 15. Backtesting

## Retrospectivo

Los challengers P32 usan hazard obtenido de la fenología P32 de agosto de 2026 para evaluar ventanas históricas anteriores.

Es útil para responder:

“Si estas probabilidades representaran la población histórica, ¿mejoraría el error?”

No responde:

“¿Qué rendimiento habría tenido el sistema con información disponible en t0?”

## Causal

El proyecto heredado contiene un backtest causal de matrices: cada conteo usa la última matriz disponible a su fecha.

En Fase 16, el propio resultado marca como `evaluacion_global_causal=True` únicamente a A.

## Split Fase 16

Semanas ordenadas:

* 60% inicial TRAIN.
* 40% final VALIDATION.

Para podas, TRAIN se divide otra vez:

* 70% SUBTRAIN.
* 30% INNER_VALIDATION.

Luego el alpha se refija en todo TRAIN.

## Posibles fugas de información

| Riesgo | Severidad | Descripción |
| --- | --- | --- |
| P32 posterior al t0 histórico | Alta | B1/B2/B3/C y E usan hazard disponible en agosto para semanas previas |
| Hazard base del ajuste climático estimado con todas las fechas P32 | Alta | La validación interna de clima hereda información del mismo conjunto usado para construir baseline |
| Selección de climate feature por `nll_train` | Alta | Selección directa sobre entrenamiento |
| D2 | Esperada por diseño | Permite clima posterior a t0, aunque con lags actuales no llega a usarlo |
| Selección C | Controlada en estructura | La variante se decide con inner validation, no con VALIDATION final |
| Data mart materializado | Media | Fase 16 no registra qué versión exacta del pipeline heredado creó las Fact_* |

## Recomendación de backtest

Implementar rolling origin con artefactos entrenados estrictamente hasta cada t0:

```text
Para cada semana t:
    entrenar matrices/hazards con datos disponibles <= t
    calibrar ingreso con historial <= t
    construir clima con información <= t
    predecir t+1
    almacenar predicción inmutable
```

# 16. Configuración y ejecución

## Entry point principal

```powershell
python -m modelo_unificado.fase16_cli configuracion_fase16.json
```

## Markov P32

```powershell
python -m modelo_unificado.markov_p32_cli configuracion_fase16.json
```

## Excel de escenarios de ingreso

```powershell
python -m modelo_unificado.excel_auditoria_cli configuracion_fase16.json
```

## Excel paso a paso

```powershell
python -m modelo_unificado.excel_detalle_modelos_cli configuracion_fase16.json
```

Este módulo aparece sin versionar en el ZIP auditado.

## Auditoría de Fase 16

```powershell
python -m modelo_unificado.fase16_auditoria_cli configuracion_fase16.json
```

## Tests

```powershell
python -m unittest discover -s tests
```

Resultado auditado: 56 tests aprobados.

## Configuración principal

`configuracion_fase16.json` define:

* fuente_conteos.
* fuente_operacion_semanal.
* fuente_operacion_diaria.
* fuente_matriz.
* fuente_fenologia.
* fuente_podas.
* fuente_clima.
* finca.
* salida_fase16.
* salida_markov_p32.

## Parámetros que no están correctamente externalizados

* Ingreso RC 20% de Fase 16.
* Tau de Fase 16, fijado a 1 en código.
* Tau loss efectivo, heredado del default 10.
* Cambio ABRIL/JULIO, fijado a 01/07/2026 en varios puntos.
* Candidatos y lags de poda.
* Candidatos y lags de clima.
* Regularización de clima.
* Split 60/40 y 70/30.
* Bins de edad.
* Week fija del Excel detallado.

## Pipeline obsoleto

`python -m modelo_unificado.pipeline_cli configuracion_pipeline.json` no representa la ruta final. `configuracion_pipeline.json` conserva stock inicial cero y fuentes opcionales vacías de una fase anterior.

# 17. Dependencias

## `implementacion_nueva`

No existe `requirements.txt` ni `pyproject.toml` propio.

Dependencias observadas:

| Biblioteca | Uso |
| --- | --- |
| Python standard library | CSV, JSON, fechas, XML, ZIP, dataclasses, math |
| `openpyxl` | Generación y lectura de Excel de auditoría y tests |

El lector principal de XLSX evita pandas y openpyxl para tablas del motor, leyendo XML directamente.

## Proyectos heredados

`referencias/01_markov_hazard/requirements.txt` y `referencias/02_markov_alineamientos/requirements.txt` declaran:

* numpy >= 1.26.
* pandas >= 2.1.
* openpyxl >= 3.1.
* xlsxwriter >= 3.2.

## Dependencias críticas

* Excel como formato contractual central.
* Estructura exacta de hojas `Fact_*`.
* Nombres de columnas del data mart heredado.
* Archivo P32 con estructura no tabular convencional.

# 18. Calidad de código

## Cohesión

La cohesión es buena dentro de módulos recientes:

* `fase16_evaluacion.py` concentra motor y métricas.
* `fase16_podas.py` concentra selección C.
* `fase16_clima.py` concentra modifiers climáticos.

`fase2.py` concentra demasiadas responsabilidades: construcción de intervalos, imputación probabilística de edad, regularización, supervivencia, sensibilidad y procedencia.

## Acoplamiento

Acoplamiento moderado a alto.

Ejemplos:

* Fase 16 importa helpers privados desde Fase 14.
* Fase 16 importa `normalizar_podas` desde Fase 11.
* Markov P32 importa funciones privadas de Fase 16 y Fase 14.
* Excel de auditoría ejecuta `markov_p32.ejecutar` repetidamente para cada porcentaje.

## Complejidad

Los mayores puntos de complejidad están en:

* `fase2.py`.
* `fase16.py`.
* `markov_p32.py`.
* Proyecto heredado `modelo_datos_power_bi.py`.

La complejidad no es únicamente ciclomática. Hay alta complejidad conceptual por mezcla de escala, causalidad, vigencias, estado, edad y extrapolación.

## Duplicación

Se repiten:

* `_date`, `_num`, `_yes` en varias fases.
* Writers CSV.
* Cálculo de métricas.
* División TRAIN/VALIDATION.
* Proyección Markov/Semi Markov con contratos similares.

## Tests

Distribución aproximada encontrada:

* `test_fenologia.py`: 30 tests.
* `test_fase16.py`: 18 tests.
* `test_markov_p32.py`: 5 tests.
* `test_excel_auditoria.py`: 2 tests.
* `test_excel_detalle_modelos.py`: 1 test.

Los 56 pasan.

No se obtuvo una cifra confiable de cobertura instrumentada dentro del límite de ejecución de auditoría. Por tanto, no se asigna porcentaje de cobertura.

## Componentes sin uso directo en Fase 16

* `pipeline_cli.py`.
* `integracion.py`.
* `adapters.py`.
* `alineamientos.py` en su arquitectura inicial.
* `clima.py` en su arquitectura inicial.
* `backtest.py` genérico.
* `fase13.py`.
* `fase15.py`.
* gran parte de `fase14.py`, salvo helpers usados indebidamente como utilidades.

No todo es código muerto. Parte funciona como historial experimental y tests. Sin embargo, no pertenece al grafo de ejecución principal.

## Bugs lógicos identificados

### BUG 1, probabilidad de evento en posterior de edad

Archivo: `fase2.py:289-296`.  
Severidad: **Alta**.  
Impacto: posterior de edad y hazards P32.

### BUG 2, fallback sobrescrito

Archivo: `fase2.py:559-580`.  
Severidad: **Alta para auditoría, baja para predicción**.  
Impacto: un bin que usa M3 fallback queda reportado como `fallback=False`.

### BUG 3, PI_OCC cola 3+

Archivo: `semimarkov.py:53-68`.  
Severidad: **Media/Alta cuando PI_OCC se active**.  
Impacto: masa excesiva en 3+.

### BUG 4, edad latente en clima

Archivo: `fase16_clima.py:93-123`.  
Severidad: **Alta**.  
Impacto: modifier climático mal especificado.

### BUG 5, selección climate por train

Archivo: `fase16_clima.py:123`.  
Severidad: **Alta**.  
Impacto: sobreajuste y selección inestable.

### BUG 6, parsing de columnas que contienen `HORA`

Archivo: `dataio.py:85`.  
Severidad: **Media**.  
La función trata cualquier encabezado que contenga `HORA` como fecha serial. Columnas como `Horas_registradas` y `Horas_temperatura_validas` entran en esa condición. Fase 16 no usa esos dos campos para el modifier actual, por lo que el pronóstico auditado no se altera por este punto.

### BUG 7, auditoría de C usa nombre obsoleto

Archivo: `fase16_auditoria.py:51`.  
Busca `modelo=='C'`, mientras el modelo vigente se llama `C_MEJOR`. El campo `ingreso_RC` de esa auditoría queda vacío.

### BUG 8, ruta AP obsoleta en auditoría

Archivo: `fase16_auditoria.py:55`.  
Busca `auditoria_AP.csv`, aunque el cierre de Fase 16 escribe `auditoria_AP_final.csv` como salida final.

# 19. Riesgos metodológicos

## 1. Edad latente domina el soporte

Fase 2 auditada:

* Intervalos observados: 1082.
* Válidos: 693.
* Edad conocida: 170.
* Edad latente: 523.
* Excluidos: 389.

La mayoría del soporte válido depende del modelo de edad latente. Cualquier error en rho, likelihood o límite de edad se propaga a los hazards.

## 2. Rho no identificable

`RHO_UNIFORME`, `RHO_POR_GRUPO` y `RHO_RESTRINGIDA_MICROESTADO` producen el mismo log likelihood en las salidas actuales. La configuración no contiene soporte microestado diferenciado.

## 3. x0 sin distribución de edad

El modelo aprende hazards por edad, pero no observa edad del stock semanal real. Este desacople es uno de los mayores límites conceptuales del Semi Markov actual.

## 4. Transferencia P32

Un único bloque/fuente P32 se usa como hazard para todas las ventanas de PRADERA. No se modela heterogeneidad por bloque, cama, manejo o fecha.

## 5. Ingreso RC

El ingreso diario puede dominar el stock RC a medida que avanza el horizonte. Un porcentaje fijo mal calibrado se propaga en todos los modelos.

## 6. Escala podas

La unidad de `Cantidad_proy` no está cerrada. Un alpha numéricamente grande no es interpretable sin unidad.

## 7. Clima exterior frente a cultivo cubierto

La estación representa ambiente externo. La relación con microclima bajo cubierta no está modelada explícitamente.

## 8. Sobreajuste climático

Pocas fechas P32, múltiples intervalos correlacionados por tallo, baseline ya estimado con toda la muestra y selección por TRAIN.

## 9. Binning 3+

Agrupa toda edad mayor o igual a 3. Si el hazard cambia entre 3, 5 y 8 días, esa información desaparece.

## 10. Pérdida constante por estado

No depende de edad, fecha, clima ni bloque.

## 11. Evaluación con un único holdout temporal

Seis semanas VALIDATION actuales. No existe distribución de rendimiento sobre múltiples cortes temporales rolling origin.

## 12. Métricas sin incertidumbre

No se reportan intervalos de confianza, bootstrap por semana/bloque ni error estándar.

## 13. Real condicionado a ventana completa

Es correcto para comparabilidad, pero el WAPE no describe rendimiento sobre semanas incompletas o futuras. La cobertura debe reportarse separadamente.

# 20. Deuda técnica

1. Crear paquete reproducible con `pyproject.toml` y lockfile.
2. Cerrar y etiquetar una release Git limpia.
3. Eliminar dependencia de helpers privados de fases históricas.
4. Centralizar parsing de fecha, número y booleano.
5. Convertir constantes hardcodeadas a configuración validada.
6. Crear contratos de esquema para cada fuente.
7. Versionar homologación fenológica como dato/configuración.
8. Integrar `Cama` del P32.
9. Integrar fecha/versión de disponibilidad de cada dataset.
10. Corregir posterior de edad.
11. Corregir fallback de procedencia.
12. Corregir PI_OCC.
13. Reescribir ajuste climático con posterior de edad ponderado.
14. Seleccionar clima con inner validation temporal.
15. Unificar hazard climático heredado y modifier Fase 16, o retirar uno de los dos.
16. Parametrizar ingreso RC en Fase 16.
17. Reconciliar escenarios 10% a 15% con el pipeline principal.
18. Definir unidad contractual de `Cantidad_proy`.
19. Corregir señal de poda durante días de lead.
20. Separar data preparation de model evaluation.
21. Construir pipeline raw→features→forecast en una única DAG reproducible.
22. Añadir MAPE si se requiere, con tratamiento explícito de cero.
23. Redefinir cobertura.
24. Añadir tests unitarios específicos para los bugs estadísticos detectados.
25. Retirar o archivar módulos históricos fuera del grafo productivo.
26. Corregir `fase16_auditoria.py` para `C_MEJOR` y `auditoria_AP_final.csv`.
27. Corregir `dataio.py` para diferenciar fecha/hora de conteo de horas.
28. Evitar redondear con `ceil` antes de métricas si se busca identidad exacta entre Excel y core.
29. Añadir hash de fuentes y configuración a cada salida.
30. Crear manifiesto de modelo con causalidad, fecha de entrenamiento y versión.

# 21. Recomendaciones

## Alta prioridad

| Área | Recomendación | Razón |
| --- | --- | --- |
| Estadística | Corregir `_probability_for_event` y regenerar todos los hazards P32 | Afecta el núcleo de edad latente |
| Estadística | Construir x0 por distribución de edad, no PI0 fijo | Es la condición necesaria para capturar el valor del Semi Markov |
| Datos | Conservar `Cama` y, si existe, bloque/tamaño en P32 | Permite enlace y heterogeneidad espacial |
| Estadística | Rehacer clima con ponderación posterior de edad y selección temporal | La versión actual está mal especificada |
| Datos/Negocio | Calibrar ingreso RC causalmente y parametrizarlo | 20% está hardcodeado y los escenarios recientes exploran 10% a 15% |
| Arquitectura | Hacer que el pipeline vigente reconstruya sus hechos desde fuentes originales o registre una versión inmutable del data mart | Elimina dependencia opaca |
| Estadística | Definir unidad y transformación exacta de podas antes de seguir calibrando alpha | Sin unidad no existe interpretación estable |
| MLOps | Crear release reproducible, lockfile, hash de fuentes y manifiesto | Requisito para comparar ejecuciones |

## Media prioridad

| Área | Recomendación | Razón |
| --- | --- | --- |
| Arquitectura | Extraer utilidades comunes de Fase 11/14 | Reduce acoplamiento histórico |
| Estadística | Evaluar bins de edad 0,1,2,3,4,5+ con regularización jerárquica | 3+ comprime mucha dinámica |
| Estadística | Rolling origin temporal | Rendimiento más estable y causal |
| Estadística | Bootstrap por semana/bloque | Cuantifica incertidumbre del WAPE |
| Datos | Contratos de esquema y validación de tipos | Evita silencios por `_num(None)=0` |
| Clima | Añadir microclima o proxies bajo cubierta | Mejora validez fisiológica |
| Rendimiento | Cachear lectura de XLSX y features por fecha | Excel de escenarios repite ejecuciones completas |

## Baja prioridad

| Área | Recomendación | Razón |
| --- | --- | --- |
| Código | Archivar fases 13, 14 y 15 después de preservar documentación | Reduce ruido del repositorio |
| Código | Unificar writers CSV | Menos duplicación |
| Reporting | Separar métricas core de redondeo Excel | Identidad numérica |
| Reporting | Añadir dashboard de procedencia y causalidad | Facilita auditoría |
| Rendimiento | Migrar data mart a Parquet o base tabular para ejecución interna | Menor costo I/O que Excel |

## Cinco componentes más críticos para mejorar WAPE

### 1. Ingreso RC diario

**Prioridad 1.**

Razón: se añade todos los días y alimenta la cadena completa. El pipeline principal fija 20%, mientras el módulo más reciente de escenarios estudia 10% a 15%. Un error sistemático en este término se acumula y altera RC, SS, AP y corte.

Acción:

* Estimar `alpha_ingreso` por rolling origin.
* Permitir alpha por finca, bloque/segmento o época únicamente si existe soporte.
* Comparar 0%, 5%, 10%, 12%, 15%, 20% con la misma población causal.

### 2. Distribución de edad de x0

**Prioridad 2.**

Razón: B1 mejora A aun con PI0, lo que indica señal en P32 RC. Una distribución de edad coherente puede trasladar esa señal al timing diario y evitar el sesgo de tratar todos los tallos como recién entrados.

Acción:

* Inferir `pi_stock` desde microestado del conteo, historial de conteos o distribución de ocupación validada.
* Corregir PI_OCC antes de promoverla.

### 3. Inferencia de edad latente y hazards P32

**Prioridad 3.**

Razón: 523 de 693 intervalos válidos son latentes. El bug de likelihood está exactamente en esa capa.

Acción:

* Corregir fórmula.
* Aumentar seguimiento que observe entradas canonical.
* Reducir dependencia de rho.
* Evaluar sensibilidad de `age_max`.

### 4. AP y probabilidad de corte

**Prioridad 4.**

Razón: B3 empeora con fuerza frente a B2. El salto WAPE de B2 37.05% a B3 52.00% indica que sustituir AP M3 por AP P32 actual degrada la predicción. AP es el estado directamente conectado con PC.

Acción:

* No promover AP P32 actual.
* Aumentar observaciones AP→PC.
* Revisar edad y censura de AP.
* Ajustar AP por microestados SP1, SP1 1/2 y SP2.

### 5. Extrapolación y representatividad de muestra

**Prioridad 5.**

Razón: el modelo puede acertar en muestra y fallar al escalar si camas muestreadas no representan distribución fenológica del bloque. WAPE se calcula en escala bloque.

Acción:

* Auditar error antes y después de factor.
* Estratificar representatividad por cama.
* Comparar factor de camas con expansión basada en densidad/tallos si existe ese dato.

## Sobre clima y podas dentro del top 5

No entran todavía entre los cinco primeros porque D1/D2/E empeoran drásticamente y C tampoco mejora B2 en validación. Antes de añadir complejidad, conviene corregir stock, edad, ingreso y AP. Clima y podas deben volver como challengers una vez estabilizado el núcleo.

# 22. Mapa final del sistema

## Flujo conceptual solicitado

```text
CONTEO FENOLÓGICO
RC, SS, AP por bloque
    ↓
SELECCIÓN DE CONTEO
último conteo válido de la semana
    ↓
x0
stock muestral inicial
    ↓
COHORTES POR EDAD
actualmente PI0 en Fase 16
    ↓
TRANSICIONES
M3 o hazards P32 por estado/edad
    ↓
SEMI MARKOV / HAZARD
permanencia, avance, pérdida, corte
    ↓
INGRESO RC
20% fijo en Fase 16 o escenarios externos
    ↓
CORTE DIARIO MUESTRAL
AP → PC
    ↓
CORTE DIARIO DE BLOQUE
multiplicación por factor de extrapolación
    ↓
CORTE SEMANAL
suma de 7 días
    ↓
EVALUACIÓN
WAPE, MAE, RMSE, sesgo
    ↓
AUDITORÍA
CSV, Excel, cardinalidad, causalidad
```

## Clasificación final de componentes

### Producción / operacional

**A, M3 oficial**

El código lo clasifica como `OPERACIONAL_CAUSAL`.

También se consideran operacionales como infraestructura:

* Selección de conteos desde `Fact_Conteos`.
* Calendario operacional desde `Fact_Operacion_Diaria`.
* Factor de extrapolación ya materializado.
* Motor diario y conservación de masa.

### Experimentales

* B1, B2, B3 Semi Markov P32.
* C_MEJOR, podas.
* D1, clima causal en ejecución pero hazard P32/climate fit retrospectivo en su evidencia.
* D2, clima futuro benchmark.
* E, composición podas + clima.
* PI_OCC.
* Ajuste climático Fase 16.

### Benchmarks

* MARKOV_P32_RAW.
* MARKOV_P32_CANONICO.
* MARKOV_P32_REG.
* Híbrido RC P32 + M3.
* Hazard climático heredado como referencia de arquitectura anterior.

### Auditoría y reporting

* `fase16_auditoria.py`.
* `excel_auditoria.py`.
* `excel_detalle_modelos.py`.

No deben confundirse con el motor estadístico.

## Riesgo principal para precisión por capa

```text
Datos de conteo
    ↓ representatividad y escala
x0
    ↓ distribución de edad ausente
Edad latente
    ↓ bug de likelihood y alto porcentaje latente
Hazards
    ↓ soporte P32 corto y transferencia temporal
AP → PC
    ↓ B3 muestra degradación fuerte
Ingreso RC
    ↓ 20% fijo y acumulativo
Clima / podas
    ↓ escala y selección aún inestables
Extrapolación
    ↓ amplifica error muestral
WAPE final
```

## Dictamen técnico

La mejor ruta de evolución no es añadir más señales al modelo actual. Primero debe cerrarse el núcleo causal y de estado:

1. Corregir edad latente.
2. Construir una distribución de edad para x0.
3. Recalibrar ingreso RC con backtest causal.
4. Mantener AP M3 hasta obtener soporte P32 suficiente.
5. Validar escala muestra→bloque.
6. Después reintroducir podas.
7. Finalmente volver a evaluar clima con diseño temporal correcto.

La evidencia del propio repositorio respalda este orden. B1 mejora A, B3 degrada B2, C no mejora B2 en validation y D1/D2/E degradan con fuerza. El mayor valor inmediato está en mejorar el mecanismo base de stock, duración, ingreso y corte, no en aumentar complejidad de covariables.

# Apéndice A. Registro de componentes y estado

| Componente | Rol | Causal actual | Recomendación |
| --- | --- | --- | --- |
| M3 A | Baseline oficial | Sí | Mantener como champion |
| B1 | Semi Markov RC | No en backtest actual | Reentrenar causalmente |
| B2 | Semi Markov RC+SS | No | Mantener challenger |
| B3 | Semi Markov RC+SS+AP | No | No promover |
| C_MEJOR | B2 + podas | Parcial, base no causal | Revisar escala antes de seguir |
| D1 | M3 + modifier clima | Feature causal, fit no cerrado | Rehacer entrenamiento |
| D2 | Benchmark clima futuro | No | Rediseñar o retirar |
| E | C + D1 | No | Pausar |
| Markov P32 | Benchmark | No | Conservar como ablación |
| Hazard legacy | Challenger histórico | Según modo | Referencia, no mezclar con Fase 16 |
| Excel escenarios | Auditoría | N/A | Conservar |

# Apéndice B. Acciones de corrección sugeridas antes de la siguiente comparación

1. Añadir test donde `sum(P(eventos)) == 1` para `_probability_for_event` a cada edad.
2. Regenerar `posterior_edad_inicial.csv` y comparar distribución antes/después.
3. Regenerar hazards P32 y repetir A/B1/B2/B3.
4. Corregir PI_OCC y crear identidad de masa esperada por edad.
5. Crear `porcentaje_ingreso_RC` en `configuracion_fase16.json` y eliminar `.2` hardcodeado.
6. Ejecutar grid causal de ingreso sobre TRAIN e inner validation.
7. Conservar Cama en ingesta P32.
8. Rehacer clima excluyendo intervalos latentes o expandiéndolos con peso posterior por bin.
9. Seleccionar features climáticas por inner validation temporal.
10. Corregir señales de poda en días de lead.
11. Formalizar unidad de `Cantidad_proy`.
12. Cerrar una release Git con tests y hashes de entradas.

