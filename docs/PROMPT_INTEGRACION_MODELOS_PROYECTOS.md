# Prompt de ejecucion: integracion de modelos en los proyectos operativos

## Rol

Actua como arquitecto/a senior de datos, ML y backend Python. Debes trabajar
sobre el repositorio `markov-v3` y sobre los dos proyectos entregados en los
archivos ZIP presentes en la raiz. Tu trabajo es inspeccionar, diseñar,
implementar, probar y validar una integracion mantenible de los modelos de
pronostico de corte de rosas con los pipelines operativos existentes.

No asumas que los dos ZIP tienen la misma arquitectura. Debes conservar sus
responsabilidades actuales y añadir una frontera explicita entre:

1. Vision artificial: video, deteccion, conteo, QC, agregacion y exportacion.
2. Plataforma ALMER: API, persistencia, servicios de procesamiento,
   configuracion, modelos de segmentacion y proyecciones.
3. `markov-v3`: dataset canonico, modelos de pronostico, backtest causal,
   comparacion y seleccion de champion.

El resultado debe permitir añadir, retirar, reemplazar o desactivar modelos
sin reescribir el pipeline completo.

## Regla critica de ejecucion

Antes de cambiar codigo, descomprime los ZIP en carpetas separadas dentro de
la raiz del repositorio:

```text
integracion_externa/
  Vision-artificial/
  entrega_almer_20260826-2041/
```

Usa exactamente un directorio por proyecto. No sobrescribas `src/`, `config/`,
`data/` ni `outputs/` de `markov-v3`. Los ZIP originales son insumos y no deben
modificarse, descomprimirse encima de otro proyecto ni incluirse en el commit.
Si los directorios extraidos son solo material de trabajo, añadelos al
locales preexistentes.

## Contexto que debes respetar

Lee primero, en este orden:

1. `AGENTS.md`.
2. `config/pipeline.yaml`.
3. `docs/CONTRATOS_DATOS.md`, `docs/ARQUITECTURA_MODELO.md` y el prompt maestro
   si existen en el checkout.
4. `src/canonical.py`, `src/fase1_pipeline.py`, `src/models/`,
   `src/evaluacion_rolling.py` y `src/fase8_comparacion.py`.
5. Los README, manifiestos, configuraciones, entrypoints y tests de ambos
   proyectos descomprimidos.

El baseline M3 es obligatorio. Ningún modelo nuevo puede desplazarlo por
complejidad, velocidad o una métrica calculada sobre otra población. La
promoción se decide con WAPE en la población causal común de rolling-origin.
Todo parámetro experimental debe estar en `config/pipeline.yaml`; no
hardcodees semillas, lags, umbrales, vigencias, rutas ni hiperparámetros.

La restricción de leakage es inviolable: para un origen `t0`, todo dato o
feature debe tener timestamp disponible menor o igual a `t0`. Los modelos
retrospectivos deben marcarse como `RETROSPECTIVO_ORACLE_NO_CAUSAL` y excluirse
del ranking causal.

## Objetivos concretos

1. Levantar un mapa de ambos proyectos: entrypoints, flujo de datos, formatos,
   nombres de columnas, persistencia, contratos, dependencias y puntos donde
   se generan conteos, estados fenológicos, cortes o proyecciones.
2. Comparar ese mapa con el pipeline de `markov-v3` y señalar incompatibilidades
   reales, no hipotéticas.
3. Proponer y ejecutar la integración más orgánica, con cambios mínimos en los
   proyectos operativos.
4. Evaluar la integración de todos los modelos disponibles por familia y
   conservar un mecanismo declarativo para activarlos o retirarlos.
5. Seleccionar el mejor modelo por tipo, reportar su evidencia y distinguirlo
   del champion global.
6. Entregar pruebas unitarias, contract tests, pruebas de integración,
   pruebas de leakage y una validación reproducible de extremo a extremo.

## Inventario mínimo de modelos a evaluar

Construye un catálogo, no una cadena de `if` dispersos. Como mínimo incluye:

| Familia | Modelos o variantes |
|---|---|
| Mecanístico | M3 baseline, M3 con ingreso calibrado |
| Fenología | M3_P32 y Semi-Markov, solo como retrospectivos si usan información no causal |
| Extensiones | M3+poda, M3+clima, con nested rolling cuando corresponda |
| Supervisado | GLM Negative Binomial, Random Forest |
| Árboles extra | Extra Trees, HistGradientBoosting |
| Híbrido | Residual RF sobre M3, modelos específicos por horizonte |
| Bayes | Dirichlet-Multinomial, NB jerárquico y NB jerárquico con covariables |

No inventes resultados. Si un modelo no puede adaptarse por falta de datos,
dependencias o contrato causal, registra `NO_EVALUABLE` con causa y no lo
compares como si tuviera una métrica válida.

## Arquitectura objetivo

Adapta los nombres al estilo real descubierto, pero conserva estas
responsabilidades y contratos:

```text
config/
  pipeline.yaml                 # fuente única de parámetros
  model_registry.yaml           # modelos habilitados, familias y adaptadores

src/integration/
  __init__.py
  contracts.py                  # esquemas de entrada/salida y versión
  registry.py                   # descubrimiento y validación de modelos
  adapters.py                   # adaptadores operativos <-> markov-v3
  orchestration.py              # ejecución por modelo y por corrida
  provenance.py                 # hashes, timestamps, versión y lineage
  validation.py                 # causalidad, cobertura y rangos

src/models/
  ...                           # implementaciones existentes sin duplicarlas

tests/integration/
  test_contracts.py
  test_registry.py
  test_adapters.py
  test_pipeline_integration.py
  test_leakage_integration.py

docs/
  INVENTARIO_PROYECTOS_EXTERNOS.md
  MAPA_INTEGRACION_MODELOS.md
  VALIDACION_INTEGRACION_MODELOS.md
outputs/integration/             # generado, gitignored
```

El registro debe describir, como mínimo, `id`, `family`, `enabled`,
`implementation`, `adapter`, `features`, `causal`, `retrospective_reason`,
debe fallar silenciosamente ni alterar los demás.

La interfaz lógica de cada adaptador debe ser equivalente a:

```python
class ForecastModelAdapter(Protocol):
    model_id: str
    family: str
    causal: bool

    def validate_input(self, frame: pd.DataFrame) -> None: ...
    def fit(self, train: pd.DataFrame, context: RunContext) -> None: ...
    def predict(self, current: pd.DataFrame, context: RunContext) -> pd.DataFrame: ...
    def metadata(self) -> dict: ...
```

No fuerces los modelos de visión a fingir que son modelos estadísticos. La
visión debe producir un contrato intermedio versionado con, como mínimo,
`finca`, `bloque`, `fecha_origen`, `fecha_observacion`, `conteo`,
`fenologia`/estado disponible, `fuente`, `quality_status`, `timestamp_origen`
y `provenance_id`. El adaptador debe convertirlo al esquema canónico sin
perder trazabilidad.

## Fase 1: auditoría antes de implementar

Registra en `docs/INVENTARIO_PROYECTOS_EXTERNOS.md`:

- ruta de extracción y hash SHA-256 de cada ZIP;
- árbol de directorios relevante;
- comandos de instalación y entrypoints existentes;
- pipeline de video hasta conteo y exportación;
- pipeline de backend desde ingesta hasta proyección y API;
- archivos de configuración y precedencia de valores;
- modelos, pesos, formatos de persistencia y versiones;
- tablas SQLite/SQLAlchemy, endpoints y contratos frontend/backend;
- puntos de unión posibles con `fact_bloque_dia`, `forecast_windows` y los
  datasets Parquet de `markov-v3`;
- riesgos: columnas ambiguas, timezone, fecha de disponibilidad, unidades,
  duplicados, datos faltantes, reintentos y modelos no reproducibles.

No cambies nombres de negocio sin documentar un mapa bidireccional. Los
bloques son texto, las fincas deben normalizarse a ALMER, LA PRADERA y SANTA
HELENA, y los ceros de corte son observaciones reales.

## Fase 2: contratos y frontera de datos

Implementa o documenta contratos tipados y validables para:

1. Salida de detección/conteo.
2. Observación fenológica y conteo disponible en `t0`.
3. Ventana de pronóstico.
4. Predicción por día y por bloque.
5. Métricas y metadata de evaluación.

Cada fila de predicción debe conservar `model_id`, `family`, `features`,
`causal`, `run_id`, `config_hash`, `source_hashes`, `fecha_origen`,
`fecha_objetivo`, `finca`, `bloque`, `horizonte_dia`, `real`, `proyectado` y
`estado_ventana`. Rechaza columnas críticas ausentes, fechas inválidas,
predicciones negativas no permitidas y duplicados de la clave natural.

No hagas forward-fill silencioso de conteos. Transporta siempre
`fecha_conteo_origen` y `dias_desde_conteo`. Todo agregado debe guardar su
ventana temporal y disponibilidad.

## Fase 3: integración operativa

Elige la menor modificación compatible con cada proyecto:

- Preferir importar `markov-v3` como paquete o exponer una CLI/API interna
  estable antes que copiar archivos de modelos.
- Mantener los pipelines de visión y backend funcionando si el servicio de
  pronóstico está apagado.
- Añadir un servicio o comando explícito para construir el dataset canónico,
  entrenar/evaluar y publicar una predicción.
- Hacer idempotentes las corridas mediante `run_id`, hashes y manifest.
- Separar entrenamiento, evaluación y predicción operacional; la API no debe
  entrenar accidentalmente en una petición de usuario.
- Persistir el artefacto del modelo y su manifest con versión de código,
  configuración, features, datos y entorno.
- Si se requiere un modelo de visión para generar inputs, tratar sus salidas
  como datos observados con timestamp, calidad y procedencia, no como verdad
  absoluta.

La configuración debe permitir una selección como esta, sin cambios de código:

```yaml
model_registry:
  active:
    - E00_M3_BASE_ROLLING
    - GLM_NB_FENO_ROLLING
    - RF_FENO_ROLLING
    - EXTRA_TREES_FENO_ROLLING
    - HIST_GRADIENT_BOOSTING_FENO_ROLLING
  champion_policy:
    split: ROLLING_ORIGIN_COMMON
    primary_metric: wape
    baseline_required: E00_M3_BASE_ROLLING
  models:
    EXTRA_TREES_FENO_ROLLING:
      family: EXTRA_TREES
      enabled: true
      implementation: src.models.selection:estimator_from_spec
      adapter: supervised
      features: FENO
      causal: true
```

Usa los nombres y secciones existentes cuando ya haya un contrato equivalente;
no dupliques configuraciones por proyecto.

## Fase 4: evaluación del mejor modelo por tipo

Ejecuta el backtest causal rolling-origin sobre una población común. Para cada
familia produce:

- mejor modelo por WAPE diario;
- mejor modelo por WAPE semanal completo;
- MAE, RMSE, sesgo porcentual, R2 y volumen real;
- intervalo bootstrap del WAPE, agrupado por finca-bloque-fecha_origen;
- cobertura y número de ventanas válidas;
- resultado por finca, bloque y horizonte 1..7;
- estabilidad temporal y sensibilidad a faltantes/calidad de entrada;
- tiempo de inferencia, tamaño de artefacto y requisitos de memoria;
- diferencia contra M3 y decisión: `promocionar`, `challenger`, `no_gana`,
  `no_evaluable` o `excluido_no_causal`.

No uses MAPE como métrica principal. No mezcles datasets o poblaciones. Si un
modelo usa parámetros seleccionados previamente para evaluar rolling, corrige
el procedimiento con nested rolling o márcalo como no comparable. Mantén
separadas las métricas diarias y semanales, y explica cualquier discrepancia.

La promoción a champion requiere simultáneamente:

1. superar o justificar frente a M3 en WAPE de la población común;
2. no violar causalidad;
3. tener cobertura suficiente y sin degradación crítica por finca;
4. pasar los tests de contrato y reproducibilidad;
5. quedar registrado en un manifest versionado.

## Fase 5: pruebas obligatorias

Añade pruebas automatizadas para:

- extracción segura y detección de rutas/archivos faltantes;
- normalización de fincas y bloques;
- compatibilidad de columnas y tipos;
- timestamps: ninguna feature posterior a `t0` llega al entrenamiento;
- no forward-fill de conteos y propagación correcta de su origen;
- ceros, faltantes, duplicados y ventanas parciales;
- predicciones no negativas y esquema de salida estable;
- registro: activar/desactivar un modelo sin modificar orquestación;
- modelo desconocido o configuración inválida produce error claro;
- aislamiento entre corridas y reproducibilidad con la misma semilla;
- selección de población común entre todos los modelos;
- exclusión de modelos retrospectivos del ranking causal;
- integración mínima con cada proyecto extraído;
- persistencia y reconstrucción del artefacto seleccionado;
- API/CLI: no entrenar en modo predicción y devolver metadata de modelo;
- regresión de todos los tests existentes de los tres proyectos.

Usa fixtures pequeños y sintéticos para las pruebas rápidas. Usa datos reales
solo en una validación de integración controlada. No subas videos, pesos
grandes, bases de datos, Parquet generados, credenciales ni salidas locales.

## Fase 6: validación reproducible

Ejecuta, documenta y conserva el resultado de los comandos apropiados para
los tres proyectos, por ejemplo:

```powershell
python -m pytest tests -q
python -m pytest integracion_externa/Vision-artificial/tests -q
python -m pytest integracion_externa/entrega_almer_20260826-2041 -q
python -m pytest tests/integration -q
```

Adapta los comandos a los entrypoints reales encontrados. Ejecuta lint/type
check si ya existen. Ejecuta una corrida sintética end-to-end, una corrida con
los datos disponibles y una predicción operacional sin reentrenar. Valida que
los artefactos salgan en `outputs/integration/`, que puedan reconstruirse y
que el manifest sea suficiente para auditar la corrida.

Genera `docs/VALIDACION_INTEGRACION_MODELOS.md` con:

- comandos exactos y entorno utilizado;
- tests pasados y fallidos;
- tabla de modelos por familia;
- mejor modelo de cada familia y criterio usado;
- champion global, baseline M3 y challengers;
- métricas sobre población común;
- resultados por finca/horizonte;
- pruebas de leakage y su evidencia;
- incompatibilidades pendientes y su impacto;
- instrucciones de operación, rollback y desactivación de un modelo.

## Entregables esperados

Al finalizar, el árbol de cambios debe incluir solo lo necesario para la
integración y su documentación. Como mínimo:

- adaptadores, contratos, registro y orquestación;
- cambios de configuración centralizados;
- tests automatizados;
- documentación de inventario, arquitectura y validación;
- manifest de modelos y ejemplo de configuración;
- `.gitignore` actualizado para ZIP, directorios extraídos y outputs locales;
- sin copiar innecesariamente código de los proyectos externos;
- sin modificar ni incluir `Vision-artificial.zip` ni
  `entrega_almer_20260826-2041.zip`.

Antes de terminar, revisa `git diff`, `git status`, lista los archivos
versionados y comprueba explícitamente que los ZIP no están staged. No borres
ni reviertas cambios ajenos. Si una decisión arquitectónica no puede
validarse, deja la limitación escrita y no presentes una integración parcial
como terminada.

## Criterio de aceptación

La tarea solo está terminada cuando otra persona puede:

1. instalar las dependencias documentadas;
2. activar o quitar un modelo editando configuración;
3. ejecutar el pipeline desde datos de entrada hasta predicción;
4. saber qué datos y código produjeron cada resultado;
5. reproducir las métricas causales y el ranking;
6. distinguir el mejor modelo de cada familia del champion global;
7. comprobar que M3 sigue presente como baseline;
8. verificar que no hubo leakage;
9. ejecutar los tests y obtener resultados claros;
10. desactivar la integración sin romper visión, backend ni API.

Al responder, entrega un resumen factual de cambios, archivos creados o
modificados, resultados de pruebas, ranking por familia, riesgos pendientes y
el commit realizado. No afirmes que un resultado fue validado si no existe un
comando y un artefacto que lo respalden.
