# Reporte Fase 5 - Clima

## Alcance y causalidad

- Finca: `LA PRADERA`.
- Estacion: `746`.
- Fuente: `EXTERIOR_ESTACION`; no hay medicion de microclima de invernadero.
- Las features operacionales usan exclusivamente fechas `<= t0`.
- No se ejecuto un oracle con clima futuro.

## Procesamiento

- Se agregaron registros horarios a `clima_diario`.
- Lluvia y ET0 se suman; variables meteorologicas se agregan con estadisticos
  apropiados.
- VPD se calcula horario con la ecuacion de presion de vapor del contrato.
- Se calcularon GDD con bases 5,0 y 5,2 C.
- La radiacion se conserva como `solar_MJ_m2_d` y se estima
  `DLI_PAR_est_mol_m2_d` con los factores configurados (0,45 y 2,02), sin
  presentarlo como medicion directa bajo cubierta.

## Evaluacion

La validacion usa 476 dias evaluables de LA PRADERA. El baseline local obtiene
WAPE 49,04%. El mejor resultado exploratorio es `M3_CLIMA_TRANSICION` con
coeficiente 0,10 y WAPE 48,84%, una mejora de 0,16 puntos porcentuales.
El coeficiente fue explorado sobre esta validacion externa, por lo que no se
promueve sin una validacion interna y una prueba de estabilidad por bloque y
periodo.

La variante de ingreso no mejora el baseline. La variante de transicion mejora
ligeramente en el agregado, mientras la hibrida no lo hace. M3 continua siendo
el baseline oficial y el challenger queda etiquetado como exploratorio.

## Artefactos

- `clima_diario.parquet`
- `clima_features.parquet`
- `qa_clima.csv`
- `metrics_fase5_clima.csv`
- `clima_manifest.json`

Siguiente paso recomendado: Fase 6, GLM NB y Random Forest, usando M3 como
referencia y dejando clima como feature challenger documentada.
