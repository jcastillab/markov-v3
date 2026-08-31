# Contratos de datos - Fase 0 (cerrada con evidencia)

> Generado en Fase 0 a partir de `outputs/data_quality/contratos_fase0.md`
> e `inventario_fuentes.csv`. Cada contrato cita su evidencia.
> Leyenda: [OK] resuelto con evidencia. [PROP] propuesta que requiere
> confirmacion agronomica; el tratamiento por defecto es conservador.

---

## 1. Homologacion de fincas [OK]

Solo se modelan 3 fincas objetivo. Normalizacion previa al mapa:
`upper + strip + colapso de espacios`.

| finca_canonica | nombres crudos detectados | fuentes |
|---|---|---|
| `ALMER` | ALMER, Almer | conteos, camas, podas, plano, fenologia |
| `LA PRADERA` | PRADERA, Pradera, LA PRADERA, LA PRADERA BQ P11 | todas |
| `SANTA HELENA` | SANTA HELENA, Santa Helena, SANTA HELENA BL 7 | todas |

Notas:

- `LA PRADERA BQ P11` y `SANTA HELENA BL 7` (fenologia Julio) son
  seguimientos a nivel bloque (P11 y 7). Se conserva el bloque como
  atributo del seguimiento; la finca canonica es la indicada.
- En `camas_muestreadas` y `podas` existen ~20 fincas fuera de alcance
  con variantes de escritura (`inf`, `San Mguel`, `LA FUENTE SEM 21`).
  No se homologan; el join queda condicionado a la poblacion del modelo
  (prompt 4.2).
- `plano_siembra` trae mojibake `P�TALOS` (= PETALOS, fuera de alcance).

## 2. Bloques [OK]

- Identificadores de **texto**, siempre. Existen numericos (`1`..`60`)
  y codigos (`P11`, `P31`, ...). Nunca codificacion ordinal en modelos.
- Bloques Freedom con conteos: ALMER {1,2,3,4,9}; SANTA HELENA
  {1,2,3,4,7,9,12}; LA PRADERA {4,5,6,8,9,10,11,12,13,16,21,23,24,26,33,
  43,60} + {P11,P31,P32,P33,P35,P36,P39,P40,P41,P42,P43,P44,P46,P47,P48,P49}.
- Los bloques de podas (`Block`) y conteos (`Bloque`) coinciden en las 3
  fincas objetivo (verificado en sonda 3).

## 3. conteo_CO [OK]

**Evidencia**: en las 447/447 filas con conteo se cumple exactamente

```text
conteo_CO = conteo_total - (conteo_RC + conteo_SS + conteo_AP)
```

**Contrato**: `conteo_CO` es el stock residual contado que no esta en
RC/SS/AP (estados tempranos PRE_RC y/o no clasificados). Es derivado, no
medido. Mediana 35, max 1232 (~0.8% del total en mediana).
**Uso**: no entra a x0 de M3/Semi Markov; se conserva en `fact_bloque_dia`
y como feature candidata de modelos directos (p_CO).

## 4. Regla de conteo origen semanal [OK]

**Evidencia**: 445 combinaciones finca+bloque+semana con conteo; solo 2
tienen 2 conteos (PRADERA bloques 26 y 60, semana 202628, dias 06 y 08).

**Contrato**: para cada finca+bloque+semana el **ultimo conteo valido**
(fecha maxima) es el conteo origen. Regla reproducida tal cual; su impacto
es marginal (2/445) pero se mantiene por contrato historico.

## 5. Podas [OK con nota de unidad]

**Evidencia**:

- Destinos crudos (FREEDOM): CORTE 68.333, ALINEAMIENTO 4.289,
  ESTIMADO 54, mas variantes de escritura (`corte` 48, `Corte` 15,
  `ALINEAMIENTO ` con espacio 12). Se normalizan con upper+strip.
- En las 3 fincas objetivo (11.937 filas): `Cantidad == Cantidad_proy`
  en el 100%. Sin nulos, negativos ni ceros.
- Rango FREEDOM: 2026-01-01 a 2026-08-26.

**Contrato**:

```text
Destino = CORTE        -> poda_corte       (NO es corte comercial)
Destino = ALINEAMIENTO -> poda_alineamiento
poda_total = poda_corte + poda_alineamiento
Destino = ESTIMADO     -> excluido de features, contado en QA
```

**Unidad**: se usa `Cantidad` (igual a `Cantidad_proy`). Unidad de negocio
declarada como "tallos de la operacion" [PROP: validar con campo que
fraccion de la planta representa un corte de poda; afecta interpretacion
de coeficientes, no el signo del analisis de lags].

## 6. Camas muestreadas [OK]

**Evidencia**: 1.844 filas; exactamente 1 fila viola el patron
`Semana = S<digitos>`: `Resultados | S21 | El Cerezo | 16` (encabezado de
seccion incrustado). Se excluye por **regla de patron**, no por posicion.

**Contrato**: semanas validas S17-S32; columnas `Semana, Finca, Bloque,
Cantidad_csv` -> `semana, finca, bloque, camas_muestreadas`. Homologar
finca segun seccion 1 y filtrar a fincas objetivo en el join.

## 7. Clima [OK]

**Evidencia**:

- 19 estaciones por archivo; estacion PRADERA confirmada:
  `IdEstacion=746, La Pradera - Sunshine` (8.784 registros en 2025,
  5.207 en 2026 hasta 2026-08-05, como documento el prompt).
- `FechaHora` es **texto** en 2025 (166.896/166.896) y casi todo texto en
  2026 (98.932 str + 1 datetime). No hay seriales Excel numericos.
- Solape 2025/2026: 455 claves `IdEstacion+FechaHora` repetidas (texto).

**Contrato**: parseo de texto con dayfirst; si apareciera datetime se
acepta directo; deduplicar por `IdEstacion + FechaHora_parseada`
conservando la primera ocurrencia y contando el dedup en QA.

## 8. Fenologia: codigos y contrato SP [PROP]

### 8.1 Codigos crudos detectados (todos)

| codigo | abril/junio | julio | P32 | macro propuesto | estado |
|---|---:|---:|---:|---|---|
| G, DG | 2.318 | 308 | 346 | PRE_RC | OK |
| RC1..RC4 | 2.254 | 311 | 255 | RC | OK |
| RC5 | 0 | 0 | 46 | RC | OK (solo P32) |
| PE | 1.056 | 60 | 0 | RC | OK |
| S1S..S5S | 2.002 | 350 | 198 | SS | OK |
| SP1 | 457 | 77 | 51 | AP | OK |
| SP1 1/2 (+ SP1/2, SP 1 1/2) | 850 | 105 | 95 | AP | OK |
| SP2 (+ SP 2) | 345 | 48 | 27 | AP | OK |
| PC | 728 | 90 | 110 | PC | OK |
| X | 40 | 0 | 55 | L | OK |
| SP 1 | 0 | 0 | 2 | AP (alias SP1) | OK |
| **SP aislado** | 0 | 0 | **160** | **PENDIENTE** | **PROP** |
| SP2 1/2 | 3 | 0 | 0 | PENDIENTE | PROP |
| R4C | 0 | 1 | 0 | PENDIENTE | PROP |
| S1 | 0 | 1 | 1 | PENDIENTE | PROP |
| 1P | 0 | 0 | 1 | PENDIENTE | PROP |
| 1 1/2P | 0 | 0 | 1 | PENDIENTE | PROP |
| 2P | 0 | 0 | 4 | PENDIENTE | PROP |
| R6 | 0 | 0 | 1 | PENDIENTE | PROP |

### 8.2 Evidencia sobre SP aislado

- 160 ocurrencias, solo en P32: 84 en hoja `Definiendo P`, 71 en
  `Separando S`, 5 en `Rayando 1`.
- En `Definiendo P`, SP es el estado de etiquetado inicial y progresa a
  microestados AP: p.ej. tallo 151 `SP -> SP 1/2 -> SP1 -> PC`,
  tallo 153 `SP -> SP 1 1/2 -> SP 2 -> PC`.
- El archivo historico (abril/julio) **nunca** usa SP aislado: siempre
  con fraccion. Esto sugiere que SP aislado es convencion reciente del
  levantamiento P32 para "separando petalos sin fraccion asignada".

### 8.3 Decision de contrato (tratamiento por defecto)

1. `SP` aislado **no** se homologa a `SP1` (prohibido por prompt 4.6).
2. Se registra como microestado propio `SP_SIN_FRACCION` con
   `macro_propuesto = AP`, `estado_QA = PENDIENTE_CONFIRMACION`.
3. Los intervalos que tocan `SP_SIN_FRACCION` **no entran** a la
   estimacion de hazards/transiciones; se reportan en QA
   (`transiciones_excluidas_codigo_pendiente`).
4. Consecuencia conocida: reduce los intervalos usables P32 por debajo de
   las 200 etiquetas crudas (la implementacion vieja los absorbia en
   silencio via SP->SP1). Esta reconciliacion queda documentada.
5. Si agronomia confirma que SP = microestado AP inicial (anterior a
   SP1), se activa como microestado valido cambiando solo configuracion.

## 9. Ceros de corte comercial [OK]

Solo 1 dia con corte = 0 en 4.951 filas; sin nulos ni negativos.
Se conserva como observacion real (prompt 5.7).

## 10. Estructuras de ingesta confirmadas

- `FENOLOGIAS ABRIL/JULIO`: ancho fijo `FINCA | DIA | FECHA | 1..30`
  (30 tallos). Hojas: `Abril` (21 fincas, 2026-04-07 a 2026-06-03),
  `Junio` (solapa con Abril, 2026-05-26 a 2026-06-16), `JULIO`
  (3 fincas-bloque, 2026-07-13 a 2026-08-08).
- `Fenologias13.08Final-1` (P32): 4 hojas de seguimiento
  (Garbanzo, Rayando 1, Separando S, Definiendo P). Fila 0 = encabezado
  con fechas; columnas `Etiqueta, Fenologia, [Tamano], Cama`, luego una
  columna por fecha (Garbanzo: pares Fenologia+Tamano por fecha).
  `Graficas` no es fuente.
- Vigencias M3 (Abril vs Julio): las fuentes solapan en calendario; la
  regla de vigencia operativa (que matriz aplica a que fecha origen) se
  define en Fase 2 con configuracion explicita, sin fechas hardcodeadas.
