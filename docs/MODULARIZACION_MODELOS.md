# Modularizacion de modelos

La frontera portable vive en `src/modeling/` y no depende de este proyecto.

## Estructura

```text
src/modeling/
├── contracts.py      Interfaz ModelAdapter y helpers (nonnegative, features)
├── bundle.py         Persistencia content-addressed (joblib + manifest)
├── registry.py       Registro declarativo por familia y seleccion causal
├── extrapolation.py  Expansion validada camas muestreadas -> bloque
└── adapters/
    ├── m3.py         E00_M3_BASE_OPERATIONAL
    ├── rf.py         RF_OPT_FENO_OPERATIONAL, RF_H1_H7_FENO_OPERATIONAL
    ├── glm.py        GLM_NB_FENO_OPERATIONAL
    └── bayes.py      NB_JERARQUICO_OPERATIONAL, M3_DIRICHLET_MULTINOMIAL_OPERATIONAL
```

## Contrato

```python
class ModelAdapter(Protocol):
    family: str
    name: str
    features: list[str]
    def fit(self, train: pd.DataFrame, cfg: dict | None) -> "ModelAdapter": ...
    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...
    def metadata(self) -> dict: ...
```

Reglas del contrato:

- El target de scoring es siempre el total del bloque, no el conteo de la muestra.
- La extrapolacion se calcula antes del scoring y nunca se imputa en silencio.
- Las salidas son no negativas.
- Las features se declaran en orden en `features`; el scoring valida su presencia.

## Persistencia

`bundle.save_bundle(adapter, registry, extra)` crea:

```text
<registry>/<version>/
    modelo.joblib
    manifest.json
```

La version es `sha256(artefacto + manifest)[:16]`. El mismo contenido produce
la misma version; no se sobrescribe nada.

## Seleccion por familia

`registry.best_by_family(ranking_path)` lee `outputs/evaluation/ranking_final.csv`
y devuelve el mejor modelo causal de cada familia por WAPE rolling.

## Integracion a otro proyecto

1. Copie `src/modeling/` completo.
2. Provea un `pd.DataFrame` con las features declaradas y, para entrenar, la
   columna `target`.
3. Use `save_bundle` / `load_bundle` para persistir y reconstruir modelos.
4. `predict` devuelve el total de bloque esperado por fila.