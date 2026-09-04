import numpy as np
import pandas as pd

from src.evaluacion_rolling import _common_observed, _selected_predictions
from src.reporte_excel import weekly_status


def test_rolling_semaphore_boundaries_remain_consistent():
    assert weekly_status(0.93) == "ACIERTO"
    assert weekly_status(1.10) == "CERCA"


def test_rolling_uses_selected_feature_group_and_family(monkeypatch):
    frame = pd.DataFrame({
        "finca": ["A"] * 6,
        "bloque": ["1"] * 6,
        "fecha_origen": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",
                                         "2026-01-04", "2026-01-05", "2026-01-06"]),
        "fecha_objetivo": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04",
                                           "2026-01-05", "2026-01-06", "2026-01-07"]),
        "semana_proyeccion": [1] * 6, "horizonte_dia": [1] * 6,
        "estado_ventana": ["VALIDA"] * 6, "target": [1., 2., 3., 4., 5., 6.],
        "base": np.arange(6.), "clima": np.arange(6.) + 10,
    })
    monkeypatch.setattr("src.evaluacion_rolling.feature_groups", lambda _: {
        "FENO": ["base"], "FENO_CLIMA": ["base", "clima"]
    })
    seen = []

    class Estimator:
        def fit(self, x, y):
            seen.append((list(x.columns), list(y.index))); return self
        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr("src.evaluacion_rolling.estimator_from_spec", lambda spec, cfg: Estimator())
    cfg = {"supervised": {"rolling_min_train_origins": 2}}
    spec = {"model": "RF_X", "family": "RF", "features": "FENO_CLIMA", "hyperparameters": {}}
    result = _selected_predictions(frame, cfg, spec)
    assert seen and all(columns == ["base", "clima"] for columns, _ in seen)
    for (_, training_indices), origin in zip(seen, sorted(frame.fecha_origen.unique())[2:]):
        assert (frame.loc[training_indices, "fecha_objetivo"] < origin).all()
    assert result["features"].eq("FENO_CLIMA").all()


def test_common_observed_population_has_identical_keys():
    keys = {"finca": ["A", "A"], "bloque": ["1", "1"],
            "fecha_origen": ["2026-01-01", "2026-01-02"],
            "fecha_objetivo": ["2026-01-02", "2026-01-03"], "horizonte_dia": [1, 1]}
    left = pd.DataFrame({**keys, "real": [1, 2], "proyectado": [1, 2]})
    right = pd.DataFrame({**keys, "real": [1, np.nan], "proyectado": [1, 2]})
    left_common, right_common = _common_observed(left, right)
    assert len(left_common) == len(right_common) == 1
