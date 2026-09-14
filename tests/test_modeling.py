import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    GLMAdapter, RFAdapter, RFHorizonAdapter, load_bundle, save_bundle,
)


def _frame():
    rows = []
    for origin in pd.to_datetime(["2026-07-01", "2026-07-08"]):
        for horizon in range(1, 8):
            rows.append({
                "finca": "ALMER", "bloque": "1", "fecha_origen": origin,
                "fecha_objetivo": origin + pd.Timedelta(days=horizon),
                "horizonte_dia": horizon, "target": float(origin.day % 20 + horizon),
                "RC_t0": 10.0, "SS_t0": 20.0, "AP_t0": 30.0, "CO_t0": 1.0,
                "TOTAL_t0": 61.0, "p_RC": 0.1, "p_SS": 0.2, "p_AP": 0.3,
                "RC_AP_ratio": 1 / 3, "SS_AP_ratio": 2 / 3,
                "log1p_RC": 1.0, "log1p_SS": 2.0, "log1p_AP": 3.0,
                "M3_pred_muestra": 5.0, "M3_pred_bloque": 10.0,
                "camas_activas": 4.0, "camas_muestreadas": 2.0,
                "factor_extrapolacion": 2.0, "cobertura_muestreo": 0.5,
                "corte_dias_observados_28d": 28.0,
            })
    return pd.DataFrame(rows)


def test_rf_adapter_predicts_nonnegative():
    frame = _frame()
    model = RFAdapter({"n_estimators": 2, "max_depth": 2}).fit(frame)
    pred = model.predict(frame)
    assert pred.shape == (len(frame),)
    assert (pred >= 0).all()
    assert model.family == "RF"


def test_rf_horizon_adapter_covers_all_horizons():
    frame = _frame()
    model = RFHorizonAdapter({"n_estimators": 2, "max_depth": 2}).fit(frame)
    pred = model.predict(frame)
    assert (pred >= 0).all()
    assert set(model._estimators) == set(range(1, 8))


def test_glm_adapter_predicts_nonnegative():
    frame = _frame()
    model = GLMAdapter(alpha=1.0, max_iter=100).fit(frame)
    pred = model.predict(frame)
    assert (pred >= 0).all()
    assert model.family == "GLM_NB"


def test_bundle_round_trip_is_content_addressed(tmp_path):
    frame = _frame()
    model = RFAdapter({"n_estimators": 2, "max_depth": 2}).fit(frame)
    artifact, manifest = save_bundle(model, tmp_path)
    loaded, loaded_manifest = load_bundle(tmp_path, manifest["version"])
    np.testing.assert_allclose(loaded.predict(frame), model.predict(frame))
    assert loaded_manifest["family"] == "RF"
    assert loaded_manifest["artifact_sha256"] == manifest["artifact_sha256"]


def test_bundle_rejects_tampered_artifact(tmp_path):
    frame = _frame()
    model = RFAdapter({"n_estimators": 2, "max_depth": 2}).fit(frame)
    artifact, manifest = save_bundle(model, tmp_path)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact_sha256 no coincide"):
        load_bundle(tmp_path, manifest["version"])