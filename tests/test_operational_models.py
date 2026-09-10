import hashlib
import json

import numpy as np
import pandas as pd

from src.dashboard_validation import (load_latest_operational_run,
                                      operational_model_comparison)
from src.models.operational import (build_operational_windows,
                                    load_operational_models,
                                    save_operational_models,
                                    score_operational_models,
                                    train_operational_models, training_rows,
                                    validate_scoring_cutoff)
from src.proyeccion_vision import _operational_run_id, _weekly_predictions


def _cfg():
    params = {
        "n_estimators": 2, "max_depth": 2, "min_samples_leaf": 1,
        "min_samples_split": 2, "max_features": 1.0,
        "criterion": "squared_error", "random_state": 42, "n_jobs": 1,
    }
    return {
        "forecast": {"horizon_days": 7},
        "vision": {"conteo_co_semantics": "CORTE_INMEDIATO_VISUAL"},
        "operational": {
            "minimum_history_days_28d": 14,
            "models": {
                "rf_feno": {"name": "RF_FENO", "hyperparameters": params},
                "rf_h1_h7": {"name": "RF_H", "hyperparameters": params},
            },
        },
    }


def _training_frame():
    rows = []
    for origin_number, origin in enumerate(pd.to_datetime(["2026-07-01", "2026-07-08"])):
        for horizon in range(1, 8):
            rows.append({
                "finca": "ALMER", "bloque": "1", "fecha_origen": origin,
                "fecha_objetivo": origin + pd.Timedelta(days=horizon),
                "semana_proyeccion": 202628, "horizonte_dia": horizon,
                "target": float(origin_number + horizon), "RC_t0": 10.0,
                "SS_t0": 20.0, "AP_t0": 30.0, "CO_t0": 1.0, "TOTAL_t0": 61.0,
                "p_RC": 0.1, "p_SS": 0.2, "p_AP": 0.3, "p_CO": 0.01,
                "RC_AP_ratio": 1 / 3, "SS_AP_ratio": 2 / 3,
                "log1p_RC": 1.0, "log1p_SS": 2.0, "log1p_AP": 3.0,
                "M3_pred_muestra": 5.0, "M3_pred_bloque": 10.0,
                "camas_activas": 4.0, "camas_muestreadas": 2.0,
                "factor_extrapolacion": 2.0, "cobertura_muestreo": 0.5,
                "corte_dias_observados_28d": 28.0,
            })
    return pd.DataFrame(rows)


def test_training_cutoff_excludes_future_targets():
    frame = _training_frame()
    selected = training_rows(frame, pd.Timestamp("2026-07-10"))
    assert selected["fecha_objetivo"].max() <= pd.Timestamp("2026-07-10")
    assert len(selected) < len(frame)


def test_pooled_and_horizon_models_score_without_target():
    frame = _training_frame()
    bundle, manifest = train_operational_models(frame, _cfg())
    scoring = frame.iloc[:7].copy()
    scoring["target"] = np.nan
    result = score_operational_models(scoring, bundle, _cfg())

    assert set(result["modelo"]) == {"RF_FENO", "RF_H"}
    assert len(result) == 14
    assert result["proyectado"].notna().all()
    assert manifest["training_rows"] == 14


def test_missing_cut_history_blocks_rf_instead_of_zero_imputation():
    frame = _training_frame()
    bundle, _ = train_operational_models(frame, _cfg())
    scoring = frame.iloc[:7].copy()
    scoring["corte_dias_observados_28d"] = 0
    result = score_operational_models(scoring, bundle, _cfg())

    assert result["proyectado"].isna().all()
    assert result["motivo"].eq("HISTORIAL_CORTES_INSUFICIENTE").all()


def test_operational_windows_target_next_iso_week():
    counts = pd.DataFrame([{
        "finca": "ALMER", "bloque": "1", "fecha_conteo": pd.Timestamp("2026-09-01"),
        "semana_iso": 202636, "conteo_rc": 1, "conteo_ss": 2, "conteo_ap": 3,
        "conteo_corte_inmediato_visual": 4, "conteo_total": 10,
        "camas_muestreadas": 2, "camas_activas": 4, "factor_extrapolacion": 2,
    }])
    windows = build_operational_windows(counts, 7)
    assert windows["fecha_objetivo"].min() == pd.Timestamp("2026-09-07")
    assert windows["fecha_objetivo"].max() == pd.Timestamp("2026-09-13")
    assert windows["semana_proyeccion"].eq(202637).all()


def test_weekly_prediction_remains_when_real_is_unavailable():
    daily = pd.DataFrame({
        "modelo": ["RF"] * 7, "finca": ["ALMER"] * 7, "bloque": ["1"] * 7,
        "fecha_origen": [pd.Timestamp("2026-09-01")] * 7,
        "semana_proyeccion": [202637] * 7, "proyectado": np.arange(1, 8),
        "estado_modelo": ["DISPONIBLE"] * 7, "motivo": [""] * 7,
    })
    weekly = _weekly_predictions(daily)
    assert weekly.loc[0, "proyectado"] == 28


def test_dashboard_loads_latest_immutable_run(tmp_path):
    runs = tmp_path / "outputs" / "operational"
    run = runs / "s36-abcdef123456"
    run.mkdir(parents=True)
    pd.DataFrame({"modelo": ["RF"], "proyectado": [1]}).to_parquet(
        run / "predicciones_diarias.parquet", index=False)
    pd.DataFrame({"modelo": ["RF"], "proyectado": [7]}).to_parquet(
        run / "predicciones_semanales.parquet", index=False)
    for name in ("comparacion_modelos.xlsx", "conteos_consolidados.parquet", "qa.csv",
                 "videos_validos.parquet"):
        (run / name).write_bytes(b"fixture")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in run.iterdir()}
    dependencies = {"fixture": "value"}
    digest = hashlib.sha256(json.dumps(
        dependencies, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    run_id = f"s36-{digest[:12]}"
    renamed_run = runs / run_id
    run.rename(renamed_run)
    run = renamed_run
    manifest = {"run_id": run_id, "semana_entrada": "S36",
                "dependencies": dependencies, "output_sha256": hashes}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
    (runs / "latest.json").write_text(
        json.dumps({"run_id": run_id, "manifest_sha256": manifest_hash}), encoding="utf-8")

    daily, weekly, manifest = load_latest_operational_run(tmp_path, "outputs/operational")
    assert daily.loc[0, "proyectado"] == 1
    assert weekly.loc[0, "proyectado"] == 7
    assert manifest["run_id"] == run_id


def test_operational_run_id_changes_with_input_or_model():
    cfg = _cfg()
    original = _operational_run_id("S36", {"input": "a", "model": "a"})
    assert original == _operational_run_id("S36", {"input": "a", "model": "a"})
    assert original != _operational_run_id("S36", {"input": "b", "model": "a"})
    assert original != _operational_run_id("S36", {"input": "a", "model": "b"})


def test_scoring_origin_must_be_after_training_cutoff():
    frame = pd.DataFrame({"fecha_origen": pd.to_datetime(["2026-08-09"])})
    with np.testing.assert_raises_regex(ValueError, "no es posterior"):
        validate_scoring_cutoff(frame, {"training_cutoff": "2026-08-09"})


def test_model_artifacts_are_content_addressed(tmp_path):
    bundle, manifest = train_operational_models(_training_frame(), _cfg())
    registry = tmp_path / "models"
    pointer = registry / "active.json"
    artifact, frozen_manifest = save_operational_models(bundle, manifest, registry, pointer)
    loaded, loaded_manifest = load_operational_models(registry, pointer)

    assert artifact.parent.name == loaded_manifest["version"]
    assert frozen_manifest.parent == artifact.parent
    assert loaded["features"] == bundle["features"]


def test_operational_comparison_preserves_observed_cardinality():
    weekly = pd.DataFrame({
        "modelo": ["RF", "M3", "RF", "M3"],
        "finca": ["ALMER"] * 4, "bloque": ["1", "1", "2", "2"],
        "fecha_origen": pd.to_datetime(["2026-09-01"] * 4),
        "semana_proyeccion": [202637] * 4, "proyectado": [10, 11, 20, 21],
    })
    comparison = operational_model_comparison(weekly)
    assert len(comparison) == 2
    assert set(comparison.columns) >= {"RF", "M3"}
