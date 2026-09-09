import pandas as pd

from src.dashboard_validation import (complete_windows, load_validation_predictions, model_scores,
                                      operational_summary, operational_weekly, operational_weekly_selected,
                                      weekly_status)
from src.evaluacion_hyperparametros import parallel_plan


def _window(model, origin, status="VALIDA", days=7):
    return pd.DataFrame({
        "modelo": model, "split": "FIXED_VALIDATION", "finca": "A", "bloque": "1",
        "fecha_origen": pd.Timestamp(origin), "semana_proyeccion": 202601,
        "fecha_objetivo": pd.date_range("2026-01-05", periods=days),
        "horizonte_dia": range(1, days + 1), "estado_ventana": status,
        "real": [10.0] * days, "proyectado_modelo": [9.0] * days,
    })


def test_complete_windows_excludes_partial_and_incomplete_weeks():
    daily = pd.concat([_window("M1", "2026-01-01"),
                       _window("M1", "2026-01-02", status="PARCIAL"),
                       _window("M1", "2026-01-03", days=6)], ignore_index=True)
    weekly, complete_daily = complete_windows(daily, 7)
    assert len(weekly) == 1
    assert len(complete_daily) == 7
    assert weekly.iloc[0].fecha_origen == pd.Timestamp("2026-01-01")


def test_complete_windows_requires_complete_predictions_and_consecutive_dates():
    frame = _window("M1", "2026-01-01")
    frame.loc[frame.index[-1], "proyectado_modelo"] = None
    weekly, daily = complete_windows(frame, 7)
    assert weekly.empty
    assert daily.empty


def test_weekly_keys_keep_distinct_origins():
    daily = pd.concat([_window("M1", "2026-01-01"), _window("M1", "2026-01-02")],
                      ignore_index=True)
    weekly, _ = complete_windows(daily, 7)
    assert len(weekly) == 2


def test_model_scores_are_calculated_per_model():
    first = _window("M1", "2026-01-01")
    second = _window("M2", "2026-01-01")
    second["proyectado_modelo"] = 5.0
    weekly, _ = complete_windows(pd.concat([first, second], ignore_index=True), 7)
    scores = model_scores(weekly).set_index("modelo")
    assert scores.loc["M1", "wape"] == 0.1
    assert scores.loc["M2", "wape"] == 0.5


def test_weekly_bias_is_signed_and_not_one_minus_wape():
    first = _window("M1", "2026-01-01")
    first["proyectado_modelo"] = 12.0
    weekly, _ = complete_windows(first, 7)
    score = model_scores(weekly).iloc[0]
    assert score["bias_pct"] == 0.2
    assert score["acierto_global"] == 0.8


def test_predictions_without_window_status_are_visible(tmp_path):
    path = tmp_path / "predicciones.csv"
    frame = _window("M1", "2026-01-01").drop(columns="estado_ventana")
    frame = frame.rename(columns={"proyectado_modelo": "proyectado"})
    frame.to_csv(path, index=False)
    daily = load_validation_predictions(tmp_path, {"M1": "predicciones.csv"}, "FIXED_VALIDATION")
    weekly, complete_daily = complete_windows(daily, 7)
    assert len(weekly) == 1
    assert len(complete_daily) == 7
    assert daily["estado_ventana"].eq("VALIDA").all()


def test_parallel_plan_uses_one_inner_thread_per_forest(monkeypatch):
    monkeypatch.setattr("src.evaluacion_hyperparametros.os.cpu_count", lambda: 8)
    monkeypatch.setattr("src.evaluacion_hyperparametros.available_memory_gb", lambda: 8.0)
    plan = parallel_plan({"n_jobs": "auto", "reserve_cpus": 1,
                          "reserve_memory_gb": 1.0, "estimated_memory_per_worker_gb": 1.0})
    assert plan["workers"] == 7
    assert plan["model_n_jobs"] == 1


def test_weekly_status_uses_company_operational_ranges():
    assert weekly_status(0.93) == "ACIERTO"
    assert weekly_status(1.07) == "ACIERTO"
    assert weekly_status(0.90) == "CERCA"
    assert weekly_status(1.10) == "CERCA"
    assert weekly_status(0.89) == "NO ACIERTO"
    assert weekly_status(1.11) == "NO ACIERTO"


def test_operational_weekly_and_summary_expose_ratio_and_hit_counts():
    weekly = pd.DataFrame({
        "modelo": ["M", "M", "M"], "split": ["S"] * 3,
        "finca": ["A"] * 3, "bloque": ["1"] * 3,
        "fecha_origen": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15"]),
        "semana_proyeccion": [1, 2, 3], "real": [100, 100, 100],
        "proyectado_modelo": [100, 92, 80],
    })
    detail = operational_weekly(weekly)
    assert detail["indicador"].tolist() == ["ACIERTO", "CERCA", "NO ACIERTO"]
    summary = operational_summary(weekly).iloc[0]
    assert summary.aciertos == 1
    assert summary.cerca == 1
    assert summary.no_aciertos == 1


def test_operational_selection_chooses_origin_closest_to_monday_and_keeps_date():
    first = _window("M1", "2026-01-10")
    second = _window("M1", "2026-01-15")
    first["semana_proyeccion"] = 202602
    second["semana_proyeccion"] = 202602
    first["fecha_objetivo"] = pd.date_range("2026-01-12", periods=7)
    second["fecha_objetivo"] = pd.date_range("2026-01-12", periods=7)
    selected = operational_weekly_selected(pd.concat([first, second], ignore_index=True))
    assert len(selected) == 1
    assert selected.iloc[0]["fecha_origen_seleccionada"] == "2026-01-10"
    assert selected.iloc[0]["distancia_inicio_dias"] == 2


def test_operational_selection_breaks_equal_distance_with_earlier_origin():
    first = _window("M1", "2026-01-10")
    second = _window("M1", "2026-01-14")
    first["semana_proyeccion"] = second["semana_proyeccion"] = 202602
    first["fecha_objetivo"] = second["fecha_objetivo"] = pd.date_range("2026-01-12", periods=7)
    selected = operational_weekly_selected(pd.concat([first, second], ignore_index=True))
    assert selected.iloc[0]["fecha_origen_seleccionada"] == "2026-01-10"


def test_operational_selection_is_global_by_farm_and_week_by_default():
    block_one = _window("M1", "2026-01-10")
    block_one["bloque"] = "1"
    block_two = _window("M1", "2026-01-15")
    block_two["bloque"] = "2"
    for frame in (block_one, block_two):
        frame["semana_proyeccion"] = 202602
        frame["fecha_objetivo"] = pd.date_range("2026-01-12", periods=7)
    selected = operational_weekly_selected(pd.concat([block_one, block_two], ignore_index=True))
    assert len(selected) == 1
    assert selected.iloc[0]["fecha_origen_seleccionada"] == "2026-01-10, 2026-01-15"
    assert selected.iloc[0]["real"] == 140


def test_operational_selection_can_segment_by_block():
    first = _window("M1", "2026-01-10")
    second = _window("M1", "2026-01-15")
    first["bloque"] = "1"
    second["bloque"] = "2"
    first["semana_proyeccion"] = second["semana_proyeccion"] = 202602
    first["fecha_objetivo"] = second["fecha_objetivo"] = pd.date_range("2026-01-12", periods=7)
    selected = operational_weekly_selected(pd.concat([first, second], ignore_index=True), by_block=True)
    assert len(selected) == 2
    global_view = operational_weekly_selected(pd.concat([first, second], ignore_index=True))
    assert "bloque" not in global_view.columns
