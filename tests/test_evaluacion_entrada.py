import numpy as np
import pandas as pd

from src.canonical import build_forecast_windows
from src.evaluacion_entrada import _status, _weekly


def test_entry_status_uses_ratio_ranges():
    assert _status(1.0) == "ACIERTO"
    assert _status(0.92) == "CERCA"
    assert _status(1.11) == "NO ACIERTO"
    assert _status(np.nan) == "SIN REAL"


def test_weekly_entry_sums_days_before_calculating_ratio():
    daily = pd.DataFrame({
        "modelo": ["M"] * 7, "finca": ["A"] * 7, "bloque": ["1"] * 7,
        "fecha_origen": pd.Timestamp("2026-05-01"),
        "fecha_objetivo": pd.date_range("2026-05-04", periods=7),
        "semana_proyeccion": 202619, "real": [10] * 7,
        "proyectado": [11, 9, 10, 10, 10, 10, 10],
    })
    weekly = _weekly(daily)
    row = weekly.iloc[0]
    assert row.real == 70
    assert row.proyectado == 70
    assert row.razon_proyectado_real == 1
    assert row.indicador == "ACIERTO"


def test_entry_window_uses_last_count_and_next_monday_to_sunday():
    dates = pd.date_range("2026-05-04", "2026-05-17")
    fact = pd.DataFrame({
        "finca": "ALMER", "bloque": "1", "fecha": dates,
        "semana_iso": [202618] * 7 + [202619] * 7,
        "corte_comercial_real": 1.0, "conteo_RC": [None] * len(dates),
        "conteo_SS": [None] * len(dates), "conteo_AP": [None] * len(dates),
        "conteo_CO": [None] * len(dates), "conteo_total": [None] * len(dates),
        "es_fecha_conteo": [False] * len(dates), "factor_extrapolacion": 1.0,
        "camas_muestreadas": 1.0, "camas_activas": 1.0,
        "fecha_conteo_origen": pd.Timestamp("2026-05-06"), "dias_desde_conteo": 0,
        "conteo_RC_origen": 4, "conteo_SS_origen": 5, "conteo_AP_origen": 6,
        "conteo_CO_origen": 0, "conteo_total_origen": 15,
    })
    fact.loc[fact.fecha.eq(pd.Timestamp("2026-05-05")), ["conteo_RC", "conteo_SS", "conteo_AP", "conteo_total", "es_fecha_conteo"]] = [1, 2, 3, 6, True]
    fact.loc[fact.fecha.eq(pd.Timestamp("2026-05-06")), ["conteo_RC", "conteo_SS", "conteo_AP", "conteo_total", "es_fecha_conteo"]] = [4, 5, 6, 15, True]
    windows = build_forecast_windows(fact)
    assert len(windows) == 7
    assert windows.fecha_origen.iloc[0] == pd.Timestamp("2026-05-06")
    assert windows.fecha_objetivo.min() == pd.Timestamp("2026-05-11")
    assert windows.fecha_objetivo.max() == pd.Timestamp("2026-05-17")
