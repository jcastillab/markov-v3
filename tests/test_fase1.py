from pathlib import Path

import pandas as pd

from src.canonical import (active_beds_by_date, build_forecast_windows,
                           canonical_block, canonical_farm, load_config,
                           parse_iso_week)


ROOT = Path(__file__).resolve().parents[1]


def test_farm_and_block_normalization():
    aliases = {"Pradera": "LA PRADERA", "ALMER": "ALMER"}
    assert canonical_farm("  pradera  ", aliases) == "LA PRADERA"
    assert canonical_block(12.0) == "12"
    assert canonical_block(" P11 ") == "P11"


def test_sample_week_parsing():
    assert parse_iso_week("S17") == 202617
    assert parse_iso_week("Resultados") is None


def test_active_beds_respects_sowing_and_eradicating_dates():
    beds = pd.DataFrame([
        {"finca": "ALMER", "bloque": "1", "cama_id": "1",
         "fecha_siembra": pd.Timestamp("2026-01-02"),
         "fecha_erradicacion": pd.Timestamp("2026-01-03"),
         "plantas": 10, "area_sembrada": 2.0},
    ])
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"]))
    result = active_beds_by_date(beds, dates).set_index("fecha")
    assert result.loc[pd.Timestamp("2026-01-01"), "camas_activas"] == 0
    assert result.loc[pd.Timestamp("2026-01-02"), "camas_activas"] == 1
    assert result.loc[pd.Timestamp("2026-01-04"), "camas_activas"] == 0


def test_forecast_windows_are_seven_consecutive_days_and_future():
    dates = pd.date_range("2026-04-20", "2026-04-27")
    fact = pd.DataFrame({
        "finca": "ALMER", "bloque": "1", "fecha": dates,
        "semana_iso": [202617] * 7 + [202618], "semana_objetivo": [202618] * 7 + [202619],
        "corte_comercial_real": range(8), "conteo_RC": [1] + [None] * 7,
        "conteo_SS": [2] + [None] * 7, "conteo_AP": [3] + [None] * 7,
        "conteo_CO": [0] + [None] * 7, "conteo_total": [6] + [None] * 7,
        "es_fecha_conteo": [True] + [False] * 7,
        "fecha_conteo_origen": [dates[0]] * 8, "dias_desde_conteo": range(8),
        "conteo_RC_origen": [1] * 8, "conteo_SS_origen": [2] * 8,
        "conteo_AP_origen": [3] * 8, "conteo_CO_origen": [0] * 8,
        "conteo_total_origen": [6] * 8, "camas_activas": [10] * 8,
        "camas_muestreadas": [5] * 8, "factor_extrapolacion": [2] * 8,
    })
    windows = build_forecast_windows(fact)
    assert len(windows) == 7
    assert windows["fecha_objetivo"].min() == pd.Timestamp("2026-04-27")
    assert windows["fecha_objetivo"].diff().dropna().eq(pd.Timedelta(days=1)).all()
