from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical import load_config
from src.models.climate import build_daily_climate, load_hourly_climate


ROOT = Path(__file__).resolve().parents[1]


def test_climate_daily_contract_and_vpd():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    hourly, qa = load_hourly_climate(ROOT / "data" / "raw",
                                     ["2025.xlsx", "2026.xlsx"], 746)
    daily = build_daily_climate(hourly.head(48), cfg)
    assert len(daily) > 0
    assert qa["duplicados_eliminados"] >= 0
    assert daily["rain_mm"].ge(0).all()
    assert daily["solar_MJ_m2_d"].ge(0).all()
    assert np.isfinite(daily["vpd_mean_kPa"]).any()
    assert daily["DLI_PAR_est_mol_m2_d"].ge(0).all()


def test_climate_vpd_formula():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    hourly = pd.DataFrame({"fecha_hora": pd.to_datetime(["2026-01-01 12:00"]),
        "IdEstacion": [746], "Lluvia (mm)": [0], "Temperatura (C)": [20],
        "TemperaturaMaxima (C)": [20], "TemperaturaMinima (C)": [20],
        "Humedad(%)": [50], "Vel_Viento (km/h)": [1],
        "Evapotranspiracion (mm/h)": [0], "Punto_Rocio (C)": [10],
        "UV (Index)": [1], "Radiacion_Solar (w/m2)": [100],
        "Presion Atmosferica (hPa)": [750]})
    daily = build_daily_climate(hourly, cfg)
    expected = 0.6108 * np.exp(17.27 * 20 / (20 + 237.3)) * .5
    assert np.isclose(daily.loc[0, "vpd_mean_kPa"], expected)
