from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical import load_config
from src.models.m3 import fit_m3, load_traditional_intervals, simulate


ROOT = Path(__file__).resolve().parents[1]


def test_traditional_intervals_only_have_daily_steps():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    intervals = load_traditional_intervals(ROOT / "data" / "raw", cfg)
    assert len(intervals) > 0
    assert intervals["delta_dias"].eq(1).all()
    assert intervals["estado_origen"].isin(["RC", "SS", "AP"]).all()


def test_m3_columns_are_stochastic_and_nonnegative():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    intervals = load_traditional_intervals(ROOT / "data" / "raw", cfg)
    for farm in cfg["project"]["target_farms"]:
        for period in [p["name"] for p in cfg["m3"]["periods"]]:
            matrix = fit_m3(intervals, farm, period, pd.Timestamp("2026-08-09"))
            assert np.all(matrix.Q >= 0)
            assert np.all(matrix.r >= 0)
            assert np.all(matrix.p >= 0)
            assert np.allclose(matrix.Q.sum(axis=0) + matrix.r + matrix.p, 1.0)


def test_m3_simulation_returns_seven_daily_cuts():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    intervals = load_traditional_intervals(ROOT / "data" / "raw", cfg)
    matrix = fit_m3(intervals, "LA PRADERA", "JULIO", pd.Timestamp("2026-08-09"))
    result = simulate(matrix, np.array([100.0, 200.0, 300.0]), 7, 0.10)
    assert len(result) == 7
    assert result["PC_dia_muestra"].ge(0).all()
