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


def test_m3_matches_power_bi_exposure_method_for_pradera():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    intervals = load_traditional_intervals(ROOT / "data" / "raw", cfg)
    expected = {
        "ABRIL": (0.8183143197, 0.7630337504, 0.5221693122, 0.4778306878),
        "JULIO": (0.6295634921, 0.8075231481, 0.6787918871, 0.3212081129),
    }
    for period, (rc_stay, ss_stay, ap_stay, ap_cut) in expected.items():
        matrix = fit_m3(intervals, "LA PRADERA", period)
        assert np.isclose(matrix.Q[0, 0], rc_stay)
        assert np.isclose(matrix.Q[1, 1], ss_stay)
        assert np.isclose(matrix.Q[2, 2], ap_stay)
        assert np.isclose(matrix.r[2], ap_cut)
        assert np.allclose(matrix.Q.sum(axis=0) + matrix.r + matrix.p, 1.0)
