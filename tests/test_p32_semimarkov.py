from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical import load_config
from src.models.m3 import fit_m3
from src.models.p32 import build_p32_intervals, load_p32_observations
from src.models.semimarkov import (SemiMarkov, competitive_hazards,
                                   estimate_age_distributions)


ROOT = Path(__file__).resolve().parents[1]


def _data():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    obs = load_p32_observations(ROOT / "data" / "raw" / cfg["sources"]["fenologia_p32"], cfg)
    return cfg, obs, build_p32_intervals(obs)


def test_p32_keeps_pending_codes_out_of_valid_intervals():
    _, obs, intervals = _data()
    assert len(obs) == 1297
    assert (obs["estado_raw"] == "SP").sum() == 160
    pending = intervals[intervals["motivo_exclusion"].astype(str).str.contains("PENDIENTE")]
    assert len(pending) > 0
    assert intervals.loc[intervals["valido"], "delta_dias"].eq(1).all()


def test_age_distributions_and_hazards_normalize():
    cfg, _, intervals = _data()
    valid = intervals[intervals["valido"]]
    age = estimate_age_distributions(valid)
    m3 = fit_m3(pd.read_parquet(ROOT / "outputs/datasets/transition_intervals_tradicional.parquet"),
                "LA PRADERA", "JULIO", pd.Timestamp("2026-08-09"))
    hazards = competitive_hazards(valid, m3, max_bin=3)
    for values in age.values():
        assert np.isclose(values.sum(), 1)
    for values in hazards.values():
        assert np.isclose(sum(values.values()), 1)
        assert all(0 <= value <= 1 for value in values.values())
    assert len(cfg["semimarkov"]["age_bins"]) == 4


def test_semimarkov_initial_x0_is_distributed_over_age():
    _, _, intervals = _data()
    valid = intervals[intervals["valido"]]
    age = estimate_age_distributions(valid)
    m3 = fit_m3(pd.read_parquet(ROOT / "outputs/datasets/transition_intervals_tradicional.parquet"),
                "LA PRADERA", "JULIO", pd.Timestamp("2026-08-09"))
    model = SemiMarkov(competitive_hazards(valid, m3), age)
    initial = model.initial_mass(np.array([100.0, 200.0, 300.0]))
    assert np.isclose(sum(initial.values()), 600)
    assert sum(value > 0 for (state, _), value in initial.items() if state == "AP") > 1


def test_semimarkov_mass_does_not_increase_without_rc_ingress():
    _, _, intervals = _data()
    valid = intervals[intervals["valido"]]
    m3 = fit_m3(pd.read_parquet(ROOT / "outputs/datasets/transition_intervals_tradicional.parquet"),
                "LA PRADERA", "JULIO", pd.Timestamp("2026-08-09"))
    model = SemiMarkov(competitive_hazards(valid, m3), estimate_age_distributions(valid))
    result = model.simulate(np.array([100.0, 200.0, 300.0]), 7, 0)
    assert result["masa_activa"].diff().dropna().le(1e-9).all()
