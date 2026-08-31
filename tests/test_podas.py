from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical import load_config, load_pruning
from src.models.m3 import fit_m3
from src.models.pruning import adjust_matrix, build_pruning_features


ROOT = Path(__file__).resolve().parents[1]


def test_pruning_contract_excludes_estimated_and_normalizes_destinations():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    pruning = load_pruning(ROOT / "data" / "raw", cfg["farm_aliases"], cfg["project"]["target_farms"])
    assert len(pruning) > 0
    assert pruning["poda_total"].eq(pruning["poda_corte"] + pruning["poda_alineamiento"]).all()
    assert pruning[["poda_corte", "poda_alineamiento", "poda_total"]].ge(0).all().all()


def test_pruning_features_only_use_history_at_origin():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    pruning = pd.DataFrame([{"finca": "ALMER", "bloque": "1", "fecha": pd.Timestamp("2026-01-01"),
                             "poda_corte": 10., "poda_alineamiento": 0., "poda_total": 10.},
                            {"finca": "ALMER", "bloque": "1", "fecha": pd.Timestamp("2026-03-01"),
                             "poda_corte": 99., "poda_alineamiento": 0., "poda_total": 99.}])
    origins = pd.DataFrame([{"finca": "ALMER", "bloque": "1", "fecha_origen": pd.Timestamp("2026-02-15")}])
    features = build_pruning_features(pruning, origins, cfg)
    assert features.loc[0, "poda_total_acumulada_84d"] == 10


def test_pruning_transition_adjustment_preserves_stochastic_columns():
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    intervals = pd.read_parquet(ROOT / "outputs" / "datasets" / "transition_intervals_tradicional.parquet")
    matrix = fit_m3(intervals, "LA PRADERA", "JULIO", pd.Timestamp("2026-08-09"))
    adjusted = adjust_matrix(matrix, 2.0, 0.3, True)
    assert np.allclose(adjusted.Q.sum(axis=0) + adjusted.r + adjusted.p, 1.0)
    assert (adjusted.Q >= 0).all() and (adjusted.r >= 0).all() and (adjusted.p >= 0).all()
