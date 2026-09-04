"""Fase 4: screening de lags y challengers M3 + podas."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import load_config, load_pruning
from evaluation.metrics import metrics
from evaluation.split import validation_start
from models.m3 import _period_for_date, fit_m3, simulate
from models.pruning import (adjust_matrix, build_pruning_features,
                            pruning_signal)


def _split_origins(windows: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = (windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)
               .agg(ventana_evaluable=("ventana_evaluable", "first"))
               .sort_values("fecha_origen"))
    cutoff = validation_start(origins["fecha_origen"], cfg)
    train = origins[pd.to_datetime(origins.fecha_origen).lt(cutoff)]
    validation = origins[pd.to_datetime(origins.fecha_origen).ge(cutoff)]
    validation = validation[validation["ventana_evaluable"]].copy()
    return train, validation


def _predict(row, matrix, alpha):
    x0 = np.array([row["conteo_RC_t0"], row["conteo_SS_t0"], row["conteo_AP_t0"]], float)
    days = (pd.Timestamp(row["fecha_objetivo"]) -
            pd.Timestamp(row["fecha_origen"])).days
    result = simulate(matrix, x0, days, alpha).iloc[-1]
    return result["PC_dia_muestra"] * row["factor_extrapolacion_t0"]


def _score(validation, feature_map, matrices, coefficient, mode, cfg):
    pred, real = [], []
    for _, row in validation.iterrows():
        key = (row["finca"], row["bloque"], row["fecha_origen"])
        signal = float(feature_map[key])
        matrix = matrices[key]
        alpha = float(cfg["m3"]["baseline_ingress"])
        if mode in ("ingreso", "hibrido"):
            alpha = min(1.0, alpha + coefficient * signal)
        matrix = adjust_matrix(matrix, signal, coefficient,
                               mode in ("transicion", "hibrido"))
        pred.append(_predict(row, matrix, alpha))
        real.append(row["corte_real_dia"])
    return metrics(pd.Series(real), pd.Series(pred))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    raw = root / cfg["paths"]["raw"]
    output = root / cfg["paths"]["outputs"]
    datasets, evaluation, models = [output / x for x in ("datasets", "evaluation", "models")]
    for directory in (datasets, evaluation, models):
        directory.mkdir(parents=True, exist_ok=True)
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    traditional = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    pruning = load_pruning(raw, cfg["farm_aliases"], cfg["project"]["target_farms"])
    train_origins, val_origins = _split_origins(windows, cfg)
    all_origins = pd.concat([train_origins, val_origins], ignore_index=True)
    features = build_pruning_features(pruning, all_origins, cfg)
    features.to_parquet(datasets / "poda_features.parquet", index=False)
    signal = pruning_signal(features, cfg)
    feature_map = {(r.finca, r.bloque, r.fecha_origen): s
                   for r, s in zip(features.itertuples(), signal)}

    # Screening simple por bloque: solo se usa para describir temporalidad.
    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    screening = []
    for lag in cfg["pruning"]["priority_lags_days"]:
        left = fact[["finca", "bloque", "fecha", "corte_comercial_real"]].copy()
        left["poda_fecha"] = left["fecha"] - pd.Timedelta(days=lag)
        right = pruning[["finca", "bloque", "fecha", "poda_total"]].rename(columns={"fecha": "poda_fecha"})
        joined = left.merge(right, on=["finca", "bloque", "poda_fecha"], how="left").fillna({"poda_total": 0})
        screening.append({"lag_dias": lag, "pearson": joined["poda_total"].corr(joined["corte_comercial_real"]),
                          "spearman": joined["poda_total"].corr(joined["corte_comercial_real"], method="spearman"),
                          "n": len(joined)})
    pd.DataFrame(screening).to_csv(evaluation / "poda_lag_screening.csv", index=False)

    matrices = {}
    for _, origin in all_origins.iterrows():
        key = (origin["finca"], origin["bloque"], origin["fecha_origen"])
        period = _period_for_date(pd.Timestamp(origin["fecha_origen"]), cfg["m3"]["periods"])
        matrices[key] = fit_m3(traditional, origin["finca"], period, pd.Timestamp(origin["fecha_origen"]))
    val_windows = windows.merge(val_origins[["finca", "bloque", "fecha_origen"]],
                                on=["finca", "bloque", "fecha_origen"], how="inner")
    grid = cfg["pruning"]["coefficient_grid"]
    rows = []
    for mode in ("base", "ingreso", "transicion", "hibrido"):
        for coefficient in grid:
            score = _score(val_windows, feature_map, matrices, float(coefficient), mode, cfg)
            rows.append({"experiment_id": f"M3_PODA_{mode.upper()}", "split": "VALIDATION",
                         "causal": True, "coefficient": coefficient, **score})
    result = pd.DataFrame(rows)
    result.to_csv(evaluation / "metrics_fase4_podas.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
