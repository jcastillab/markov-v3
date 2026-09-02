"""Ablacion causal: RF H1-H7 sin usar las predicciones auxiliares de M3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from canonical import load_config
from evaluation.metrics import metrics
from models.supervised import build_supervised_dataset, feature_groups
from diagnostico_modelos import split_frame


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    out = root / cfg["paths"]["outputs"]
    datasets, evaluation = out / "datasets", out / "evaluation"
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    intervals = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    pruning = pd.read_parquet(datasets / "poda_features.parquet")
    climate = pd.read_parquet(datasets / "clima_features.parquet")
    frame = build_supervised_dataset(windows, fact, intervals, cfg, pruning, climate)
    numeric = frame.select_dtypes(include=[np.number]).columns
    cols = [c for c in feature_groups(frame)["FENO"]
            if c in numeric and not c.startswith("M3_") and c != "target"]
    if any(c.startswith("M3_") for c in cols):
        raise AssertionError("La ablacion contiene una feature M3")
    train, valid = split_frame(frame, windows, cfg)
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = frame.target.to_numpy(float)
    rf_cfg = cfg["random_forest"]
    prediction_rows, metric_rows = [], []
    for horizon in range(1, 8):
        train_h = train & frame.horizonte_dia.eq(horizon).to_numpy()
        valid_h = valid & frame.horizonte_dia.eq(horizon).to_numpy()
        model = RandomForestRegressor(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth_grid"][0],
            min_samples_leaf=rf_cfg["min_samples_leaf_grid"][0],
            min_samples_split=rf_cfg["min_samples_split_grid"][0],
            max_features=rf_cfg["max_features"][0],
            random_state=rf_cfg["random_state"], n_jobs=-1)
        model.fit(x[train_h], y[train_h])
        pred = model.predict(x[valid_h])
        trace = frame.loc[valid_h, ["finca", "bloque", "fecha_origen", "fecha_objetivo",
                                    "semana_proyeccion", "horizonte_dia", "target"]].copy()
        trace = trace.rename(columns={"target": "real"}).assign(pred=pred)
        prediction_rows.append(trace)
        metric_rows.append({"experiment_id": "RF_H1_H7_FENO_SIN_M3", "horizonte": horizon,
                            "split": "VALIDATION", "causal": True, "n": int(valid_h.sum()),
                            **metrics(pd.Series(y[valid_h]), pd.Series(pred))})
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(evaluation / "predictions_rf_h1_h7_feno_sin_m3.csv", index=False)
    weekly = predictions.groupby(["finca", "bloque", "fecha_origen"], as_index=False).agg(
        real=("real", "sum"), pred=("pred", "sum"))
    metric_rows.append({"experiment_id": "RF_H1_H7_FENO_SIN_M3", "horizonte": "TODOS",
                        "split": "VALIDATION", "causal": True, "n": len(predictions),
                        **metrics(predictions.real, predictions.pred)})
    metric_rows.append({"experiment_id": "RF_H1_H7_FENO_SIN_M3", "horizonte": "SEMANAL",
                        "split": "VALIDATION", "causal": True, "n": len(weekly),
                        **metrics(weekly.real, weekly.pred)})
    pd.DataFrame(metric_rows).to_csv(evaluation / "metrics_rf_ablation_m3.csv", index=False)
    (evaluation / "rf_ablation_m3_manifest.json").write_text(
        __import__("json").dumps({"model": "RF_H1_H7_FENO_SIN_M3", "features": cols,
                                  "n_features": len(cols), "causal": True}, indent=2),
        encoding="utf-8")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"Features sin M3: {len(cols)}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
