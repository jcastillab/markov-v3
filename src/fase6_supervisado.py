"""Fase 6: dataset supervisado, GLM NB, RF y residual sobre M3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

from canonical import load_config
from evaluation.metrics import metrics
from models.m3 import load_traditional_intervals
from models.supervised import (NegativeBinomialGLM, build_supervised_dataset,
                               feature_groups)


def main():
    root = Path(__file__).resolve().parents[1]; cfg = load_config(root / "config" / "pipeline.yaml")
    out = root / cfg["paths"]["outputs"]
    datasets, evaluation, models = [out / x for x in ("datasets", "evaluation", "models")]
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    intervals = load_traditional_intervals(root / cfg["paths"]["raw"], cfg)
    pruning = pd.read_parquet(datasets / "poda_features.parquet")
    climate = pd.read_parquet(datasets / "clima_features.parquet")
    frame = build_supervised_dataset(windows, fact, intervals, cfg, pruning, climate)
    frame.to_parquet(datasets / "dataset_supervisado_diario.parquet", index=False)
    origins = (windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)
               .size().drop(columns="size").sort_values("fecha_origen"))
    n_train = max(cfg["evaluation"]["min_train_windows"], int(len(origins) * .60))
    train_keys = set(map(tuple, origins.iloc[:n_train].itertuples(index=False, name=None)))
    key = list(zip(frame.finca, frame.bloque, frame.fecha_origen))
    train = np.array([x in train_keys for x in key]); valid = ~train
    groups = feature_groups(frame)
    numeric = frame.select_dtypes(include=[np.number]).columns
    x_base = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    rows = []
    for name, cols in groups.items():
        cols = list(dict.fromkeys(c for c in cols if c in x_base.columns))
        x = x_base[cols]
        glm = NegativeBinomialGLM(cfg["supervised"]["nb_alpha"], cfg["supervised"]["nb_max_iter"])
        glm.fit(x[train], frame.target.to_numpy()[train])
        pred = glm.predict(x[valid])
        rows.append({"experiment_id": f"GLM_NB_{name}", "split": "VALIDATION", "causal": True,
                     "n": int(valid.sum()), **metrics(frame.target[valid], pd.Series(pred))})
        from sklearn.ensemble import RandomForestRegressor
        rf_cfg = cfg["random_forest"]
        rf = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"], max_depth=rf_cfg["max_depth_grid"][0],
            min_samples_leaf=rf_cfg["min_samples_leaf_grid"][0], min_samples_split=rf_cfg["min_samples_split_grid"][0],
            max_features=rf_cfg["max_features"][0], random_state=rf_cfg["random_state"], n_jobs=-1)
        rf.fit(x[train], frame.target[train]); pred = rf.predict(x[valid])
        pd.DataFrame({"real": frame.target[valid].to_numpy(), "pred": pred}).to_csv(
            evaluation / f"predictions_{name.lower()}.csv", index=False)
        rows.append({"experiment_id": f"RF_DIARIO_POOLED_{name}", "split": "VALIDATION", "causal": True,
                     "n": int(valid.sum()), **metrics(frame.target[valid], pd.Series(pred))})
        importance = permutation_importance(rf, x[valid], frame.target[valid], n_repeats=5,
                                            random_state=rf_cfg["random_state"], n_jobs=-1)
        pd.DataFrame({"experiment_id": f"RF_DIARIO_POOLED_{name}", "feature": cols,
                      "importance_mean": importance.importances_mean}).to_csv(
                          models / f"importance_{name.lower()}.csv", index=False)
    # Residual: la prediccion mecanistica queda como nivel y RF aprende solo el ajuste.
    cols = list(dict.fromkeys(c for c in groups["FENO"] if c in x_base.columns))
    x = x_base[cols]
    rf_cfg = cfg["random_forest"]
    residual_rf = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth_grid"][0], min_samples_leaf=rf_cfg["min_samples_leaf_grid"][0],
        min_samples_split=rf_cfg["min_samples_split_grid"][0], max_features=rf_cfg["max_features"][0],
        random_state=rf_cfg["random_state"], n_jobs=-1)
    residual = frame.target.to_numpy() - frame.M3_pred_bloque.to_numpy()
    residual_rf.fit(x[train], residual[train])
    pred = np.maximum(0, frame.M3_pred_bloque.to_numpy()[valid] + residual_rf.predict(x[valid]))
    pd.DataFrame({"real": frame.target[valid].to_numpy(), "pred": pred}).to_csv(
        evaluation / "predictions_rf_residual_m3_feno.csv", index=False)
    rows.append({"experiment_id": "RF_RESIDUAL_M3_FENO", "split": "VALIDATION", "causal": True,
                 "n": int(valid.sum()), **metrics(frame.target[valid], pd.Series(pred))})
    # Ablacion de horizonte: siete RF independientes y suma semanal equivalente.
    h_pred = []
    for horizon in range(1, 8):
        mask_train = train & frame.horizonte_dia.eq(horizon).to_numpy()
        mask_valid = valid & frame.horizonte_dia.eq(horizon).to_numpy()
        model = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth_grid"][0], min_samples_leaf=rf_cfg["min_samples_leaf_grid"][0],
            min_samples_split=rf_cfg["min_samples_split_grid"][0], max_features=rf_cfg["max_features"][0],
            random_state=rf_cfg["random_state"], n_jobs=-1)
        model.fit(x[mask_train], frame.target[mask_train])
        h_pred.append(pd.DataFrame({"finca": frame.loc[mask_valid, "finca"].to_numpy(),
            "bloque": frame.loc[mask_valid, "bloque"].to_numpy(),
            "fecha_origen": frame.loc[mask_valid, "fecha_origen"].to_numpy(),
            "real": frame.target[mask_valid].to_numpy(), "pred": model.predict(x[mask_valid])}))
    h_frame = pd.concat(h_pred, ignore_index=True)
    h_frame.to_csv(evaluation / "predictions_rf_h1_h7_feno.csv", index=False)
    rows.append({"experiment_id": "RF_H1_H7_FENO", "split": "VALIDATION", "causal": True,
                 "n": len(h_frame), **metrics(h_frame.real, h_frame.pred)})
    weekly = h_frame.groupby(["finca", "bloque", "fecha_origen"], as_index=False).agg(
        real=("real", "sum"), pred=("pred", "sum"))
    rows.append({"experiment_id": "RF_SEMANAL_FENO", "split": "VALIDATION", "causal": True,
                 "n": len(weekly), **metrics(weekly.real, weekly.pred)})
    result = pd.DataFrame(rows); result.to_csv(evaluation / "metrics_fase6_supervisado.csv", index=False)
    (models / "supervised_manifest.json").write_text(pd.Series({"phase": 6, "causal": True,
        "n_rows": len(frame), "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
        "models": result.experiment_id.tolist()}).to_json(indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
