"""Busqueda causal de modelos de conteo y hiperparametros.

La validacion conserva los mismos origenes 60/40 de Fase 6. Se reportan
metricas diarias y semanales porque un R2 semanal alto no implica buen ajuste
por bloque y dia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from canonical import load_config
from evaluation.metrics import metrics
from models.supervised import build_supervised_dataset, feature_groups


def split_frame(frame, windows, cfg):
    origins = (windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)
               .size().drop(columns="size").sort_values("fecha_origen"))
    n_train = max(cfg["evaluation"]["min_train_windows"], int(len(origins) * .60))
    keys = set(map(tuple, origins.iloc[:n_train].itertuples(index=False, name=None)))
    valid = np.array([tuple(x) in keys for x in zip(frame.finca, frame.bloque, frame.fecha_origen)])
    return ~valid, valid


def score(y, pred, frame):
    daily = metrics(pd.Series(y), pd.Series(pred))
    detail = pd.DataFrame({"finca": frame.finca.to_numpy(), "bloque": frame.bloque.to_numpy(),
                           "fecha_origen": frame.fecha_origen.to_numpy(), "real": y, "pred": pred})
    weekly = detail.groupby(["finca", "bloque", "fecha_origen"], as_index=False).agg(
        real=("real", "sum"), pred=("pred", "sum"))
    weekly_metrics = metrics(weekly.real, weekly.pred)
    return {**{f"daily_{k}": v for k, v in daily.items()},
            **{f"weekly_{k}": v for k, v in weekly_metrics.items()}}


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
    train, valid = split_frame(frame, windows, cfg)
    groups = feature_groups(frame)
    numeric = frame.select_dtypes(include=[np.number]).columns
    x_base = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = frame.target.to_numpy(float)
    rf_cfg = cfg["random_forest"]
    depths = [5, 7, 10]
    leaves = [1, 2, 5]
    splits = [2, 5]
    max_features = ["sqrt", 0.5, 1.0]
    criteria = ["squared_error", "poisson"]
    rows = []
    predictions = {}

    for name, requested_cols in groups.items():
        cols = list(dict.fromkeys(c for c in requested_cols if c in x_base.columns))
        x = x_base[cols]
        for depth in depths:
            for leaf in leaves:
                for split in splits:
                    for max_feature in max_features:
                        for criterion in criteria:
                            model = RandomForestRegressor(
                                n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                                min_samples_split=split, max_features=max_feature,
                                criterion=criterion, random_state=rf_cfg["random_state"], n_jobs=-1)
                            model.fit(x[train], y[train])
                            pred = model.predict(x[valid])
                            key = f"RF_{name}_d{depth}_l{leaf}_s{split}_f{max_feature}_{criterion}"
                            rows.append({"model": key, "family": "RF", "features": name,
                                         "max_depth": depth, "min_samples_leaf": leaf,
                                         "min_samples_split": split, "max_features": str(max_feature),
                                         "criterion": criterion, **score(y[valid], pred, frame.loc[valid])})
                            predictions[key] = (frame.loc[valid].copy(), pred)

    # Dos challengers de arboles con la misma poblacion y features FENO.
    cols = list(dict.fromkeys(c for c in groups["FENO"] if c in x_base.columns))
    x = x_base[cols]
    challengers = [
        ("ExtraTrees_FENO", ExtraTreesRegressor(n_estimators=300, max_depth=None,
            min_samples_leaf=1, max_features=1.0, random_state=rf_cfg["random_state"], n_jobs=-1)),
        ("HistGradientBoosting_FENO", HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31,
            learning_rate=.05, l2_regularization=1.0, loss="poisson", random_state=rf_cfg["random_state"])),
    ]
    for key, model in challengers:
        model.fit(x[train], y[train])
        pred = model.predict(x[valid])
        rows.append({"model": key, "family": "TREE_CHALLENGER", "features": "FENO", **score(y[valid], pred, frame.loc[valid])})
        predictions[key] = (frame.loc[valid].copy(), pred)

    result = pd.DataFrame(rows).sort_values(["daily_r2", "daily_wape"], ascending=[False, True])
    result.to_csv(evaluation / "metrics_hyperparametros.csv", index=False)
    for label, subset in {"r2": result.sort_values("daily_r2", ascending=False).head(1),
                          "wape": result.sort_values("daily_wape").head(1)}.items():
        key = subset.iloc[0].model
        trace, pred = predictions[key]
        trace[["finca", "bloque", "fecha_origen", "fecha_objetivo", "semana_proyeccion",
               "horizonte_dia", "target"]].rename(columns={"target": "real"}).assign(pred=pred).to_csv(
                   evaluation / f"predictions_best_{label}.csv", index=False)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
