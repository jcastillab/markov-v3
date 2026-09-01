"""Busqueda causal de modelos de conteo y hiperparametros.

La validacion conserva los mismos origenes 60/40 de Fase 6. Se reportan
metricas diarias y semanales porque un R2 semanal alto no implica buen ajuste
por bloque y dia.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
    return valid, ~valid


def score(y, pred, frame):
    daily = metrics(pd.Series(y), pd.Series(pred))
    detail = pd.DataFrame({"finca": frame.finca.to_numpy(), "bloque": frame.bloque.to_numpy(),
                           "fecha_origen": frame.fecha_origen.to_numpy(), "real": y, "pred": pred})
    weekly = detail.groupby(["finca", "bloque", "fecha_origen"], as_index=False).agg(
        real=("real", "sum"), pred=("pred", "sum"))
    weekly_metrics = metrics(weekly.real, weekly.pred)
    return {**{f"daily_{k}": v for k, v in daily.items()},
            **{f"weekly_{k}": v for k, v in weekly_metrics.items()}}


def available_jobs(parallel_cfg):
    """Calcula los procesos de experimentos sin sobreasignar la máquina."""
    requested = parallel_cfg.get("n_jobs", "auto")
    if requested == "auto":
        requested = (os.cpu_count() or 1) - int(parallel_cfg.get("reserve_cpus", 0))
    return max(1, min(int(requested), os.cpu_count() or 1))


def fit_rf_experiment(name, cols, x, y, train, valid, frame, params, model_n_jobs):
    model = RandomForestRegressor(**params, n_jobs=model_n_jobs)
    model.fit(x[train], y[train])
    pred = model.predict(x[valid])
    key = (f"RF_{name}_n{params['n_estimators']}_d{params['max_depth']}"
           f"_l{params['min_samples_leaf']}_s{params['min_samples_split']}"
           f"_f{params['max_features']}_{params['criterion']}")
    row = {"model": key, "family": "RF", "features": name,
           "n_estimators": params["n_estimators"], "max_depth": params["max_depth"],
           "min_samples_leaf": params["min_samples_leaf"],
           "min_samples_split": params["min_samples_split"],
           "max_features": str(params["max_features"]),
           "criterion": params["criterion"], **score(y[valid], pred, frame.loc[valid])}
    return row, pred


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
    depths = rf_cfg["max_depth_grid"]
    leaves = rf_cfg["min_samples_leaf_grid"]
    splits = rf_cfg["min_samples_split_grid"]
    max_features = rf_cfg["max_features"]
    criteria = rf_cfg["criterion_grid"]
    n_estimators = rf_cfg.get("n_estimators_grid", [rf_cfg["n_estimators"]])
    jobs = available_jobs(rf_cfg.get("parallel", {}))
    model_n_jobs = int(rf_cfg.get("parallel", {}).get("model_n_jobs", 1))
    tasks = []

    for name, requested_cols in groups.items():
        cols = list(dict.fromkeys(c for c in requested_cols if c in x_base.columns))
        x = x_base[cols]
        for n in n_estimators:
            for depth in depths:
                for leaf in leaves:
                    for split in splits:
                        for max_feature in max_features:
                            for criterion in criteria:
                                tasks.append((name, cols, x, y, train, valid, frame,
                                              {"n_estimators": n, "max_depth": depth,
                                               "min_samples_leaf": leaf,
                                               "min_samples_split": split,
                                               "max_features": max_feature, "criterion": criterion,
                                               "random_state": rf_cfg["random_state"]},
                                              model_n_jobs))

    print(f"Ejecutando {len(tasks)} combinaciones RF con {jobs} procesos y "
          f"{model_n_jobs} hilo(s) por modelo")
    results = Parallel(n_jobs=jobs, prefer="processes")(
        delayed(fit_rf_experiment)(*task) for task in tasks)
    rows = [row for row, _ in results]
    predictions = {row["model"]: pred for (row, pred) in results}

    # Dos challengers de arboles con la misma poblacion y features FENO.
    cols = list(dict.fromkeys(c for c in groups["FENO"] if c in x_base.columns))
    x = x_base[cols]
    challengers = [
        ("ExtraTrees_FENO", ExtraTreesRegressor(n_estimators=300, max_depth=None,
            min_samples_leaf=1, max_features=1.0, random_state=rf_cfg["random_state"], n_jobs=model_n_jobs)),
        ("HistGradientBoosting_FENO", HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31,
            learning_rate=.05, l2_regularization=1.0, loss="poisson", random_state=rf_cfg["random_state"])),
    ]
    for key, model in challengers:
        model.fit(x[train], y[train])
        pred = model.predict(x[valid])
        rows.append({"model": key, "family": "TREE_CHALLENGER", "features": "FENO", **score(y[valid], pred, frame.loc[valid])})
        predictions[key] = pred

    result = pd.DataFrame(rows).sort_values(["daily_r2", "daily_wape"], ascending=[False, True])
    result.to_csv(evaluation / "metrics_hyperparametros.csv", index=False)
    for label, subset in {"r2": result.sort_values("daily_r2", ascending=False).head(1),
                          "wape": result.sort_values("daily_wape").head(1)}.items():
        selected = subset.iloc[0]
        key = selected.model
        pred = predictions[key]
        trace = frame.loc[valid, ["finca", "bloque", "fecha_origen", "fecha_objetivo",
                                  "semana_proyeccion", "horizonte_dia", "target"]].copy()
        trace = trace.rename(columns={"target": "real"}).assign(pred=pred)
        metadata = {
            "selected_by": label,
            "model": selected.model,
            "family": selected.family,
            "features": selected.features,
            "n_estimators": selected.get("n_estimators", np.nan),
            "max_depth": selected.get("max_depth", np.nan),
            "min_samples_leaf": selected.get("min_samples_leaf", np.nan),
            "min_samples_split": selected.get("min_samples_split", np.nan),
            "max_features": selected.get("max_features", np.nan),
            "criterion": selected.get("criterion", np.nan),
            "daily_wape": selected.daily_wape,
            "daily_r2": selected.daily_r2,
            "daily_mae": selected.daily_mae,
            "daily_rmse": selected.daily_rmse,
            "weekly_wape": selected.weekly_wape,
            "weekly_r2": selected.weekly_r2,
        }
        for column, value in metadata.items():
            trace.insert(0, column, value)
        trace.to_csv(evaluation / f"predictions_best_{label}.csv", index=False)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
