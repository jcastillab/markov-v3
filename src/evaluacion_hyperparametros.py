"""Busqueda causal de modelos de conteo y hiperparametros.

La validacion conserva fechas de origen completas en el corte temporal. Se reportan
metricas diarias y semanales porque un R2 semanal alto no implica buen ajuste
por bloque y dia.
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from canonical import load_config
from evaluation.metrics import metrics
from evaluation.split import selection_masks
from models.supervised import build_supervised_dataset, feature_groups


def split_frame(frame, windows, cfg):
    return selection_masks(frame, cfg, windows["fecha_origen"])


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
           "criterion": params["criterion"],
           "hyperparameters": json.dumps({k: params[k] for k in (
               "n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",
               "max_features", "criterion", "random_state")}, sort_keys=True),
           **score(y[valid], pred, frame.loc[valid])}
    return row, pred


def add_selection_scores(result, cfg):
    """Rankea todas las metricas y crea un score compuesto orientado a semana."""
    lower_metrics = ["wape", "mae", "rmse"]
    scopes = ("daily", "weekly")
    for scope in scopes:
        for metric_name in lower_metrics:
            column = f"{scope}_{metric_name}"
            result[f"rank_{column}"] = result[column].rank(method="min", pct=True)
        bias = f"{scope}_bias_pct"
        result[f"rank_{bias}"] = result[bias].abs().rank(method="min", pct=True)
        r2 = f"{scope}_r2"
        result[f"rank_{r2}"] = 1 - result[r2].rank(method="min", pct=True)
    weights = cfg["evaluation"]["selection_metric_weights"]
    weekly_score = sum(weights[m] * result[f"rank_weekly_{m}"] for m in weights)
    daily_score = sum(weights[m] * result[f"rank_daily_{m}"] for m in weights)
    scope = cfg["evaluation"]["selection_scope_weights"]
    result["selection_score"] = scope["weekly"] * weekly_score + scope["daily"] * daily_score
    return result


def write_selected_prediction(label, result, predictions, frame, valid, evaluation):
    """Guarda trazabilidad, hiperparametros y metricas del modelo seleccionado."""
    selected = result.iloc[0]
    key = selected.model
    trace = frame.loc[valid, ["finca", "bloque", "fecha_origen", "fecha_objetivo",
                              "semana_proyeccion", "horizonte_dia", "target"]].copy()
    trace = trace.rename(columns={"target": "real"}).assign(pred=predictions[key])
    metadata = {"selected_by": label, "model": selected.model, "family": selected.family,
                "features": selected.features, "n_estimators": selected.get("n_estimators", np.nan),
                "max_depth": selected.get("max_depth", np.nan),
                "min_samples_leaf": selected.get("min_samples_leaf", np.nan),
                "min_samples_split": selected.get("min_samples_split", np.nan),
                "max_features": selected.get("max_features", np.nan),
                "criterion": selected.get("criterion", np.nan),
                "hyperparameters": selected.get("hyperparameters", "")}
    for metric_name in ("daily_wape", "daily_r2", "daily_mae", "daily_rmse", "daily_bias_pct",
                        "weekly_wape", "weekly_r2", "weekly_mae", "weekly_rmse", "weekly_bias_pct",
                        "selection_score"):
        metadata[metric_name] = selected.get(metric_name, np.nan)
    for column, value in reversed(list(metadata.items())):
        trace.insert(0, column, value)
    trace.to_csv(evaluation / f"predictions_best_{label}.csv", index=False)
    return selected


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
        hyperparameters = model.get_params(deep=False)
        family = "EXTRA_TREES" if isinstance(model, ExtraTreesRegressor) else "HIST_GRADIENT_BOOSTING"
        rows.append({"model": key, "family": family, "features": "FENO",
                     "hyperparameters": json.dumps(hyperparameters, sort_keys=True, default=str),
                     **score(y[valid], pred, frame.loc[valid])})
        predictions[key] = pred

    result = add_selection_scores(pd.DataFrame(rows), cfg)
    result = result.sort_values("selection_score")
    result.to_csv(evaluation / "metrics_hyperparametros.csv", index=False)
    selection_targets = {
        "daily_wape": "daily_wape", "daily_r2": "daily_r2", "daily_mae": "daily_mae",
        "daily_rmse": "daily_rmse", "daily_bias_pct": "daily_bias_pct",
        "weekly_wape": "weekly_wape", "weekly_r2": "weekly_r2", "weekly_mae": "weekly_mae",
        "weekly_rmse": "weekly_rmse", "weekly_bias_pct": "weekly_bias_pct", "composite": "composite",
    }
    for label in selection_targets:
        metric = selection_targets[label]
        if metric == "composite":
            result_for_selection = result
        else:
            result_for_selection = result.copy()
            column = metric
            if metric.endswith("r2"):
                result_for_selection["_selection"] = -result_for_selection[column]
            elif metric.endswith("bias_pct"):
                result_for_selection["_selection"] = result_for_selection[column].abs()
            else:
                result_for_selection["_selection"] = result_for_selection[column]
            result_for_selection = result_for_selection.sort_values("_selection")
        write_selected_prediction(label, result_for_selection, predictions, frame, valid, evaluation)
    selection_target = cfg["evaluation"]["model_selection_target"]
    selected_path = evaluation / f"predictions_best_{selection_target}.csv"
    selected = pd.read_csv(selected_path, nrows=1).iloc[0]
    manifest = {
        "model": selected["model"],
        "family": selected["family"],
        "features": selected["features"],
        "selected_by": selection_target,
        "hyperparameters": json.loads(selected["hyperparameters"]),
        "selection": {
            key: float(selected[key]) for key in (
                "daily_wape", "daily_r2", "weekly_wape", "weekly_r2", "selection_score"
            )
        },
        "selection_rows": int(valid.sum()),
    }
    (evaluation / "selected_model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
