"""Evaluacion rolling-origin causal para cubrir todas las semanas posibles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from canonical import load_config
from evaluation.metrics import metrics
from models.m3 import _period_for_date, fit_m3, simulate
from models.supervised import build_supervised_dataset, feature_groups


def _m3_predictions(windows, intervals, cfg):
    rows = []
    for _, row in windows.iterrows():
        origin = pd.Timestamp(row.fecha_origen)
        period = _period_for_date(origin, cfg["m3"]["periods"])
        matrix = fit_m3(intervals, row.finca, period, origin)
        x0 = np.array([row.conteo_RC_t0, row.conteo_SS_t0, row.conteo_AP_t0], float)
        lead = (pd.Timestamp(row.fecha_objetivo) - origin).days
        result = simulate(matrix, x0, lead, cfg["m3"]["baseline_ingress"]).iloc[-1]
        rows.append({"modelo": "E00_M3_BASE_ROLLING", "finca": row.finca, "bloque": row.bloque,
                     "fecha_origen": row.fecha_origen, "fecha_objetivo": row.fecha_objetivo,
                     "semana_proyeccion": row.semana_proyeccion, "horizonte_dia": row.horizonte_dia,
                     "estado_ventana": row.estado_ventana, "real": row.corte_real_dia,
                     "proyectado": result.PC_dia_muestra * row.factor_extrapolacion_t0})
    return pd.DataFrame(rows)


def _rf_predictions(frame, cfg):
    groups = feature_groups(frame)
    cols = list(dict.fromkeys(c for c in groups["FENO"] if c in frame.select_dtypes(include=[np.number]).columns))
    x_all = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    origins = sorted(pd.to_datetime(frame.fecha_origen).unique())
    min_origins = int(cfg["supervised"]["rolling_min_train_origins"])
    rf_cfg = cfg["random_forest"]
    predictions = []
    for position, origin in enumerate(origins):
        if position < min_origins:
            continue
        current = pd.to_datetime(frame.fecha_origen).eq(origin).to_numpy()
        previous = (pd.to_datetime(frame.fecha_origen) < origin).to_numpy() & frame.target.notna().to_numpy()
        if previous.sum() == 0 or current.sum() == 0:
            continue
        model = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth_grid"][0], min_samples_leaf=rf_cfg["min_samples_leaf_grid"][0],
            min_samples_split=rf_cfg["min_samples_split_grid"][0], max_features=rf_cfg["max_features"][0],
            random_state=rf_cfg["random_state"], n_jobs=-1)
        model.fit(x_all[previous], frame.loc[previous, "target"])
        pred = model.predict(x_all[current])
        current_frame = frame.loc[current, ["finca", "bloque", "fecha_origen", "fecha_objetivo",
                                            "semana_proyeccion", "horizonte_dia", "estado_ventana", "target"]].copy()
        current_frame = current_frame.rename(columns={"target": "real"})
        current_frame["modelo"] = "RF_H1_H7_FENO_ROLLING"
        current_frame["proyectado"] = pred
        predictions.append(current_frame)
    return pd.concat(predictions, ignore_index=True)


def _metrics_by_week(frame):
    rows = []
    for (model, finca, week), group in frame.groupby(["modelo", "finca", "semana_proyeccion"]):
        observed = group[group.real.notna()]
        if observed.empty:
            rows.append({"modelo": model, "finca": finca, "semana_proyeccion": week,
                         "estado": "PENDIENTE_REAL", "n_dias_reales": 0})
            continue
        error = observed.proyectado - observed.real
        denom = observed.real.abs().sum()
        rows.append({"modelo": model, "finca": finca, "semana_proyeccion": week,
                     "estado": "VALIDA" if len(observed) == len(group) else "PARCIAL",
                     "n_dias_reales": len(observed), "n_dias_pronosticados": len(group),
                     "real_comparable": observed.real.sum(), "proyectado_comparable": observed.proyectado.sum(),
                     "proyectado_total": group.proyectado.sum(),
                     "wape": error.abs().sum() / denom if denom else np.nan,
                     "mae": error.abs().mean(), "rmse": np.sqrt((error ** 2).mean()),
                     "acierto_pct": 1 - (observed.real.sum() - observed.proyectado.sum()) / observed.real.sum()
                     if observed.real.sum() else np.nan})
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config/pipeline.yaml")
    out = root / cfg["paths"]["outputs"]
    datasets, evaluation = out / "datasets", out / "evaluation"
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    intervals = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    pruning = pd.read_parquet(datasets / "poda_features.parquet")
    climate = pd.read_parquet(datasets / "clima_features.parquet")
    frame = build_supervised_dataset(windows, fact, intervals, cfg, pruning, climate, include_incomplete=True)
    m3 = _m3_predictions(windows, intervals, cfg)
    rf = _rf_predictions(frame, cfg)
    for model, predictions in (("E00_M3_BASE_ROLLING", m3), ("RF_H1_H7_FENO_ROLLING", rf)):
        predictions.to_csv(evaluation / f"predictions_{model.lower()}.csv", index=False)
    weekly = pd.concat([_metrics_by_week(m3), _metrics_by_week(rf)], ignore_index=True)
    weekly.to_csv(evaluation / "metrics_rolling_origin_semanal.csv", index=False)
    daily_metrics = []
    for model, group in pd.concat([m3, rf]).groupby("modelo"):
        observed = group[group.real.notna()]
        if observed.empty:
            continue
        daily_metrics.append({"modelo": model, "nivel": "DIARIO_OBSERVADO", "n": len(observed),
                              **metrics(observed.real, observed.proyectado)})
    pd.DataFrame(daily_metrics).to_csv(evaluation / "metrics_rolling_origin.csv", index=False)
    print(weekly.groupby(["modelo", "estado"]).size().to_string())
    print(pd.DataFrame(daily_metrics).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
