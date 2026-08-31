"""Fase 5: clima diario causal y challenger M3 + clima."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import load_config
from evaluation.metrics import metrics
from models.climate import (build_climate_features, build_daily_climate,
                            load_hourly_climate)
from models.m3 import _period_for_date, fit_m3, simulate


def _split(windows, cfg):
    origins = (windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)
               .agg(ventana_evaluable=("ventana_evaluable", "first"))
               .sort_values("fecha_origen"))
    n = max(cfg["evaluation"]["min_train_windows"], int(len(origins) * 0.60))
    return origins.iloc[:n], origins.iloc[n:][lambda x: x.ventana_evaluable]


def _modify(matrix, signal, coefficient, mode):
    if coefficient == 0 or mode == "base":
        return matrix
    q, r, loss = matrix.Q.copy(), matrix.r.copy(), matrix.p.copy()
    factor = float(np.clip(1.0 + coefficient * signal, 0.1, 2.0))
    for j, destination in enumerate((1, 2, None)):
        old = r[j] if j == 2 else q[destination, j]
        new = min(old * factor, old + q[j, j], 1.0 - loss[j])
        delta = new - old
        if j == 2:
            r[j] = new
        else:
            q[destination, j] = new
        q[j, j] = max(0.0, q[j, j] - delta)
    from dataclasses import replace
    return replace(matrix, Q=q, r=r)


def _predict(row, matrix, alpha):
    x0 = np.array([row["conteo_RC_t0"], row["conteo_SS_t0"], row["conteo_AP_t0"]], float)
    days = (pd.Timestamp(row["fecha_objetivo"]) - pd.Timestamp(row["fecha_origen"])).days
    return simulate(matrix, x0, days, alpha).iloc[-1]["PC_dia_muestra"] * row["factor_extrapolacion_t0"]


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    raw, output = root / cfg["paths"]["raw"], root / cfg["paths"]["outputs"]
    datasets, evaluation, models = [output / x for x in ("datasets", "evaluation", "models")]
    for directory in (datasets, evaluation, models):
        directory.mkdir(parents=True, exist_ok=True)
    files = [cfg["sources"]["clima_2025"], cfg["sources"]["clima_2026"]]
    hourly, qa = load_hourly_climate(raw, files, cfg["climate"]["station_by_farm"]["LA PRADERA"])
    daily = build_daily_climate(hourly, cfg)
    daily.to_parquet(datasets / "clima_diario.parquet", index=False)
    pd.DataFrame([qa]).to_csv(evaluation / "qa_clima.csv", index=False)
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    windows = windows[windows["finca"].eq("LA PRADERA")].copy()
    train_origins, val_origins = _split(windows, cfg)
    origins = pd.concat([train_origins, val_origins], ignore_index=True)
    features = build_climate_features(daily, origins, cfg)
    features.to_parquet(datasets / "clima_features.parquet", index=False)
    feature_col = "gdd5_0_sum_7d"
    train_signal = features[features.fecha_origen.isin(train_origins.fecha_origen)][feature_col]
    mean, std = train_signal.mean(), train_signal.std() or 1.0
    signals = ((features[feature_col] - mean) / std).fillna(0.0)
    fmap = {(r.finca, r.bloque, r.fecha_origen): float(s)
            for r, s in zip(features.itertuples(), signals)}
    traditional = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    matrices = {}
    for _, origin in origins.iterrows():
        key = (origin.finca, origin.bloque, origin.fecha_origen)
        period = _period_for_date(pd.Timestamp(origin.fecha_origen), cfg["m3"]["periods"])
        matrices[key] = fit_m3(traditional, origin.finca, period, pd.Timestamp(origin.fecha_origen))
    val = windows.merge(val_origins[["finca", "bloque", "fecha_origen"]],
                        on=["finca", "bloque", "fecha_origen"], how="inner")
    rows = []
    for mode in ("base", "ingreso", "transicion", "hibrido"):
        for coefficient in cfg["climate"]["coefficient_grid"]:
            pred, real = [], []
            for _, row in val.iterrows():
                key = (row.finca, row.bloque, row.fecha_origen)
                signal = fmap[key]
                alpha = cfg["m3"]["baseline_ingress"]
                if mode in ("ingreso", "hibrido"):
                    alpha = float(np.clip(alpha * (1 + coefficient * signal), 0, 1))
                matrix = _modify(matrices[key], signal, coefficient,
                                 "transicion" if mode in ("transicion", "hibrido") else mode)
                pred.append(_predict(row, matrix, alpha))
                real.append(row.corte_real_dia)
            rows.append({"experiment_id": f"M3_CLIMA_{mode.upper()}", "split": "VALIDATION",
                         "finca_scope": "LA PRADERA", "causal": True,
                         "feature": feature_col, "coefficient": coefficient,
                         **metrics(pd.Series(real), pd.Series(pred))})
    result = pd.DataFrame(rows)
    result.to_csv(evaluation / "metrics_fase5_clima.csv", index=False)
    (models / "clima_manifest.json").write_text(pd.Series({"station_id": 746,
        "source": "EXTERIOR_ESTACION", "microclima_invernadero_disponible": False,
        "causal": True, "n_hourly": len(hourly), "n_daily": len(daily)}).to_json(indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
