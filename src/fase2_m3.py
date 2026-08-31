"""Fase 2: M3 causal, ingreso RC fijo y calibrado."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import load_config
from evaluation.metrics import metrics
from models.m3 import _period_for_date, fit_m3, load_traditional_intervals, simulate


def predict_window(row: pd.Series, intervals: pd.DataFrame, cfg: dict,
                   alpha: float) -> list[dict]:
    origin = pd.Timestamp(row["fecha_origen"])
    target = pd.Timestamp(row["fecha_objetivo"])
    period = _period_for_date(origin, cfg["m3"]["periods"])
    matrix = fit_m3(intervals, row["finca"], period, max_date=origin)
    x0 = np.array([row["conteo_RC_t0"], row["conteo_SS_t0"], row["conteo_AP_t0"]], float)
    lead = (target - origin).days
    sim = simulate(matrix, x0, lead, alpha)
    out = sim.iloc[-1]
    return [{"experiment_id": "", "finca": row["finca"], "bloque": row["bloque"],
             "fecha_origen": origin, "fecha_objetivo": target,
             "horizonte_dia": row["horizonte_dia"],
             "semana_proyeccion": row["semana_proyeccion"],
             "pred_muestra": float(out["PC_dia_muestra"]),
             "pred_bloque": float(out["PC_dia_muestra"] * row["factor_extrapolacion_t0"]),
             "real": row["corte_real_dia"], "alpha_ingreso_RC": alpha,
             "periodo": period, "causal": True,
             "fecha_max_dato_modelo": origin}]


def run_experiment(windows: pd.DataFrame, intervals: pd.DataFrame, cfg: dict,
                   experiment_id: str, alpha: float, eligible: pd.Series) -> tuple[pd.DataFrame, dict]:
    rows = []
    for _, row in windows[eligible].iterrows():
        item = predict_window(row, intervals, cfg, alpha)[0]
        item["experiment_id"] = experiment_id
        rows.append(item)
    pred = pd.DataFrame(rows)
    return pred, metrics(pred["real"], pred["pred_bloque"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    raw = root / cfg["paths"]["raw"]
    datasets = root / cfg["paths"]["outputs"] / "datasets"
    eval_dir = root / cfg["paths"]["outputs"] / "evaluation"
    model_dir = root / cfg["paths"]["outputs"] / "models"
    pred_dir = root / cfg["paths"]["outputs"] / "predictions"
    for directory in (eval_dir, model_dir, pred_dir):
        directory.mkdir(parents=True, exist_ok=True)
    intervals = load_traditional_intervals(raw, cfg)
    intervals.to_parquet(datasets / "transition_intervals_tradicional.parquet", index=False)
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    grouped = (windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)
               .agg(ventana_evaluable=("ventana_evaluable", "first")))
    origins = grouped.sort_values("fecha_origen").reset_index(drop=True)
    n_train = max(cfg["evaluation"]["min_train_windows"], int(len(origins) * 0.60))
    train_origins = set(map(tuple, origins.iloc[:n_train][["finca", "bloque", "fecha_origen"]].itertuples(index=False, name=None)))
    validation_origins = set(map(tuple, origins.iloc[n_train:][["finca", "bloque", "fecha_origen"]].itertuples(index=False, name=None)))
    keys = list(zip(windows["finca"], windows["bloque"], windows["fecha_origen"]))
    train_mask = windows["ventana_evaluable"] & pd.Series([k in train_origins for k in keys], index=windows.index)
    val_mask = windows["ventana_evaluable"] & pd.Series([k in validation_origins for k in keys], index=windows.index)
    grid = cfg["m3"]["ingress_grid"]
    train_scores = []
    for alpha in grid:
        _, score = run_experiment(windows, intervals, cfg, "", alpha, train_mask)
        train_scores.append({"alpha_ingreso_RC": alpha, **score, "split": "TRAIN"})
    alpha_best = min(train_scores, key=lambda x: x["wape"])["alpha_ingreso_RC"]
    predictions = []
    metric_rows = []
    matrix_rows = []
    max_observed_date = intervals["fecha"].max()
    for finca in cfg["project"]["target_farms"]:
        for period in cfg["m3"]["periods"]:
            matrix = fit_m3(intervals, finca, period["name"], max_date=max_observed_date)
            for _, audit_row in matrix.audit.iterrows():
                matrix_rows.append(audit_row.to_dict())
    baseline_alpha = float(cfg["m3"]["baseline_ingress"])
    for experiment_id, alpha in [("E00_M3_BASE", baseline_alpha), ("E01_M3_INGRESO_CALIBRADO", alpha_best)]:
        pred, score = run_experiment(windows, intervals, cfg, experiment_id, alpha, val_mask)
        pred.to_csv(pred_dir / f"{experiment_id}.csv", index=False)
        metric_rows.append({"experiment_id": experiment_id, "split": "VALIDATION",
                            "alpha_ingreso_RC": alpha, "causal": True, **score})
        predictions.append(pred)
    pd.DataFrame(train_scores).to_csv(eval_dir / "m3_alpha_inner_validation.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(eval_dir / "metrics_fase2_m3.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(model_dir / "m3_matrix_audit.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(pred_dir / "predictions_m3_validation.csv", index=False)
    intervals.groupby(["finca", "periodo", "estado_origen", "estado_destino", "evento"], as_index=False).size().to_csv(
        model_dir / "m3_transition_audit_counts.csv", index=False)
    manifest = {
        "phase": 2,
        "models": ["E00_M3_BASE", "E01_M3_INGRESO_CALIBRADO"],
        "causal_predictions": True,
        "alpha_selected_inner_validation": float(alpha_best),
        "alpha_grid": grid,
        "transition_rows": int(len(intervals)),
        "source_max_date_used_for_predictions": "per_origin_t0",
        "matrix_orientation": "Q[destino, origen]",
        "target_scale": "BLOQUE",
    }
    (model_dir / "m3_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Intervalos M3 validos: {len(intervals):,}")
    print(f"Origenes evaluables: {len(origins[origins.ventana_evaluable]):,}; TRAIN: {train_mask.sum() // 7:,}; VALIDATION: {val_mask.sum() // 7:,}")
    print(f"Alpha ganador inner validation: {alpha_best:.2f}")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
