"""Evaluacion rolling-origin causal para cubrir todas las semanas posibles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from canonical import load_config
    from evaluation.metrics import metrics
    from evaluation.split import holdout_start
    from models.m3 import _period_for_date, fit_m3, simulate
    from models.selection import estimator_from_spec, read_selection
    from models.supervised import build_supervised_dataset, feature_groups
except ModuleNotFoundError:
    from src.canonical import load_config
    from src.evaluation.metrics import metrics
    from src.evaluation.split import holdout_start
    from src.models.m3 import _period_for_date, fit_m3, simulate
    from src.models.selection import estimator_from_spec, read_selection
    from src.models.supervised import build_supervised_dataset, feature_groups


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


KEY_COLUMNS = ["finca", "bloque", "fecha_origen", "fecha_objetivo", "horizonte_dia"]


def _selected_predictions(frame, cfg, selected, evaluation_start=None):
    groups = feature_groups(frame)
    feature_name = selected["features"]
    if feature_name not in groups:
        raise ValueError(f"Grupo de features seleccionado no disponible: {feature_name}")
    numeric = frame.select_dtypes(include=[np.number]).columns
    cols = list(dict.fromkeys(c for c in groups[feature_name] if c in numeric and c != "target"))
    if not cols:
        raise ValueError(f"El grupo {feature_name} no contiene features numericas")
    x_all = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    origins = sorted(pd.to_datetime(frame.fecha_origen).unique())
    min_origins = int(cfg["supervised"]["rolling_min_train_origins"])
    predictions = []
    for position, origin in enumerate(origins):
        if position < min_origins:
            continue
        if evaluation_start is not None and origin < evaluation_start:
            continue
        current = pd.to_datetime(frame.fecha_origen).eq(origin).to_numpy()
        previous = ((pd.to_datetime(frame.fecha_origen) < origin) &
                    (pd.to_datetime(frame.fecha_objetivo) < origin) &
                    frame.target.notna()).to_numpy()
        if previous.sum() == 0 or current.sum() == 0:
            continue
        model = estimator_from_spec(selected, cfg)
        model.fit(x_all[previous], frame.loc[previous, "target"])
        pred = model.predict(x_all[current])
        current_frame = frame.loc[current, ["finca", "bloque", "fecha_origen", "fecha_objetivo",
                                            "semana_proyeccion", "horizonte_dia", "estado_ventana", "target"]].copy()
        current_frame = current_frame.rename(columns={"target": "real"})
        current_frame["modelo"] = "MODELO_SELECCIONADO_ROLLING"
        current_frame["modelo_seleccionado"] = selected["model"]
        current_frame["familia"] = selected["family"]
        current_frame["features"] = feature_name
        current_frame["proyectado"] = pred
        predictions.append(current_frame)
    return pd.concat(predictions, ignore_index=True)


def _common_observed(left, right):
    """Recorta ambos modelos a exactamente las mismas claves con real observado."""
    left_valid = left[left.real.notna()].copy()
    right_valid = right[right.real.notna()].copy()
    common = left_valid[KEY_COLUMNS].merge(
        right_valid[KEY_COLUMNS], on=KEY_COLUMNS, how="inner"
    ).drop_duplicates()
    if common.empty:
        raise ValueError("M3 y el modelo seleccionado no tienen observaciones rolling comunes")
    return (
        left_valid.merge(common, on=KEY_COLUMNS, how="inner"),
        right_valid.merge(common, on=KEY_COLUMNS, how="inner"),
    )


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
        complete = (len(observed) == len(group) and
                    ("estado_ventana" not in group or group.estado_ventana.eq("VALIDA").all()))
        rows.append({"modelo": model, "finca": finca, "semana_proyeccion": week,
                     "estado": "VALIDA" if complete else "PARCIAL",
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
    selected = read_selection(evaluation / "selected_model_manifest.json")
    evaluation_start = holdout_start(windows["fecha_origen"], cfg)
    challenger = _selected_predictions(frame, cfg, selected, evaluation_start)
    m3_common, challenger_common = _common_observed(m3, challenger)
    for model, predictions in (("E00_M3_BASE_ROLLING", m3_common),
                               ("MODELO_SELECCIONADO_ROLLING", challenger_common)):
        predictions.to_csv(evaluation / f"predictions_{model.lower()}.csv", index=False)
    weekly = pd.concat([_metrics_by_week(m3_common), _metrics_by_week(challenger_common)], ignore_index=True)
    weekly.to_csv(evaluation / "metrics_rolling_origin_semanal.csv", index=False)
    daily_metrics = []
    prediction_files = {
        "E00_M3_BASE_ROLLING": "predictions_e00_m3_base_rolling.csv",
        "MODELO_SELECCIONADO_ROLLING": "predictions_modelo_seleccionado_rolling.csv",
    }
    for model, group in pd.concat([m3_common, challenger_common]).groupby("modelo"):
        observed = group[group.real.notna()]
        if observed.empty:
            continue
        experiment_id = "E00_M3_BASE_ROLLING" if model.startswith("E00") else selected["model"]
        daily_metrics.append({"experiment_id": experiment_id, "modelo": model,
                              "family": "M3" if model.startswith("E00") else selected["family"],
                              "features": "FENO_M3" if model.startswith("E00") else selected["features"],
                              "split": "ROLLING_ORIGIN_COMMON", "causal": True,
                              "nivel": "DIARIO_OBSERVADO", "prediction_file": prediction_files[model],
                              **metrics(observed.real, observed.proyectado)})
    pd.DataFrame(daily_metrics).to_csv(evaluation / "metrics_rolling_origin.csv", index=False)
    print(weekly.groupby(["modelo", "estado"]).size().to_string())
    print(pd.DataFrame(daily_metrics).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
