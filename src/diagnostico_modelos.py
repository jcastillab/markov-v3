"""Diagnosticos de ajuste, variables y estabilidad para modelos supervisados."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from canonical import load_config
from evaluation.metrics import metrics
from evaluation.split import temporal_masks
from models.supervised import build_supervised_dataset, feature_groups


def split_frame(frame: pd.DataFrame, windows: pd.DataFrame, cfg: dict):
    return temporal_masks(frame, cfg, origin_values=windows["fecha_origen"])


def _metric_row(y, pred, scope, model, horizon):
    values = metrics(pd.Series(y), pd.Series(pred))
    return {"modelo": model, "horizonte": horizon, "scope": scope, **values}


def fit_horizon_diagnostics(frame, windows, cfg, cols):
    """Compara train/validation con los siete RF que forman H1-H7."""
    train, valid = split_frame(frame, windows, cfg)
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = frame.target.to_numpy(float)
    rf_cfg = cfg["random_forest"]
    model_n_jobs = int(rf_cfg.get("parallel", {}).get("model_n_jobs", 1))
    rows = []
    importance_rows = []
    for horizon in range(1, int(cfg["forecast"]["horizon_days"]) + 1):
        train_h = train & frame.horizonte_dia.eq(horizon).to_numpy()
        valid_h = valid & frame.horizonte_dia.eq(horizon).to_numpy()
        model = RandomForestRegressor(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth_grid"][0],
            min_samples_leaf=rf_cfg["min_samples_leaf_grid"][0],
            min_samples_split=rf_cfg["min_samples_split_grid"][0],
            max_features=rf_cfg["max_features"][0],
            random_state=rf_cfg["random_state"], n_jobs=model_n_jobs)
        model.fit(x[train_h], y[train_h])
        importance_rows.extend({"horizonte": horizon, "variable": col,
                                "importance": float(value)}
                               for col, value in zip(cols, model.feature_importances_))
        rows.extend((_metric_row(y[train_h], model.predict(x[train_h]), "TRAIN",
                                 "RF_H1_H7_FENO", horizon),
                     _metric_row(y[valid_h], model.predict(x[valid_h]), "VALIDATION",
                                 "RF_H1_H7_FENO", horizon)))
    result = pd.DataFrame(rows)
    pivot = result.pivot(index="horizonte", columns="scope", values="wape").reset_index()
    pivot["gap_wape"] = pivot["VALIDATION"] - pivot["TRAIN"]
    result = result.merge(pivot[["horizonte", "gap_wape"]], on="horizonte", how="left")
    result["riesgo_sobreajuste"] = np.select(
        [result.gap_wape.gt(.20), result.gap_wape.gt(.10)],
        ["ALTO", "MEDIO"], default="BAJO")
    return result, pd.DataFrame(importance_rows).sort_values(
        ["horizonte", "importance"], ascending=[True, False])


def feature_diagnostics(frame, cols, train, valid):
    rows = []
    for col in cols:
        values = frame[col]
        train_values, valid_values = values[train], values[valid]
        train_mean, valid_mean = train_values.mean(), valid_values.mean()
        train_std = train_values.std()
        rows.append({
            "variable": col,
            "n": len(values),
            "faltantes": int(values.isna().sum()),
            "faltantes_pct": float(values.isna().mean()),
            "unicos": int(values.nunique(dropna=True)),
            "media_train": train_mean,
            "media_validacion": valid_mean,
            "std_train": train_std,
            "diferencia_medias_std": ((valid_mean - train_mean) / train_std
                                       if pd.notna(train_std) and train_std > 0 else 0.0),
        })
    return pd.DataFrame(rows).sort_values("faltantes_pct", ascending=False)


def collinearity_diagnostics(frame, cols):
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.loc[:, x.nunique(dropna=False).gt(1)]
    corr = x.corr().replace([np.inf, -np.inf], np.nan)
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= .80:
                pairs.append({"variable_1": left, "variable_2": right,
                              "correlacion": float(value),
                              "correlacion_abs": abs(float(value))})
    correlation = corr.to_numpy()
    inverse = np.linalg.pinv(correlation)
    vif = pd.DataFrame({"variable": corr.columns,
                        "vif": np.diag(inverse)})
    vif["vif"] = vif.vif.clip(lower=1.0)
    vif["riesgo"] = np.select([vif.vif.ge(10), vif.vif.ge(5)],
                              ["ALTO", "MEDIO"], default="BAJO")
    return pd.DataFrame(pairs).sort_values("correlacion_abs", ascending=False), vif.sort_values("vif", ascending=False)


def leakage_audit(frame, cols):
    forbidden = {"target", "real", "corte_comercial_real", "fecha_objetivo"}
    suspicious = sorted(set(cols) & forbidden)
    return pd.DataFrame([{
        "prueba": "variables_prohibidas_en_features",
        "resultado": "FALLA" if suspicious else "OK",
        "detalle": ", ".join(suspicious) if suspicious else "Ninguna",
    }, {
        "prueba": "features_numericas",
        "resultado": "OK" if all(pd.api.types.is_numeric_dtype(frame[c]) for c in cols) else "FALLA",
        "detalle": f"{len(cols)} variables",
    }])


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
    cols = list(dict.fromkeys(c for c in feature_groups(frame)["FENO"]
                             if c in frame.select_dtypes(include=[np.number]).columns
                             and c != "target"))
    train, valid = split_frame(frame, windows, cfg)
    overfit, importance = fit_horizon_diagnostics(frame, windows, cfg, cols)
    pairs, vif = collinearity_diagnostics(frame, cols)
    features = feature_diagnostics(frame, cols, train, valid)
    leakage = leakage_audit(frame, cols)
    overfit.to_csv(evaluation / "diagnostico_sobreajuste.csv", index=False)
    importance.to_csv(evaluation / "diagnostico_importancia_horizonte.csv", index=False)
    pairs.to_csv(evaluation / "diagnostico_correlaciones_altas.csv", index=False)
    vif.to_csv(evaluation / "diagnostico_vif.csv", index=False)
    features.to_csv(evaluation / "diagnostico_features.csv", index=False)
    leakage.to_csv(evaluation / "diagnostico_leakage.csv", index=False)
    (evaluation / "diagnostico_manifest.json").write_text(json.dumps({
        "model": "RF_H1_H7_FENO", "features": cols, "n_features": len(cols),
        "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
        "threshold_high_correlation": .80, "threshold_vif_medium": 5,
        "threshold_vif_high": 10,
    }, indent=2), encoding="utf-8")
    print(overfit.to_string(index=False))
    print(f"Variables auditadas: {len(cols)}; correlaciones altas: {len(pairs)}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
