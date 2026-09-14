"""Evaluacion rolling-origin causal y poblacion comun para todos los challengers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from canonical import load_config
    from evaluation.metrics import metrics
    from evaluation.split import holdout_start
    from models.bayes import CovariateHierarchicalNB, DirichletM3, HierarchicalNB
    from models.m3 import M3Matrix, _period_for_date, fit_m3, simulate
    from models.selection import estimator_from_spec, read_selection
    from models.supervised import NegativeBinomialGLM, build_supervised_dataset, feature_groups
except ModuleNotFoundError:
    from src.canonical import load_config
    from src.evaluation.metrics import metrics
    from src.evaluation.split import holdout_start
    from src.models.bayes import CovariateHierarchicalNB, DirichletM3, HierarchicalNB
    from src.models.m3 import M3Matrix, _period_for_date, fit_m3, simulate
    from src.models.selection import estimator_from_spec, read_selection
    from src.models.supervised import NegativeBinomialGLM, build_supervised_dataset, feature_groups


KEY_COLUMNS = ["finca", "bloque", "fecha_origen", "fecha_objetivo", "horizonte_dia"]
BASE_COLUMNS = ["finca", "bloque", "fecha_origen", "fecha_objetivo", "semana_proyeccion",
                "horizonte_dia", "estado_ventana", "target"]


def _frame_rows(frame, current, model, pred, family, features):
    result = frame.loc[current, BASE_COLUMNS].copy().rename(columns={"target": "real"})
    result["modelo"] = model
    result["familia"] = family
    result["features"] = features
    result["proyectado"] = np.asarray(pred, float)
    return result


def _m3_predictions(windows, intervals, cfg, model="E00_M3_BASE_ROLLING", alpha=None):
    alpha = cfg["m3"]["baseline_ingress"] if alpha is None else alpha
    rows = []
    for _, row in windows.iterrows():
        origin = pd.Timestamp(row.fecha_origen)
        period = _period_for_date(origin, cfg["m3"]["periods"])
        matrix = fit_m3(intervals, row.finca, period, origin)
        x0 = np.array([row.conteo_RC_t0, row.conteo_SS_t0, row.conteo_AP_t0], float)
        lead = (pd.Timestamp(row.fecha_objetivo) - origin).days
        result = simulate(matrix, x0, lead, alpha).iloc[-1]
        rows.append({"modelo": model, "familia": "M3", "features": "FENO_M3",
                     "finca": row.finca, "bloque": row.bloque, "fecha_origen": row.fecha_origen,
                     "fecha_objetivo": row.fecha_objetivo, "semana_proyeccion": row.semana_proyeccion,
                     "horizonte_dia": row.horizonte_dia, "estado_ventana": row.estado_ventana,
                     "real": row.corte_real_dia,
                     "proyectado": result.PC_dia_muestra * row.factor_extrapolacion_t0})
    return pd.DataFrame(rows)


def _origins(frame, cfg, evaluation_start):
    values = sorted(pd.to_datetime(frame.fecha_origen).unique())
    minimum = int(cfg["supervised"]["rolling_min_train_origins"])
    for position, origin in enumerate(values):
        if position < minimum or origin < evaluation_start:
            continue
        current = pd.to_datetime(frame.fecha_origen).eq(origin).to_numpy()
        previous = ((pd.to_datetime(frame.fecha_origen) < origin) &
                    (pd.to_datetime(frame.fecha_objetivo) < origin) & frame.target.notna()).to_numpy()
        if current.sum() and previous.sum():
            yield origin, current, previous


def _supervised_predictions(frame, cfg, spec, model, evaluation_start):
    groups = feature_groups(frame)
    name = spec["features"]
    numeric = frame.select_dtypes(include=[np.number]).columns
    cols = list(dict.fromkeys(c for c in groups[name] if c in numeric and c != "target"))
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    rows = []
    for _, current, previous in _origins(frame, cfg, evaluation_start):
        estimator = estimator_from_spec(spec, cfg)
        estimator.fit(x.loc[previous], frame.loc[previous, "target"])
        rows.append(_frame_rows(frame, current, model, estimator.predict(x.loc[current]), spec["family"], name))
    return pd.concat(rows, ignore_index=True)


def _selected_predictions(frame, cfg, spec):
    """Compatibilidad para el challenger seleccionado y sus pruebas unitarias."""
    return _supervised_predictions(frame, cfg, spec, spec.get("model", "MODELO_SELECCIONADO_ROLLING"), pd.Timestamp.min)


def _glm_predictions(frame, cfg, feature_name, model, evaluation_start):
    groups = feature_groups(frame)
    numeric = frame.select_dtypes(include=[np.number]).columns
    cols = list(dict.fromkeys(c for c in groups[feature_name] if c in numeric and c != "target"))
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    rows = []
    for _, current, previous in _origins(frame, cfg, evaluation_start):
        estimator = NegativeBinomialGLM(cfg["supervised"]["nb_alpha"], cfg["supervised"]["nb_max_iter"])
        estimator.fit(x.loc[previous], frame.loc[previous, "target"])
        rows.append(_frame_rows(frame, current, model, estimator.predict(x.loc[current]), "GLM_NB", feature_name))
    return pd.concat(rows, ignore_index=True)


def _residual_predictions(frame, cfg, evaluation_start):
    groups = feature_groups(frame)
    numeric = frame.select_dtypes(include=[np.number]).columns
    cols = list(dict.fromkeys(c for c in groups["FENO"] if c in numeric and c != "target"))
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    spec = {"family": "RF", "features": "FENO", "hyperparameters": {
        "n_estimators": cfg["random_forest"]["n_estimators"], "max_depth": cfg["random_forest"]["max_depth_grid"][0],
        "min_samples_leaf": cfg["random_forest"]["min_samples_leaf_grid"][0],
        "min_samples_split": cfg["random_forest"]["min_samples_split_grid"][0],
        "max_features": cfg["random_forest"]["max_features"][0], "random_state": cfg["random_forest"]["random_state"]}}
    residual = frame.target - frame.M3_pred_bloque
    rows = []
    for _, current, previous in _origins(frame, cfg, evaluation_start):
        previous_residual = previous & np.isfinite(residual.to_numpy())
        current_residual = current & np.isfinite(frame["M3_pred_bloque"].to_numpy())
        if not previous_residual.any() or not current_residual.any():
            continue
        estimator = estimator_from_spec(spec, cfg)
        estimator.fit(x.loc[previous_residual], residual.loc[previous_residual])
        pred = np.maximum(0, frame.loc[current_residual, "M3_pred_bloque"] +
                          estimator.predict(x.loc[current_residual]))
        rows.append(_frame_rows(frame, current_residual, "RF_RESIDUAL_M3_FENO_ROLLING",
                                pred, "RF_RESIDUAL", "FENO"))
    return pd.concat(rows, ignore_index=True)


def _horizon_rf_predictions(frame, cfg, evaluation_start):
    groups = feature_groups(frame)
    numeric = frame.select_dtypes(include=[np.number]).columns
    cols = list(dict.fromkeys(c for c in groups["FENO"] if c in numeric and c != "target"))
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    spec = {"family": "RF", "features": "FENO", "hyperparameters": {
        "n_estimators": cfg["random_forest"]["n_estimators"], "max_depth": cfg["random_forest"]["max_depth_grid"][0],
        "min_samples_leaf": cfg["random_forest"]["min_samples_leaf_grid"][0],
        "min_samples_split": cfg["random_forest"]["min_samples_split_grid"][0],
        "max_features": cfg["random_forest"]["max_features"][0], "random_state": cfg["random_forest"]["random_state"]}}
    rows = []
    for _, current, previous in _origins(frame, cfg, evaluation_start):
        pred = pd.Series(index=frame.index[current], dtype=float)
        for horizon in range(1, int(cfg["forecast"]["horizon_days"]) + 1):
            current_h = current & frame.horizonte_dia.eq(horizon).to_numpy()
            previous_h = previous & frame.horizonte_dia.eq(horizon).to_numpy()
            if not current_h.any() or not previous_h.any():
                continue
            estimator = estimator_from_spec(spec, cfg)
            estimator.fit(x.loc[previous_h], frame.loc[previous_h, "target"])
            pred.loc[frame.index[current_h]] = estimator.predict(x.loc[current_h])
        rows.append(_frame_rows(frame, current, "RF_H1_H7_FENO_ROLLING", pred.reindex(frame.index[current]).to_numpy(), "RF_HORIZON", "FENO"))
    return pd.concat(rows, ignore_index=True)


def _bayes_predictions(frame, cfg, evaluation_start):
    working = frame.copy()
    origin_dates = pd.to_datetime(working.fecha_origen)
    working["semana_del_año"] = origin_dates.dt.isocalendar().week.astype(float)
    working["periodo_ABRIL_JULIO"] = (origin_dates.dt.month >= 7).astype(float)
    covariates = ["RC_t0", "SS_t0", "AP_t0", "p_RC", "p_SS", "p_AP", "TOTAL_t0", "factor_extrapolacion",
                  "corte_lag_1d", "corte_lag_2d", "corte_lag_3d", "corte_sum_3d", "corte_sum_7d",
                  "corte_sum_14d", "corte_mean_7d", "log_exposure", "horizonte_dia", "semana_del_año", "periodo_ABRIL_JULIO"]
    rows = {"NB_JERARQUICO_ROLLING": [], "NB_JERARQUICO_COVARIABLES_ROLLING": []}
    for _, current, previous in _origins(working, cfg, evaluation_start):
        train, predict = working.loc[previous], working.loc[current]
        nb = HierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"]).fit(train)
        rows["NB_JERARQUICO_ROLLING"].append(_frame_rows(working, current, "NB_JERARQUICO_ROLLING", nb.predict(predict), "BAYES_NB", "POOLING"))
        cov = CovariateHierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"], cfg["bayes"]["covariate_ridge"]).fit(train, covariates)
        rows["NB_JERARQUICO_COVARIABLES_ROLLING"].append(_frame_rows(working, current, "NB_JERARQUICO_COVARIABLES_ROLLING", cov.predict(predict), "BAYES_NB", "COVARIABLES"))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in rows.items()}


def _dirichlet_predictions(frame, intervals, cfg, evaluation_start):
    rows = []
    for origin, current, _ in _origins(frame, cfg, evaluation_start):
        current_frame = frame.loc[current].copy()
        predictions = []
        for position, row in current_frame.iterrows():
            period = _period_for_date(origin, cfg["m3"]["periods"])
            prior = fit_m3(intervals, row.finca, period, origin)
            data = intervals[(intervals.finca == row.finca) & (intervals.periodo == period) & (intervals.fecha <= origin)]
            posterior = DirichletM3(data, prior, cfg["bayes"]["dirichlet_prior_strength"], cfg["bayes"]["seed"] + int(position))
            x0 = np.array([row.RC_t0, row.SS_t0, row.AP_t0], float)
            lead = (pd.Timestamp(row.fecha_objetivo) - origin).days
            draws = []
            for _ in range(cfg["bayes"]["posterior_draws"]):
                q, r, loss = posterior.draw_matrix()
                matrix = M3Matrix(row.finca, period, q, r, loss, prior.audit)
                draws.append(simulate(matrix, x0, lead, cfg["m3"]["baseline_ingress"]).iloc[-1].PC_dia_muestra * row.factor_extrapolacion)
            predictions.append(np.mean(draws))
        rows.append(_frame_rows(frame, current, "M3_DIRICHLET_MULTINOMIAL_ROLLING", predictions, "BAYES_DIRICHLET", "M3"))
    return pd.concat(rows, ignore_index=True)


def _common_observed(left, right):
    """Compatibilidad de pruebas para dos modelos."""
    common = _common_observed_many({"left": left, "right": right})
    return common["left"], common["right"]


def _common_observed_many(predictions):
    """Recorta todos los modelos a exactamente las mismas claves observadas y validas."""
    valid = {}
    common = None
    for name, frame in predictions.items():
        current = frame[frame.real.notna() & frame.proyectado.notna() & frame.estado_ventana.eq("VALIDA")].copy()
        keys = current[KEY_COLUMNS].drop_duplicates()
        common = keys if common is None else common.merge(keys, on=KEY_COLUMNS, how="inner")
        valid[name] = current
    if common is None or common.empty:
        raise ValueError("Los modelos rolling no tienen observaciones comunes completas")
    result = {name: frame.merge(common, on=KEY_COLUMNS, how="inner", validate="many_to_one") for name, frame in valid.items()}
    expected = common.sort_values(KEY_COLUMNS).reset_index(drop=True)
    for name, frame in result.items():
        actual = frame[KEY_COLUMNS].drop_duplicates().sort_values(KEY_COLUMNS).reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"{name} no conserva la poblacion rolling comun")
    return result


def _metrics_by_week(frame):
    rows = []
    keys = ["modelo", "finca", "bloque", "fecha_origen", "semana_proyeccion"]
    for key, group in frame.groupby(keys):
        observed = group[group.real.notna()]
        horizons = set(pd.to_numeric(group.horizonte_dia, errors="coerce").dropna().astype(int))
        dates = pd.to_datetime(group.fecha_objetivo).dropna().sort_values()
        complete = (len(observed) == len(group) == 7 and horizons == set(range(1, 8)) and len(dates) == 7 and
                    (dates.iloc[-1] - dates.iloc[0]).days == 6 and group.estado_ventana.eq("VALIDA").all())
        error = observed.proyectado - observed.real
        denom = observed.real.abs().sum()
        rows.append(dict(zip(keys, key)) | {"estado": "VALIDA" if complete else "PARCIAL",
                     "n_dias_reales": len(observed), "n_dias_pronosticados": len(group),
                     "real_comparable": observed.real.sum(), "proyectado_comparable": observed.proyectado.sum(),
                     "proyectado_total": group.proyectado.sum(), "wape": error.abs().sum() / denom if denom else np.nan,
                     "mae": error.abs().mean(), "rmse": np.sqrt((error ** 2).mean()),
                     "sesgo_pct": error.sum() / denom if denom else np.nan,
                     "ratio_pred_real": observed.proyectado.sum() / observed.real.sum() if observed.real.sum() else np.nan})
    return pd.DataFrame(rows)


def _rf_specs(evaluation, cfg):
    best = pd.read_csv(evaluation / "rf_mejores_por_grupo.csv")
    specs = {}
    for _, row in best.iterrows():
        specs[f"RF_OPT_{row.features}_ROLLING"] = {"model": row.model, "family": "RF", "features": row.features,
                                                      "hyperparameters": json.loads(row.hyperparameters)}
    specs["MODELO_SELECCIONADO_ROLLING"] = read_selection(evaluation / "selected_model_manifest.json")
    return specs


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
    start = holdout_start(windows["fecha_origen"], cfg)
    predictions = {
        "E00_M3_BASE_ROLLING": _m3_predictions(windows, intervals, cfg),
        "E01_M3_INGRESO_CALIBRADO_ROLLING": _m3_predictions(
            windows, intervals, cfg, "E01_M3_INGRESO_CALIBRADO_ROLLING", cfg["m3"]["calibrated_ingress"]),
    }
    for label, spec in _rf_specs(evaluation, cfg).items():
        predictions[label] = _supervised_predictions(frame, cfg, spec, label, start)
    for name in feature_groups(frame):
        predictions[f"GLM_NB_{name}_ROLLING"] = _glm_predictions(frame, cfg, name, f"GLM_NB_{name}_ROLLING", start)
    predictions["RF_RESIDUAL_M3_FENO_ROLLING"] = _residual_predictions(frame, cfg, start)
    predictions["RF_H1_H7_FENO_ROLLING"] = _horizon_rf_predictions(frame, cfg, start)
    predictions.update(_bayes_predictions(frame, cfg, start))
    predictions["M3_DIRICHLET_MULTINOMIAL_ROLLING"] = _dirichlet_predictions(frame, intervals, cfg, start)
    common = _common_observed_many(predictions)
    metadata = []
    for label, data in common.items():
        filename = f"predictions_{label.lower()}.csv"
        data.to_csv(evaluation / filename, index=False)
        observed = data[data.real.notna()]
        metadata.append({"experiment_id": label, "modelo": label, "family": data.familia.iloc[0],
                         "features": data.features.iloc[0], "split": "ROLLING_ORIGIN_COMMON", "causal": True,
                         "nivel": "DIARIO_OBSERVADO", "prediction_file": filename,
                         **metrics(observed.real, observed.proyectado)})
    weekly = pd.concat([_metrics_by_week(data) for data in common.values()], ignore_index=True)
    weekly.to_csv(evaluation / "metrics_rolling_origin_semanal.csv", index=False)
    pd.DataFrame(metadata).to_csv(evaluation / "metrics_rolling_origin.csv", index=False)
    manifest = {"models": list(common), "common_rows": int(len(next(iter(common.values())))),
                "excluded": {"P32_SEMIMARKOV": "RETROSPECTIVO_ORACLE_NO_CAUSAL",
                             "M3_PODA_CLIMA": "coeficientes seleccionados en validacion fija; requiere nested rolling"}}
    (evaluation / "rolling_models_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(weekly.groupby(["modelo", "estado"]).size().to_string())
    print(pd.DataFrame(metadata)[["experiment_id", "wape", "r2", "n"]].sort_values("wape").to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
