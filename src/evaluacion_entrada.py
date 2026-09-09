"""Evalua los modelos existentes sobre un archivo externo de conteos.

La entrada se usa como poblacion de scoring. Los estimadores directos se
reajustan con el dataset historico disponible hasta cada origen; nunca usan
el ``Cantidad`` de la ventana que estan pronosticando.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

try:
    from canonical import build_forecast_windows, load_config
    from evaluation.metrics import metrics
    from models.bayes import CovariateHierarchicalNB, DirichletM3, HierarchicalNB
    from models.m3 import M3Matrix, _period_for_date, fit_m3, load_traditional_intervals, simulate
    from models.selection import estimator_from_spec
    from models.supervised import NegativeBinomialGLM, build_supervised_dataset, feature_groups
except ModuleNotFoundError:
    from src.canonical import build_forecast_windows, load_config
    from src.evaluation.metrics import metrics
    from src.models.bayes import CovariateHierarchicalNB, DirichletM3, HierarchicalNB
    from src.models.m3 import M3Matrix, _period_for_date, fit_m3, load_traditional_intervals, simulate
    from src.models.selection import estimator_from_spec
    from src.models.supervised import NegativeBinomialGLM, build_supervised_dataset, feature_groups


MODEL_NAMES = (
    "E00_M3_BASE_ENTRADA",
    "E01_M3_INGRESO_CALIBRADO_ENTRADA",
    "RF_OPT_FENO_ENTRADA",
    "RF_OPT_FENO_PODA_ENTRADA",
    "RF_OPT_FENO_CLIMA_ENTRADA",
    "RF_OPT_FENO_PODA_CLIMA_ENTRADA",
    "GLM_NB_FENO_ENTRADA",
    "GLM_NB_FENO_PODA_ENTRADA",
    "GLM_NB_FENO_CLIMA_ENTRADA",
    "GLM_NB_FENO_PODA_CLIMA_ENTRADA",
    "RF_RESIDUAL_M3_FENO_ENTRADA",
    "RF_H1_H7_FENO_ENTRADA",
    "NB_JERARQUICO_ENTRADA",
    "NB_JERARQUICO_COVARIABLES_ENTRADA",
    "M3_DIRICHLET_MULTINOMIAL_ENTRADA",
)


def _read_input(path: Path, cfg: dict, template: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_excel(path)
    required = {"Finca", "Bloque", "Fecha", "Cantidad", "semana", "conteo_RC",
                "conteo_SS", "conteo_AP", "conteo_CO", "conteo_total"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Faltan columnas en la entrada: {sorted(missing)}")
    aliases = cfg["farm_aliases"]
    out = pd.DataFrame({
        "finca": source["Finca"].map(lambda value: _canonical_farm(value, aliases)),
        "bloque": source["Bloque"].map(_canonical_block),
        "fecha": pd.to_datetime(source["Fecha"], dayfirst=True, errors="coerce").dt.normalize(),
        "semana_iso": pd.to_numeric(source["semana"], errors="coerce").astype("Int64"),
        "corte_comercial_real": pd.to_numeric(source["Cantidad"], errors="coerce"),
        "conteo_RC": pd.to_numeric(source["conteo_RC"], errors="coerce"),
        "conteo_SS": pd.to_numeric(source["conteo_SS"], errors="coerce"),
        "conteo_AP": pd.to_numeric(source["conteo_AP"], errors="coerce"),
        "conteo_CO": pd.to_numeric(source["conteo_CO"], errors="coerce"),
        "conteo_total": pd.to_numeric(source["conteo_total"], errors="coerce"),
    })
    out = out[out.finca.notna() & out.fecha.notna()].copy()
    if out.duplicated(["finca", "bloque", "fecha"]).any():
        raise ValueError("La entrada tiene claves finca+bloque+fecha duplicadas")
    out["es_fecha_conteo"] = out["conteo_total"].notna()

    # Las dimensiones estaticas y las features auxiliares ya auditadas se
    # reutilizan por clave; los conteos y el real siempre vienen de la entrada.
    aux = [c for c in template.columns if c not in {
        "finca", "bloque", "fecha", "semana_iso", "corte_comercial_real",
        "conteo_RC", "conteo_SS", "conteo_AP", "conteo_CO", "conteo_total",
        "es_fecha_conteo", "fecha_conteo_origen", "dias_desde_conteo",
        "conteo_RC_origen", "conteo_SS_origen", "conteo_AP_origen",
        "conteo_CO_origen", "conteo_total_origen", "semana_objetivo",
    }]
    lookup = template[["finca", "bloque", "fecha"] + aux].drop_duplicates(
        ["finca", "bloque", "fecha"])
    out = out.merge(lookup, on=["finca", "bloque", "fecha"], how="left")
    defaults = {
        "camas_activas": 1.0, "camas_muestreadas": 1.0,
        "factor_extrapolacion": 1.0, "cobertura_muestreo": 1.0,
        "plantas_activas": 0.0, "area_activa": 0.0,
        "poda_alineamiento": 0.0, "poda_corte": 0.0, "poda_total": 0.0,
    }
    for col in aux:
        if col in defaults:
            out[col] = out[col].fillna(defaults[col])
    origins = (out[out["es_fecha_conteo"]].sort_values("fecha")
               .groupby(["finca", "bloque", "semana_iso"], as_index=False).tail(1)
               [["finca", "bloque", "semana_iso", "fecha", "conteo_RC", "conteo_SS",
                 "conteo_AP", "conteo_CO", "conteo_total"]]
               .rename(columns={"fecha": "fecha_conteo_origen",
                                "conteo_RC": "conteo_RC_origen", "conteo_SS": "conteo_SS_origen",
                                "conteo_AP": "conteo_AP_origen", "conteo_CO": "conteo_CO_origen",
                                "conteo_total": "conteo_total_origen"}))
    out = out.merge(origins, on=["finca", "bloque", "semana_iso"], how="left", validate="many_to_one")
    out["dias_desde_conteo"] = (out["fecha"] - out["fecha_conteo_origen"]).dt.days
    out["semana_objetivo"] = out["semana_iso"] + 1
    return out


def _canonical_farm(value: object, aliases: dict) -> str | None:
    if pd.isna(value):
        return None
    key = " ".join(str(value).strip().upper().split())
    normalized = {" ".join(str(k).strip().upper().split()): v for k, v in aliases.items()}
    return normalized.get(key)


def _canonical_block(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _base_rows(frame: pd.DataFrame, mask: np.ndarray, model: str, pred) -> pd.DataFrame:
    cols = ["finca", "bloque", "fecha_origen", "fecha_objetivo", "semana_proyeccion",
            "horizonte_dia", "estado_ventana", "target"]
    out = frame.loc[mask, cols].rename(columns={"target": "real"}).copy()
    out["modelo"] = model
    out["proyectado"] = np.asarray(pred, float)
    return out


def _m3(frame: pd.DataFrame, intervals: pd.DataFrame, cfg: dict, alpha: float, name: str) -> pd.DataFrame:
    rows = []
    matrices = {}
    for _, row in frame.iterrows():
        origin = pd.Timestamp(row.fecha_origen)
        period = _period_for_date(origin, cfg["m3"]["periods"])
        key = (row.finca, period, origin)
        if key not in matrices:
            matrices[key] = fit_m3(intervals, row.finca, period, origin)
        matrix = matrices[key]
        x0 = np.array([row.RC_t0, row.SS_t0, row.AP_t0], float)
        lead = (pd.Timestamp(row.fecha_objetivo) - origin).days
        pred = simulate(matrix, x0, lead, alpha).iloc[-1].PC_dia_muestra * row.factor_extrapolacion
        result = _base_rows(frame, frame.index == row.name, name, [pred])
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _direct_models(frame: pd.DataFrame, historical: pd.DataFrame, cfg: dict,
                   rf_specs: dict[str, dict]) -> dict[str, pd.DataFrame]:
    groups = feature_groups(frame)
    outputs = {}
    numeric = frame.select_dtypes(include=[np.number]).columns
    origins = sorted(pd.to_datetime(frame.fecha_origen).unique())
    first_origin = next((origin for origin in origins
                         if ((pd.to_datetime(historical.fecha_objetivo) < origin) & historical.target.notna()).any()), None)
    historical = historical[(pd.to_datetime(historical.fecha_objetivo) < first_origin) & historical.target.notna()] if first_origin is not None else historical.iloc[0:0]
    if historical.empty:
        raise ValueError("No hay historico anterior al primer origen de la entrada")
    for name, group_name, estimator_factory in [
        (f"GLM_NB_{group}_ENTRADA", group, lambda: NegativeBinomialGLM(
            cfg["supervised"]["nb_alpha"], cfg["supervised"]["nb_max_iter"]))
        for group in groups
    ]:
        cols = list(dict.fromkeys(c for c in groups[group_name] if c in numeric and c != "target"))
        x_new = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        estimator = estimator_factory().fit(
            historical[cols].replace([np.inf, -np.inf], np.nan).fillna(0), historical.target)
        eligible = pd.to_datetime(frame.fecha_origen).ge(first_origin).to_numpy()
        outputs[name] = _base_rows(frame, eligible, name, estimator.predict(x_new.loc[eligible]))

    for name, spec in rf_specs.items():
        group_name = spec["features"]
        cols = list(dict.fromkeys(c for c in groups[group_name] if c in numeric and c != "target"))
        x_new = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        estimator = estimator_from_spec(spec, cfg).fit(
            historical[cols].replace([np.inf, -np.inf], np.nan).fillna(0), historical.target)
        eligible = pd.to_datetime(frame.fecha_origen).ge(first_origin).to_numpy()
        outputs[name] = _base_rows(frame, eligible, name, estimator.predict(x_new.loc[eligible]))
    return outputs


def _additional_rf_models(frame: pd.DataFrame, historical: pd.DataFrame, cfg: dict,
                          spec: dict) -> dict[str, pd.DataFrame]:
    groups = feature_groups(frame)
    cols = list(dict.fromkeys(c for c in groups["FENO"]
                              if c in frame.select_dtypes(include=[np.number]).columns and c != "target"))
    origins = sorted(pd.to_datetime(frame.fecha_origen).unique())
    first_origin = next((origin for origin in origins
                         if ((pd.to_datetime(historical.fecha_objetivo) < origin) & historical.target.notna()).any()), None)
    train = historical[(pd.to_datetime(historical.fecha_objetivo) < first_origin) & historical.target.notna()].copy() if first_origin is not None else historical.iloc[0:0]
    if train.empty:
        return {"RF_RESIDUAL_M3_FENO_ENTRADA": pd.DataFrame(), "RF_H1_H7_FENO_ENTRADA": pd.DataFrame()}
    eligible = pd.to_datetime(frame.fecha_origen).ge(first_origin).to_numpy()
    x_train = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    x_new = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    residual_estimator = estimator_from_spec(spec, cfg).fit(
        x_train, train.target - train.M3_pred_bloque)
    residual_pred = np.maximum(0, frame.M3_pred_bloque.to_numpy() + residual_estimator.predict(x_new))
    residual = _base_rows(frame, eligible, "RF_RESIDUAL_M3_FENO_ENTRADA", residual_pred[eligible])
    horizon_rows = []
    for horizon in sorted(pd.to_numeric(train.horizonte_dia).dropna().unique()):
        train_h = train[train.horizonte_dia.eq(horizon)]
        current_h = eligible & frame.horizonte_dia.eq(horizon).to_numpy()
        if train_h.empty or not current_h.any():
            continue
        estimator = estimator_from_spec(spec, cfg).fit(
            train_h[cols].replace([np.inf, -np.inf], np.nan).fillna(0), train_h.target)
        horizon_rows.append(_base_rows(frame, current_h, "RF_H1_H7_FENO_ENTRADA",
                                       estimator.predict(x_new.loc[current_h])))
    return {"RF_RESIDUAL_M3_FENO_ENTRADA": residual,
            "RF_H1_H7_FENO_ENTRADA": pd.concat(horizon_rows, ignore_index=True) if horizon_rows else pd.DataFrame()}


def _bayes_models(frame: pd.DataFrame, historical: pd.DataFrame, intervals: pd.DataFrame,
                  cfg: dict) -> dict[str, pd.DataFrame]:
    outputs = {"NB_JERARQUICO_ENTRADA": [], "NB_JERARQUICO_COVARIABLES_ENTRADA": [],
               "M3_DIRICHLET_MULTINOMIAL_ENTRADA": []}
    covariates = ["RC_t0", "SS_t0", "AP_t0", "p_RC", "p_SS", "p_AP", "TOTAL_t0",
                  "factor_extrapolacion", "corte_lag_1d", "corte_lag_2d", "corte_lag_3d",
                  "corte_sum_3d", "corte_sum_7d", "corte_sum_14d", "corte_mean_7d",
                  "horizonte_dia"]
    origins = sorted(pd.to_datetime(frame.fecha_origen).unique())
    first_origin = next((origin for origin in origins
                         if ((pd.to_datetime(historical.fecha_objetivo) < origin) & historical.target.notna()).any()), None)
    train = historical[(pd.to_datetime(historical.fecha_objetivo) < first_origin) & historical.target.notna()].copy() if first_origin is not None else historical.iloc[0:0]
    if train.empty:
        raise ValueError("No hay historico anterior al primer origen de la entrada")
    nb = HierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"]).fit(train)
    cov = CovariateHierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"], cfg["bayes"]["covariate_ridge"])
    cov.fit(train, [c for c in covariates if c in train.columns])
    for origin in sorted(pd.to_datetime(frame.fecha_origen).unique()):
        if origin < first_origin:
            continue
        current = pd.to_datetime(frame.fecha_origen).eq(origin).to_numpy()
        current_frame = frame.loc[current]
        outputs["NB_JERARQUICO_ENTRADA"].append(_base_rows(
            frame, current, "NB_JERARQUICO_ENTRADA", nb.predict(current_frame)))
        outputs["NB_JERARQUICO_COVARIABLES_ENTRADA"].append(_base_rows(
            frame, current, "NB_JERARQUICO_COVARIABLES_ENTRADA", cov.predict(current_frame)))
        predictions = []
        for _, row in current_frame.iterrows():
            period = _period_for_date(origin, cfg["m3"]["periods"])
            prior = fit_m3(intervals, row.finca, period, origin)
            data = intervals[(intervals.finca == row.finca) & (intervals.periodo == period) & (intervals.fecha <= origin)]
            posterior = DirichletM3(data, prior, cfg["bayes"]["dirichlet_prior_strength"], cfg["bayes"]["seed"])
            x0 = np.array([row.RC_t0, row.SS_t0, row.AP_t0], float)
            lead = (pd.Timestamp(row.fecha_objetivo) - origin).days
            draws = []
            for _ in range(cfg["bayes"]["posterior_draws"]):
                q, r, loss = posterior.draw_matrix()
                matrix = M3Matrix(row.finca, period, q, r, loss, prior.audit)
                draws.append(simulate(matrix, x0, lead, cfg["m3"]["baseline_ingress"]).iloc[-1].PC_dia_muestra * row.factor_extrapolacion)
            predictions.append(np.mean(draws))
        outputs["M3_DIRICHLET_MULTINOMIAL_ENTRADA"].append(_base_rows(
            frame, current, "M3_DIRICHLET_MULTINOMIAL_ENTRADA", predictions))
    return {key: pd.concat(value, ignore_index=True) if value else pd.DataFrame()
            for key, value in outputs.items()}


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    result = (daily.groupby(["modelo", "finca", "bloque", "fecha_origen", "semana_proyeccion"], as_index=False)
              .agg(real=("real", "sum"), proyectado=("proyectado", "sum"), dias=("fecha_objetivo", "nunique")))
    result["diferencia"] = result.proyectado - result.real
    result["error_abs"] = result.diferencia.abs()
    result["razon_proyectado_real"] = np.where(result.real.ne(0), result.proyectado / result.real, np.nan)
    result["desviacion_pct"] = result.razon_proyectado_real - 1
    result["indicador"] = result.razon_proyectado_real.map(_status)
    return result


def _status(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "SIN REAL"
    if 0.93 <= value <= 1.07:
        return "ACIERTO"
    if 0.90 <= value < 0.93 or 1.07 < value <= 1.10:
        return "CERCA"
    return "NO ACIERTO"


def _write_sheet(workbook: Workbook, name: str, frame: pd.DataFrame) -> None:
    sheet = workbook.create_sheet(name)
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        sheet.append(list(row))
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(max(1, len(frame.columns)))}{len(frame) + 1}"
    for index, column in enumerate(frame.columns, 1):
        sheet.column_dimensions[get_column_letter(index)].width = min(28, max(12, len(str(column)) + 2))
        if column in {"razon_proyectado_real", "desviacion_pct"}:
            for cell in sheet[get_column_letter(index)][1:]:
                cell.number_format = "0.0%"
    if "indicador" in frame.columns:
        column = get_column_letter(list(frame.columns).index("indicador") + 1)
        rng = f"{column}2:{column}{len(frame) + 1}"
        for label, color in (("ACIERTO", "C6EFCE"), ("CERCA", "FFEB9C"), ("NO ACIERTO", "FFC7CE")):
            sheet.conditional_formatting.add(rng, CellIsRule(
                operator="equal", formula=[f'"{label}"'], fill=PatternFill("solid", fgColor=color)))


def evaluate_input(root: Path, input_path: Path) -> tuple[Path, Path]:
    cfg = load_config(root / "config/pipeline.yaml")
    out = root / cfg["paths"]["outputs"]
    datasets, evaluation = out / "datasets", out / "evaluation"
    template = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    external_fact = _read_input(input_path, cfg, template)
    windows = build_forecast_windows(external_fact, int(cfg["forecast"]["horizon_days"]))
    intervals = load_traditional_intervals(root / cfg["paths"]["raw"], cfg)
    pruning = pd.read_parquet(datasets / "poda_features.parquet")
    climate = pd.read_parquet(datasets / "clima_features.parquet")
    frame = build_supervised_dataset(windows, external_fact, intervals, cfg, pruning, climate, include_incomplete=True)
    historical = pd.read_parquet(datasets / "dataset_supervisado_diario.parquet")
    predictions = {
        "E00_M3_BASE_ENTRADA": _m3(frame, intervals, cfg, cfg["m3"]["baseline_ingress"], "E00_M3_BASE_ENTRADA"),
        "E01_M3_INGRESO_CALIBRADO_ENTRADA": _m3(frame, intervals, cfg, cfg["m3"]["calibrated_ingress"], "E01_M3_INGRESO_CALIBRADO_ENTRADA"),
    }
    best = pd.read_csv(evaluation / "rf_mejores_por_grupo.csv")
    rf_specs = {f"RF_OPT_{row.features}_ENTRADA": {"family": "RF", "features": row.features,
        "hyperparameters": json.loads(row.hyperparameters)} for _, row in best.iterrows()}
    predictions.update(_direct_models(frame, historical, cfg, rf_specs))
    feno_spec = rf_specs.get("RF_OPT_FENO_ENTRADA")
    if feno_spec:
        predictions.update(_additional_rf_models(frame, historical, cfg, feno_spec))
    predictions.update(_bayes_models(frame, historical, intervals, cfg))
    daily = pd.concat([value.assign(fuente_entrada=input_path.name) for value in predictions.values() if not value.empty], ignore_index=True)
    weekly = _weekly(daily)
    evaluation.mkdir(parents=True, exist_ok=True)
    daily_path = evaluation / "predictions_entrada_diarias.csv"
    weekly_path = evaluation / "predictions_entrada_semanales.csv"
    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_sheet(workbook, "SEMANAL", weekly)
    _write_sheet(workbook, "DIARIO", daily)
    workbook.save(evaluation / "evaluacion_entrada.xlsx")
    daily.to_csv(daily_path, index=False)
    weekly.to_csv(weekly_path, index=False)
    return daily_path, weekly_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "resultados acutuales/conteos_vs_cortes_multifinca.xlsx"
    print("\n".join(map(str, evaluate_input(root, input_path))))
