"""Dataset supervisado y modelos de conteo para Fase 6."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

try:
    from models.m3 import _period_for_date, fit_m3, simulate
except ModuleNotFoundError:
    from src.models.m3 import _period_for_date, fit_m3, simulate


def _history(fact, finca, bloque, date):
    h = fact[(fact.finca == finca) & (fact.bloque == bloque) & (fact.fecha <= date)]
    h = h.sort_values("fecha").set_index("fecha")["corte_comercial_real"]
    calendar = pd.date_range(h.index.min(), date, freq="D") if len(h) else pd.DatetimeIndex([])
    h = h.reindex(calendar)
    row = {}
    for lag in (1, 2, 3, 7, 14):
        value = h.get(date - pd.Timedelta(days=lag), np.nan)
        row[f"corte_lag_{lag}d"] = float(value) if pd.notna(value) else np.nan
        row[f"corte_lag_{lag}d_observado"] = float(pd.notna(value))
    for width in (3, 7, 14, 28):
        values = h.tail(width)
        row[f"corte_sum_{width}d"] = float(values.sum(min_count=1)) if values.notna().any() else np.nan
        row[f"corte_dias_observados_{width}d"] = float(values.notna().sum())
        row[f"corte_dias_faltantes_{width}d"] = float(values.isna().sum())
        if width in (7, 28):
            row[f"corte_mean_{width}d"] = float(values.mean()) if values.notna().any() else np.nan
    return row


def build_supervised_dataset(windows, fact, intervals, cfg, pruning=None, climate=None,
                             include_incomplete=False):
    """Una fila por ventana diaria, con historiales limitados a t0."""
    rows = []
    cache = {}
    for _, w in windows.iterrows():
        if not include_incomplete and not w.ventana_evaluable:
            continue
        key = (w.finca, w.bloque, w.fecha_origen)
        if key not in cache:
            period = _period_for_date(pd.Timestamp(w.fecha_origen), cfg["m3"]["periods"])
            matrix = fit_m3(intervals, w.finca, period, pd.Timestamp(w.fecha_origen))
            x0 = np.array([w.conteo_RC_t0, w.conteo_SS_t0, w.conteo_AP_t0], float)
            lead = (pd.Timestamp(w.fecha_objetivo) - pd.Timestamp(w.fecha_origen)).days
            cache[key] = float(simulate(matrix, x0, lead, cfg["m3"]["baseline_ingress"]).iloc[-1].PC_dia_muestra)
        row = {"finca": w.finca, "bloque": w.bloque, "fecha_origen": w.fecha_origen,
               "fecha_objetivo": w.fecha_objetivo, "semana_proyeccion": w.semana_proyeccion,
               "horizonte_dia": w.horizonte_dia, "estado_ventana": w.estado_ventana,
               "semana_objetivo": w.semana_objetivo, "target": w.corte_real_dia,
               "RC_t0": w.conteo_RC_t0, "SS_t0": w.conteo_SS_t0, "AP_t0": w.conteo_AP_t0,
               "CO_t0": w.conteo_CO_t0, "TOTAL_t0": w.conteo_total_t0,
               "p_RC": w.conteo_RC_t0 / max(w.conteo_total_t0, 1),
               "p_SS": w.conteo_SS_t0 / max(w.conteo_total_t0, 1),
               "p_AP": w.conteo_AP_t0 / max(w.conteo_total_t0, 1),
               "p_CO": w.conteo_CO_t0 / max(w.conteo_total_t0, 1),
               "RC_AP_ratio": w.conteo_RC_t0 / max(w.conteo_AP_t0, 1),
               "SS_AP_ratio": w.conteo_SS_t0 / max(w.conteo_AP_t0, 1),
               "log1p_RC": np.log1p(w.conteo_RC_t0), "log1p_SS": np.log1p(w.conteo_SS_t0),
               "log1p_AP": np.log1p(w.conteo_AP_t0), "camas_activas": w.camas_activas_t0,
               "camas_muestreadas": w.camas_muestreadas_t0,
               "factor_extrapolacion": w.factor_extrapolacion_t0,
               "cobertura_muestreo": w.camas_muestreadas_t0 / max(w.camas_activas_t0, 1),
               "M3_pred_muestra": cache[key],
               "M3_pred_bloque": cache[key] * w.factor_extrapolacion_t0}
        row.update(_history(fact, w.finca, w.bloque, pd.Timestamp(w.fecha_origen)))
        if pruning is not None:
            p = pruning[(pruning.finca == w.finca) & (pruning.bloque == w.bloque) &
                        (pruning.fecha_origen == w.fecha_origen)]
            if len(p):
                row.update(p.iloc[0].drop(labels=["finca", "bloque", "fecha_origen"]).to_dict())
        if climate is not None:
            c = climate[(climate.finca == w.finca) & (climate.fecha_origen == w.fecha_origen)]
            if len(c):
                row.update(c.iloc[0].drop(labels=["finca", "bloque", "fecha_origen"]).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


class NegativeBinomialGLM:
    """GLM NB2: Var(Y)=mu+alpha*mu^2, con enlace log."""

    def __init__(self, alpha=1.0, max_iter=300):
        self.alpha, self.max_iter = alpha, max_iter

    def fit(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        self.mean_ = np.nanmean(x, axis=0); self.scale_ = np.nanstd(x, axis=0)
        self.scale_[self.scale_ == 0] = 1
        z = (np.nan_to_num(x, nan=0.0) - self.mean_) / self.scale_
        z = np.column_stack([np.ones(len(z)), z])
        def objective(beta):
            eta = np.clip(z @ beta, -20, 20); mu = np.exp(eta); a = self.alpha
            likelihood = np.sum(y * np.log(mu / (1 + a * mu)) - np.log1p(a * mu) / a)
            return -likelihood + 0.01 * np.sum(beta[1:] ** 2)
        initial = np.zeros(z.shape[1]); initial[0] = np.log(max(y.mean(), 1e-6))
        self.coef_ = minimize(objective, initial, method="L-BFGS-B",
                              bounds=[(-10, 20)] * z.shape[1],
                              options={"maxiter": self.max_iter}).x
        return self

    def predict(self, x):
        z = (np.nan_to_num(np.asarray(x, float), nan=0.0) - self.mean_) / self.scale_
        z = np.column_stack([np.ones(len(z)), z])
        return np.exp(np.clip(z @ self.coef_, -20, 20))


def feature_groups(frame):
    all_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    feno = [c for c in all_cols if c.endswith("_t0") or c.startswith(("p_", "RC_", "SS_", "log1p_", "M3_", "horizonte"))]
    poda = [c for c in all_cols if "poda" in c or "alineamiento" in c]
    clima = [c for c in all_cols if any(x in c for x in ("gdd", "DLI", "vpd", "temp_", "rain_", "et0_"))]
    scale = [c for c in all_cols if c in ("camas_activas", "camas_muestreadas", "factor_extrapolacion", "cobertura_muestreo")]
    history = [c for c in all_cols if c.startswith("corte_")]
    return {"FENO": feno + scale + history, "FENO_PODA": feno + scale + history + poda,
            "FENO_CLIMA": feno + scale + history + clima,
            "FENO_PODA_CLIMA": feno + scale + history + poda + clima}
