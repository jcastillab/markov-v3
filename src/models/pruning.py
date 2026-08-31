"""Features y ajustes causales de poda para Fase 4."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

try:
    from models.m3 import EVENTS, M3Matrix, STATES
except ModuleNotFoundError:  # import as src.models.pruning in pytest
    from src.models.m3 import EVENTS, M3Matrix, STATES


def build_pruning_features(pruning: pd.DataFrame, origins: pd.DataFrame,
                           cfg: dict) -> pd.DataFrame:
    """Construye señales conocidas en t0, sin consultar fechas posteriores."""
    p = pruning.copy()
    p["fecha"] = pd.to_datetime(p["fecha"]).dt.normalize()
    lags = list(cfg["pruning"]["priority_lags_days"])
    windows = list(cfg["pruning"]["response_window_days"])
    rows = []
    for _, origin in origins.iterrows():
        date = pd.Timestamp(origin["fecha_origen"]).normalize()
        subset = p[(p["finca"] == origin["finca"]) &
                   (p["bloque"] == origin["bloque"]) &
                   (p["fecha"] <= date)]
        row = {"finca": origin["finca"], "bloque": origin["bloque"],
               "fecha_origen": date}
        for kind in ("poda_alineamiento", "poda_corte", "poda_total"):
            values = subset.set_index("fecha")[kind]
            for lag in lags:
                center = date - pd.Timedelta(days=lag)
                row[f"{kind}_lag_{lag}d"] = float(values.get(center, 0.0))
                for width in windows:
                    start = center - pd.Timedelta(days=width // 2)
                    end = center + pd.Timedelta(days=width - width // 2 - 1)
                    row[f"{kind}_sum_{width}d_lag_{lag}d"] = float(
                        subset.loc[subset["fecha"].between(start, end), kind].sum())
            kernel_start = date - pd.Timedelta(
                days=7 * cfg["pruning"]["kernel_window_weeks"][1])
            kernel_end = date - pd.Timedelta(
                days=7 * cfg["pruning"]["kernel_window_weeks"][0])
            row[f"{kind}_kernel_8_12w"] = float(
                subset.loc[subset["fecha"].between(kernel_start, kernel_end), kind].sum())
            for width in (28, 56, 84):
                start = date - pd.Timedelta(days=width - 1)
                row[f"{kind}_acumulada_{width}d"] = float(
                    subset.loc[subset["fecha"].between(start, date), kind].sum())
        for kind, name in (("poda_total", "poda"), ("poda_alineamiento", "alineamiento")):
            dates = subset.loc[subset[kind] > 0, "fecha"]
            row[f"dias_desde_ultima_{name}"] = ((date - dates.max()).days
                                                   if len(dates) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def pruning_signal(features: pd.DataFrame, cfg: dict) -> pd.Series:
    col = "poda_total_kernel_8_12w"
    scale = float(cfg["pruning"]["effect_scale"])
    return features[col].fillna(0.0).clip(lower=0.0) / scale


def adjust_matrix(matrix: M3Matrix, signal: float, coefficient: float,
                  transition: bool) -> M3Matrix:
    """Aumenta solo la transición de avance/corte y conserva cada columna."""
    if not transition or coefficient == 0 or signal <= 0:
        return matrix
    factor = 1.0 + coefficient * signal
    q, r, loss = matrix.Q.copy(), matrix.r.copy(), matrix.p.copy()
    for source in STATES:
        j = STATES.index(source)
        destination = {"RC": "SS", "SS": "AP", "AP": "PC"}[source]
        old = q[STATES.index(destination), j] if destination in STATES else r[j]
        new = min(old * factor, old + q[j, j], 1.0 - loss[j])
        delta = new - old
        if destination in STATES:
            q[STATES.index(destination), j] = new
        else:
            r[j] = new
        q[j, j] = max(0.0, q[j, j] - delta)
    return replace(matrix, Q=q, r=r)
