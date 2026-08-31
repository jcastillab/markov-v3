"""Metricas de conteo del proyecto."""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(real: pd.Series, pred: pd.Series) -> dict:
    y = pd.to_numeric(real, errors="coerce").to_numpy(float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    error = p - y
    denom = np.abs(y).sum()
    wape = float(np.abs(error).sum() / denom) if denom else np.nan
    total_ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / total_ss) if len(y) > 1 and total_ss else np.nan
    return {"n": int(len(y)), "wape": wape,
            "mae": float(np.abs(error).mean()) if len(y) else np.nan,
            "rmse": float(np.sqrt(np.mean(error ** 2))) if len(y) else np.nan,
            "bias_pct": float(error.sum() / denom) if denom else np.nan, "r2": r2,
            "acierto_global": 1 - wape if np.isfinite(wape) else np.nan}
