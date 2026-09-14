"""Adaptador M3 mecanistico (baseline obligatorio).

M3 no se entrena con targets; se ajusta con las trayectorias fenologicas
historicas causales. El adaptador conserva las matrices ajustadas por
finca+periodo y simula cada fila hasta su lead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import nonnegative


class M3Adapter:
    """Baseline Markov. Requiere intervalos fenologicos y config M3."""

    family = "M3"
    name = "E00_M3_BASE_OPERATIONAL"

    def __init__(self, alpha_ingress: float = 0.20):
        self.alpha_ingress = alpha_ingress
        self.features: list[str] = []
        self._intervals: pd.DataFrame | None = None
        self._cfg: dict | None = None
        self._periods: list[dict] = []

    def fit(self, train: pd.DataFrame, cfg: dict | None = None) -> "M3Adapter":
        """Guarda el contexto; M3 no consume targets del frame de entrenamiento."""
        self._cfg = cfg or {}
        self._periods = list(self._cfg.get("m3", {}).get("periods", []))
        return self

    def _load_intervals(self, cfg: dict):
        if self._intervals is not None:
            return self._intervals
        from src.models.m3 import load_traditional_intervals

        raw = __import__("pathlib").Path(cfg["paths"]["raw"])
        self._intervals = load_traditional_intervals(raw, cfg)
        return self._intervals

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._cfg:
            raise RuntimeError("M3Adapter no tiene configuracion")
        from src.models.m3 import _period_for_date, fit_m3, simulate

        intervals = self._load_intervals(self._cfg)
        periods = self._cfg["m3"]["periods"]
        cache = {}
        predictions = []
        for row in frame.itertuples(index=False):
            origin = pd.Timestamp(row.fecha_origen)
            period = _period_for_date(origin, periods)
            key = (row.finca, period, origin)
            if key not in cache:
                cache[key] = fit_m3(intervals, row.finca, period, origin)
            matrix = cache[key]
            x0 = np.array([row.RC_t0, row.SS_t0, row.AP_t0], dtype=float)
            lead = int((pd.Timestamp(row.fecha_objetivo) - origin).days)
            factor = float(getattr(row, "factor_extrapolacion", 1.0))
            value = simulate(matrix, x0, lead, self.alpha_ingress).iloc[-1].PC_dia_muestra * factor
            predictions.append(value)
        return nonnegative(np.asarray(predictions))

    def metadata(self) -> dict:
        return {"alpha_ingress": self.alpha_ingress, "periods": self._periods}