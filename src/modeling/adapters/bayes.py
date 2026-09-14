"""Adaptadores Bayesianos: NB jerarquico y Dirichlet-Multinomial M3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import nonnegative


class HierarchicalNBAdapter:
    """Gamma-Poisson con pooling parcial finca -> bloque -> horizonte."""

    family = "BAYES_NB"
    name = "NB_JERARQUICO_OPERATIONAL"

    def __init__(self, shrinkage: float = 10.0):
        self.shrinkage = shrinkage
        self.features: list[str] = []
        self._stats = None

    def fit(self, train: pd.DataFrame, cfg: dict | None = None) -> "HierarchicalNBAdapter":
        from src.models.bayes import HierarchicalNB

        if train.empty:
            raise ValueError("No se puede ajustar HierarchicalNB con un frame vacio")
        self._stats = HierarchicalNB(self.shrinkage).fit(train)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("HierarchicalNBAdapter no esta ajustado")
        return nonnegative(self._stats.predict(frame))

    def metadata(self) -> dict:
        return {"shrinkage": self.shrinkage, "features": self.features}


class DirichletAdapter:
    """Posterior Dirichlet-Multinomial sobre las transiciones M3.

    El fit fija los parametros alpha del posterior por estado. El predict usa la
    matriz posterior-media para simular la trayectoria de cada fila. Solo es
    causal si el posterior se construyo con intervalos anteriores al origen.
    """

    family = "BAYES_DIRICHLET"
    name = "M3_DIRICHLET_MULTINOMIAL_OPERATIONAL"

    def __init__(self, prior_strength: float = 5.0, seed: int = 42, alpha_ingress: float = 0.20):
        self.prior_strength, self.seed, self.alpha_ingress = prior_strength, seed, alpha_ingress
        self.features: list[str] = []
        self._alphas: dict[str, np.ndarray] | None = None
        self._events: dict[str, list[str]] | None = None
        self._states: list[str] = ["RC", "SS", "AP"]

    def fit(self, train: pd.DataFrame, cfg: dict | None = None) -> "DirichletAdapter":
        from src.models.bayes import DirichletM3

        if "estado_origen" not in train and cfg is None:
            # Sin intervalos historicos no se puede entrenar Dirichlet.
            raise ValueError("DirichletAdapter requiere intervalos de transicion")
        prior = None
        if cfg is not None:
            from src.models.m3 import M3Matrix, fit_m3, load_traditional_intervals

            raw = __import__("pathlib").Path(cfg["paths"]["raw"]) if False else None
        # Soporte principal: entrenar desde un frame de intervalos (estado_origen, evento).
        self._posterior = DirichletM3(
            train, prior if prior is not None else _identity_prior(self._states), self.prior_strength, self.seed)
        self._alphas = self._posterior.alpha
        self._events = self._posterior.events
        return self

    def _matrix_from_alphas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, r, loss = np.zeros((3, 3)), np.zeros(3), np.zeros(3)
        for source in self._states:
            j = self._states.index(source)
            values = np.asarray(self._alphas[source], float)
            values = values / values.sum()
            if source == "AP":
                q[j, j], r[j], loss[j] = values
            else:
                q[j, j], advance, loss[j] = values
                destination = "SS" if source == "RC" else "AP"
                q[self._states.index(destination), j] = advance
        return q, r, loss

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        from src.models.m3 import M3Matrix, simulate

        if self._alphas is None:
            raise RuntimeError("DirichletAdapter no esta ajustado")
        q, r, loss = self._matrix_from_alphas()
        predictions = []
        for row in frame.itertuples(index=False):
            matrix = M3Matrix("", "", q, r, loss, pd.DataFrame())
            x0 = np.array([row.RC_t0, row.SS_t0, row.AP_t0], dtype=float)
            lead = int((pd.Timestamp(row.fecha_objetivo) - pd.Timestamp(row.fecha_origen)).days)
            factor = float(getattr(row, "factor_extrapolacion", 1.0))
            value = simulate(matrix, x0, lead, self.alpha_ingress).iloc[-1].PC_dia_muestra * factor
            predictions.append(value)
        return nonnegative(np.asarray(predictions))

    def metadata(self) -> dict:
        return {"prior_strength": self.prior_strength, "seed": self.seed,
                "alpha_ingress": self.alpha_ingress, "states": self._states}


def _identity_prior(states: list[str]) -> np.ndarray:
    """Prior neutro M3 cuando no se dispone de una matriz previa."""
    from src.models.m3 import M3Matrix

    # Prior por defecto: progresion lenta entre estados.
    q = np.eye(3) * 0.9
    q[1, 0] = 0.1
    q[2, 1] = 0.1
    q[2, 2] = 0.9
    return M3Matrix("", "", q, np.zeros(3), np.zeros(3), pd.DataFrame())