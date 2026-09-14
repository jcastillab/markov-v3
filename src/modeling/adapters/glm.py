"""Adaptador GLM Negative Binomial (NB2) sobre features FENO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import nonnegative, select_features, train_target
from .rf import _feno_features


class GLMAdapter:
    """GLM NB2: Var(Y)=mu+alpha*mu^2, enlace log, features FENO."""

    family = "GLM_NB"
    name = "GLM_NB_FENO_OPERATIONAL"

    def __init__(self, alpha: float = 1.0, max_iter: int = 300):
        self.alpha_param, self.max_iter = alpha, max_iter
        self.features: list[str] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._coef: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, cfg: dict | None = None) -> "GLMAdapter":
        self.features = _feno_features(train)
        if not self.features:
            raise ValueError("No hay features FENO en el frame de entrenamiento")
        x = select_features(train, self.features).to_numpy(float)
        y = train_target(train)
        self._mean = np.nanmean(x, axis=0)
        self._scale = np.nanstd(x, axis=0)
        self._scale[self._scale == 0] = 1.0
        z = (np.nan_to_num(x, nan=0.0) - self._mean) / self._scale
        z = np.column_stack([np.ones(len(z)), z])

        def objective(beta):
            eta = np.clip(z @ beta, -20, 20)
            mu = np.exp(eta)
            a = self.alpha_param
            likelihood = np.sum(y * np.log(mu / (1 + a * mu)) - np.log1p(a * mu) / a)
            return -likelihood + 0.01 * np.sum(beta[1:] ** 2)

        from scipy.optimize import minimize

        initial = np.zeros(z.shape[1])
        initial[0] = np.log(max(y.mean(), 1e-6))
        self._coef = minimize(objective, initial, method="L-BFGS-B",
                              bounds=[(-10, 20)] * z.shape[1],
                              options={"maxiter": self.max_iter}).x
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("GLMAdapter no esta ajustado")
        x = select_features(frame, self.features).to_numpy(float)
        z = (np.nan_to_num(x, nan=0.0) - self._mean) / self._scale
        z = np.column_stack([np.ones(len(z)), z])
        return nonnegative(np.exp(np.clip(z @ self._coef, -20, 20)))

    def metadata(self) -> dict:
        return {"features": self.features, "nb_alpha": self.alpha_param,
                "max_iter": self.max_iter}