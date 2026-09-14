"""Adaptadores RandomForest: pooled y por horizonte."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..contracts import nonnegative, select_features, train_target


def _feno_features(frame: pd.DataFrame) -> list[str]:
    """Features FENO reproducibles del dataset supervisado (orden estable)."""
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns]
    feno = [c for c in numeric if c.endswith("_t0") or c.startswith(
        ("p_", "RC_", "SS_", "log1p_", "M3_", "horizonte"))]
    scale = [c for c in numeric if c in ("camas_activas", "camas_muestreadas",
                                         "factor_extrapolacion", "cobertura_muestreo")]
    history = [c for c in numeric if c.startswith("corte_")]
    return list(dict.fromkeys(feno + scale + history))


class RFAdapter:
    """Random Forest pooled sobre features FENO."""

    family = "RF"
    name = "RF_OPT_FENO_OPERATIONAL"

    def __init__(self, hyperparameters: dict | None = None):
        self.hyperparameters = dict(hyperparameters or {})
        self.hyperparameters.setdefault("n_estimators", 200)
        self.hyperparameters.setdefault("max_depth", 10)
        self.hyperparameters.setdefault("min_samples_leaf", 2)
        self.hyperparameters.setdefault("min_samples_split", 5)
        self.hyperparameters.setdefault("max_features", 0.5)
        self.hyperparameters.setdefault("criterion", "squared_error")
        self.hyperparameters.setdefault("random_state", 42)
        self.hyperparameters.setdefault("n_jobs", 1)
        self.features: list[str] = []
        self._estimator: RandomForestRegressor | None = None

    def fit(self, train: pd.DataFrame, cfg: dict | None = None) -> "RFAdapter":
        self.features = _feno_features(train)
        if not self.features:
            raise ValueError("No hay features FENO en el frame de entrenamiento")
        x = select_features(train, self.features)
        y = train_target(train)
        self._estimator = RandomForestRegressor(**self.hyperparameters).fit(x, y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("RFAdapter no esta ajustado")
        return nonnegative(self._estimator.predict(select_features(frame, self.features)))

    def metadata(self) -> dict:
        return {"features": self.features, "hyperparameters": self.hyperparameters}


class RFHorizonAdapter:
    """Siete Random Forest independientes, uno por horizonte H1-H7."""

    family = "RF_HORIZON"
    name = "RF_H1_H7_FENO_OPERATIONAL"

    def __init__(self, hyperparameters: dict | None = None):
        self.hyperparameters = dict(hyperparameters or {})
        self.hyperparameters.setdefault("n_estimators", 500)
        self.hyperparameters.setdefault("max_depth", 3)
        self.hyperparameters.setdefault("min_samples_leaf", 2)
        self.hyperparameters.setdefault("min_samples_split", 5)
        self.hyperparameters.setdefault("max_features", "sqrt")
        self.hyperparameters.setdefault("criterion", "squared_error")
        self.hyperparameters.setdefault("random_state", 42)
        self.hyperparameters.setdefault("n_jobs", 1)
        self.features: list[str] = []
        self._estimators: dict[int, RandomForestRegressor] = {}

    def fit(self, train: pd.DataFrame, cfg: dict | None = None) -> "RFHorizonAdapter":
        self.features = _feno_features(train)
        if not self.features:
            raise ValueError("No hay features FENO en el frame de entrenamiento")
        x = select_features(train, self.features)
        y = train_target(train)
        horizon_values = sorted(pd.to_numeric(train["horizonte_dia"], errors="coerce").dropna().astype(int).unique())
        if not horizon_values:
            raise ValueError("No hay horizonte_dia en el frame de entrenamiento")
        for horizon in horizon_values:
            mask = train["horizonte_dia"].eq(horizon).to_numpy()
            if not mask.any():
                continue
            estimator = RandomForestRegressor(**self.hyperparameters).fit(x[mask], y[mask])
            self._estimators[horizon] = estimator
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise RuntimeError("RFHorizonAdapter no esta ajustado")
        x = select_features(frame, self.features)
        out = np.full(len(frame), np.nan)
        for horizon, estimator in self._estimators.items():
            mask = frame["horizonte_dia"].eq(horizon).to_numpy()
            out[mask] = estimator.predict(x[mask])
        if np.isnan(out).any():
            raise ValueError("Hay filas sin horizonte cubierto por el modelo")
        return nonnegative(out)

    def metadata(self) -> dict:
        return {"features": self.features, "horizons": sorted(self._estimators),
                "hyperparameters": self.hyperparameters}