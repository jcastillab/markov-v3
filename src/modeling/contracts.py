"""Contrato minimo y portable para estimadores operacionales.

Cada adaptador implementa la interfaz ``ModelAdapter`` y puede trasladarse a
otro proyecto sin depender de este repositorio. El scoring opera sobre un
``pd.DataFrame`` con las features ordenadas que el adaptador declara en
``features``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class ModelAdapter(Protocol):
    """Contrato de un modelo listo para scoring operacional."""

    family: str
    name: str
    features: list[str]

    def fit(self, train: pd.DataFrame, cfg: dict) -> "ModelAdapter":
        """Ajusta el modelo con el dataset historico (target conocido)."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predice el total del bloque por fila (no negativo)."""

    def metadata(self) -> dict:
        """Metadata declarativa del modelo para el manifest."""


def nonnegative(values: np.ndarray) -> np.ndarray:
    """Recorta a cero: las salidas son conteos esperados, no pueden ser negativos."""
    return np.maximum(0.0, np.asarray(values, dtype=float))


def select_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Valida y ordena las features requeridas; faltantes se bloquean con error."""
    missing = [c for c in features if c not in frame]
    if missing:
        raise ValueError(f"Faltan features para scoring: {missing}")
    return frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_target(frame: pd.DataFrame) -> np.ndarray:
    """Valida el target historico y devuelve conteos finitos no negativos."""
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(float)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("target debe contener conteos finitos no negativos")
    return target