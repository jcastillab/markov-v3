"""Contrato minimo para desacoplar estimadores del proyecto anfitrion."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Regressor(Protocol):
    """Interfaz compatible con los modelos actuales y futuros."""

    def fit(self, x, y): ...

    def predict(self, x) -> np.ndarray: ...


def nonnegative_predictions(model: Regressor, x) -> np.ndarray:
    """Normaliza la salida operacional comun a todos los challengers."""
    return np.maximum(0.0, np.asarray(model.predict(x), dtype=float))
