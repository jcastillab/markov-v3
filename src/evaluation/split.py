"""Particiones temporales compartidas por los modelos y diagnosticos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validation_start(values, cfg: dict) -> pd.Timestamp:
    """Devuelve el primer origen de validacion sin partir una fecha entre splits."""
    origins = pd.DatetimeIndex(pd.to_datetime(pd.Series(values).dropna()).unique()).sort_values()
    fraction = float(cfg["evaluation"]["train_fraction"])
    n_train = max(int(cfg["evaluation"]["min_train_windows"]), int(len(origins) * fraction))
    if n_train >= len(origins):
        raise ValueError("No queda una fecha de validacion despues del minimo de entrenamiento")
    return pd.Timestamp(origins[n_train])


def holdout_start(values, cfg: dict) -> pd.Timestamp:
    """Devuelve el primer origen reservado, posterior al periodo de seleccion."""
    origins = pd.DatetimeIndex(pd.to_datetime(pd.Series(values).dropna()).unique()).sort_values()
    train_fraction = float(cfg["evaluation"]["train_fraction"])
    selection_fraction = float(cfg["evaluation"]["selection_fraction"])
    index = max(int(cfg["evaluation"]["min_train_windows"]) + 1,
                int(len(origins) * (train_fraction + selection_fraction)))
    if index >= len(origins):
        raise ValueError("No queda una fecha de holdout despues del periodo de seleccion")
    return pd.Timestamp(origins[index])


def temporal_masks(frame: pd.DataFrame, cfg: dict, require_known_target: bool = True,
                   origin_values=None) -> tuple[np.ndarray, np.ndarray]:
    """Separa por origen y evita entrenar con objetivos observables despues del corte."""
    cutoff = validation_start(origin_values if origin_values is not None else frame["fecha_origen"], cfg)
    origin = pd.to_datetime(frame["fecha_origen"])
    train = origin.lt(cutoff)
    if "fecha_objetivo" in frame:
        train &= pd.to_datetime(frame["fecha_objetivo"]).lt(cutoff)
    valid = origin.ge(cutoff)
    if require_known_target:
        target = "target" if "target" in frame else "corte_real_dia"
        train &= frame[target].notna()
        valid &= frame[target].notna()
    return train.to_numpy(), valid.to_numpy()


def selection_masks(frame: pd.DataFrame, cfg: dict, origin_values=None) -> tuple[np.ndarray, np.ndarray]:
    """Crea ajuste y seleccion sin tocar el holdout final."""
    dates = origin_values if origin_values is not None else frame["fecha_origen"]
    selection_start = validation_start(dates, cfg)
    evaluation_start = holdout_start(dates, cfg)
    origin = pd.to_datetime(frame["fecha_origen"])
    objective = pd.to_datetime(frame["fecha_objetivo"])
    train = origin.lt(selection_start) & objective.lt(selection_start) & frame.target.notna()
    selection = (origin.ge(selection_start) & origin.lt(evaluation_start) &
                 objective.lt(evaluation_start) & frame.target.notna())
    return train.to_numpy(), selection.to_numpy()
