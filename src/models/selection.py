"""Contrato para persistir y reconstruir modelos supervisados seleccionados."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor


def estimator_from_spec(spec: dict, cfg: dict):
    params = dict(spec["hyperparameters"])
    params.setdefault("random_state", cfg["random_forest"]["random_state"])
    family = spec["family"]
    if family == "RF":
        params.setdefault("n_jobs", int(cfg["random_forest"].get("parallel", {}).get("model_n_jobs", 1)))
        return RandomForestRegressor(**params)
    if family == "EXTRA_TREES":
        params.setdefault("n_jobs", int(cfg["random_forest"].get("parallel", {}).get("model_n_jobs", 1)))
        return ExtraTreesRegressor(**params)
    if family == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingRegressor(**params)
    raise ValueError(f"Familia de modelo no soportada: {family}")


def read_selection(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. Ejecute src/evaluacion_hyperparametros.py antes del rolling."
        )
    return json.loads(path.read_text(encoding="utf-8"))
