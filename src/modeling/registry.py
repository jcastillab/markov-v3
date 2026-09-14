"""Registro declarativo de modelos operacionales por familia.

La seleccion por familia se basa en el ranking rolling causal
(``outputs/evaluation/ranking_final.csv``). El registro centraliza el
mapeo familia -> adaptador para poder ampliarlo en otro proyecto.
"""

from __future__ import annotations

from .adapters import (
    DirichletAdapter,
    GLMAdapter,
    HierarchicalNBAdapter,
    M3Adapter,
    RFAdapter,
    RFHorizonAdapter,
)

# familia -> (adaptador, rol)
OPERATIONAL_MODELS: dict[str, tuple[type, str]] = {
    "M3": (M3Adapter, "baseline_obligatorio"),
    "RF": (RFAdapter, "champion_provisional"),
    "RF_HORIZON": (RFHorizonAdapter, "challenger"),
    "GLM_NB": (GLMAdapter, "challenger"),
    "BAYES_NB": (HierarchicalNBAdapter, "challenger"),
    "BAYES_DIRICHLET": (DirichletAdapter, "challenger"),
}

FAMILY_ORDER = ["M3", "RF", "RF_HORIZON", "GLM_NB", "BAYES_NB", "BAYES_DIRICHLET"]


def best_by_family(ranking_path) -> dict[str, str]:
    """Devuelve {familia: nombre_modelo} a partir del ranking rolling causal.

    Usa el mejor WAPE de cada familia excluyendo retrospectivos y baselines
    cuando hay un challenger causal mejor en la misma familia.
    """
    import json

    import pandas as pd

    from pathlib import Path

    path = Path(ranking_path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "family" not in frame:
        return {}
    causal = frame[frame.get("causal", True).fillna(True).astype(bool)].copy()
    best: dict[str, str] = {}
    for family in FAMILY_ORDER:
        subset = causal[causal["family"].eq(family)]
        if subset.empty:
            continue
        subset = subset.sort_values("wape")
        best[family] = str(subset.iloc[0]["experiment_id"])
    return best