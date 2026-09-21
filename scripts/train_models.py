"""Entrena y congela el mejor modelo de cada familia operacional.

Usa el dataset supervisado historico (targets conocidos) y guarda cada
adaptador como bundle content-addressed bajo outputs/models/operational.
La seleccion por familia se declara en src/modeling/registry.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd

try:
    from canonical import load_config
    from models.m3 import load_traditional_intervals
    from models.supervised import build_supervised_dataset
    from modeling import (
        DirichletAdapter, GLMAdapter, HierarchicalNBAdapter, M3Adapter,
        RFAdapter, RFHorizonAdapter, save_bundle, publish_family_pointers,
    )
except ModuleNotFoundError:
    from src.canonical import load_config
    from src.models.m3 import load_traditional_intervals
    from src.models.supervised import build_supervised_dataset
    from src.modeling import (
        DirichletAdapter, GLMAdapter, HierarchicalNBAdapter, M3Adapter,
        RFAdapter, RFHorizonAdapter, save_bundle, publish_family_pointers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", help="Ultima fecha objetivo permitida, YYYY-MM-DD")
    parser.add_argument("--rf-config", help="JSON con hiperparametros RF; default usa cfg operacional")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    datasets = root / cfg["paths"]["outputs"] / "datasets"
    registry = root / cfg["operational"]["bundle_registry"]
    pointer = root / cfg["operational"]["bundle_pointer"]

    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    intervals = load_traditional_intervals(root / cfg["paths"]["raw"], cfg)
    pruning = pd.read_parquet(datasets / "poda_features.parquet")
    climate = pd.read_parquet(datasets / "clima_features.parquet")
    frame = build_supervised_dataset(windows, fact, intervals, cfg, pruning, climate)
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    frame = frame[frame["target"].notna()].copy()
    if cutoff is not None:
        frame = frame[pd.to_datetime(frame["fecha_objetivo"]) <= cutoff].copy()
    if frame.empty:
        raise ValueError("No hay targets historicos para entrenar")

    rf_hp = cfg["operational"]["models"]["rf_feno"]["hyperparameters"]
    horizon_hp = cfg["operational"]["models"]["rf_h1_h7"]["hyperparameters"]

    adapters = [
        RFAdapter(rf_hp).fit(frame, cfg),
        RFHorizonAdapter(horizon_hp).fit(frame, cfg),
        GLMAdapter(cfg["supervised"]["nb_alpha"], cfg["supervised"]["nb_max_iter"]).fit(frame, cfg),
        HierarchicalNBAdapter(cfg["bayes"]["hierarchical_shrinkage"]).fit(frame, cfg),
        M3Adapter(cfg["m3"]["baseline_ingress"]).fit(frame, cfg),
    ]

    # Dirichlet-Multinomial: posterior sobre las transiciones historicas causales.
    interval_frame = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    if cutoff is not None:
        interval_frame = interval_frame[pd.to_datetime(interval_frame["fecha"]) <= cutoff].copy()
    if not interval_frame.empty:
        dirichlet = DirichletAdapter(
            cfg["bayes"]["dirichlet_prior_strength"], cfg["bayes"]["seed"],
            cfg["m3"]["baseline_ingress"]).fit(interval_frame, cfg)
        adapters.append(dirichlet)

    artifacts = {}
    versions = {}
    for adapter in adapters:
        artifact_path, frozen = save_bundle(adapter, registry, {
            "training_cutoff": (str(cutoff.date()) if cutoff is not None else
                                str(pd.Timestamp(frame["fecha_objetivo"].max()).date())),
            "training_rows": int(len(frame)),
        })
        artifacts[frozen["family"]] = frozen
        versions[frozen["family"]] = frozen["version"]
        print(f"{frozen['family']:16s} {frozen['name']:36s} version={frozen['version']}")

    # Punteros explicitos por familia, sin seleccionar hashes por orden alfabetico.
    publish_family_pointers(registry, versions, pointer)
    print(f"\nPunteros activos por familia -> {pointer}")
    print(f"Familia principal: {artifacts['RF']['name']}")


if __name__ == "__main__":
    main()
