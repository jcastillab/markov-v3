"""Entrena y congela RF FENO pooled y RF FENO por horizonte."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from canonical import load_config
    from models.m3 import load_traditional_intervals
    from models.operational import save_operational_models, train_operational_models
    from models.supervised import build_supervised_dataset
except ModuleNotFoundError:
    from src.canonical import load_config
    from src.models.m3 import load_traditional_intervals
    from src.models.operational import save_operational_models, train_operational_models
    from src.models.supervised import build_supervised_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", help="Ultima fecha objetivo permitida, YYYY-MM-DD")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    datasets = root / cfg["paths"]["outputs"] / "datasets"
    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    intervals = load_traditional_intervals(root / cfg["paths"]["raw"], cfg)
    frame = build_supervised_dataset(windows, fact, intervals, cfg)
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    bundle, manifest = train_operational_models(frame, cfg, cutoff)
    registry = root / cfg["operational"]["model_registry"]
    pointer = root / cfg["operational"]["active_model_pointer"]
    artifact, manifest_path = save_operational_models(bundle, manifest, registry, pointer)
    print(f"Filas de entrenamiento: {manifest['training_rows']:,}")
    print(f"Cutoff: {manifest['training_cutoff']}")
    print(f"Features FENO: {len(manifest['features'])}")
    print(f"Artefacto: {artifact}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
