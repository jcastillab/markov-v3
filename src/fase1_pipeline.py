"""CLI de construccion de datasets canonicos de Fase 1."""

from __future__ import annotations

from pathlib import Path

from canonical import (build_dimensions, build_fact_bloque_dia,
                       build_forecast_windows, build_qa_join_coverage,
                       load_config)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    raw = root / cfg["paths"]["raw"]
    out = root / cfg["paths"]["outputs"] / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    fact = build_fact_bloque_dia(cfg, raw)
    windows = build_forecast_windows(fact, cfg["forecast"]["horizon_days"])
    dimensions = build_dimensions(cfg, raw, fact)
    fact.to_parquet(out / "fact_bloque_dia.parquet", index=False)
    windows.to_parquet(out / "forecast_windows.parquet", index=False)
    for name, frame in dimensions.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    qa = build_qa_join_coverage(fact, windows)
    qa.to_csv(root / "outputs" / "data_quality" / "qa_join_coverage.csv", index=False)
    print(f"fact_bloque_dia: {len(fact):,} filas")
    print(f"forecast_windows: {len(windows):,} filas")
    print(f"ventanas evaluables: {windows.groupby(['finca','bloque','fecha_origen']).ventana_evaluable.first().sum():,}")


if __name__ == "__main__":
    main()
