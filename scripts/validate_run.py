"""Valida la corrida operacional mas reciente y la integridad de los bundles."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from src.canonical import load_config  # noqa: E402
from src.dashboard_validation import load_latest_operational_run  # noqa: E402
from src.modeling import FAMILY_ORDER, load_bundle  # noqa: E402


def main() -> int:
    cfg = load_config(ROOT / "config/pipeline.yaml")
    daily, weekly, manifest = load_latest_operational_run(
        ROOT, cfg["operational"]["runs_path"])
    errors = []

    if daily.empty or weekly.empty:
        errors.append("No hay corrida operacional valida")
    else:
        target_week = manifest.get("semana_objetivo")
        weeks = set(weekly["semana_proyeccion"].unique())
        if weeks != {target_week}:
            errors.append(f"La corrida no proyecta solo {target_week}: {weeks}")
        if daily["proyectado"].isna().any():
            errors.append("Hay predicciones nulas")
        if (daily["proyectado"] < 0).any():
            errors.append("Hay predicciones negativas")
        global_sum = weekly["proyectado"].sum()
        block_sum = daily["proyectado"].sum()
        if abs(global_sum - block_sum) > 1e-6:
            errors.append(f"Suma global {global_sum} != suma de bloques {block_sum}")

    registry = ROOT / cfg["operational"]["bundle_registry"]
    for family in FAMILY_ORDER:
        versions = {}
        if registry.is_dir():
            for version_dir in registry.iterdir():
                mf = version_dir / "manifest.json"
                if mf.is_file():
                    import json
                    fam = json.loads(mf.read_text(encoding="utf-8")).get("family")
                    if fam == family:
                        versions[family] = version_dir.name
        if family in versions:
            load_bundle(registry, versions[family])
            print(f"OK bundle {family}: {versions[family]}")
        else:
            print(f"AVISO bundle {family}: no presente")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"\nOK corrida {manifest.get('run_id')} -> semana {target_week} "
          f"({len(weekly)} semanales, {len(daily)} diarias)")
    return 0


if __name__ == "__main__":
    sys.exit(main())