"""Pipeline completo de produccion: historico -> modelos -> proyeccion -> validacion."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

DEFAULT = [
    "scripts/run_historical.py",
    "scripts/train_models_cmd.py",
    "scripts/run_operational.py",
    "scripts/validate_run.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-historical", action="store_true",
                        help="Omite fases historicas y de seleccion")
    args = parser.parse_args()

    steps = DEFAULT if not args.skip_historical else DEFAULT[1:]
    for step in steps:
        print(f"\n{'='*70}\n== {step}\n{'='*70}")
        result = subprocess.run([PY, str(ROOT / step)], cwd=ROOT)
        if result.returncode != 0:
            print(f"FALLO en {step}", file=sys.stderr)
            return result.returncode
    print("\nPipeline productivo completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())