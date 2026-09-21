"""Interfaz publica para preparar, evaluar, entrenar y operar modelos.

Ejemplos:
    python scripts/modelos.py consolidar
    python scripts/modelos.py evaluar
    python scripts/modelos.py entrenar --cutoff 2026-08-09
    python scripts/modelos.py operacional -- --semana S36
    python scripts/modelos.py dashboard
    python scripts/modelos.py todo --cutoff 2026-08-09
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import run_stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("consolidar", "evaluar", "entrenar",
                                           "operacional", "dashboard", "todo"))
    parser.add_argument("--cutoff", help="Ultima fecha objetivo para entrenar")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="Argumentos para operacional despues de '--'")
    args = parser.parse_args()

    if args.stage == "todo":
        for stage in ("consolidar", "evaluar", "entrenar"):
            code = run_stage(stage, ROOT, args.cutoff)
            if code:
                return code
        return 0

    extra = args.args[1:] if args.args and args.args[0] == "--" else args.args
    return run_stage(args.stage, ROOT, args.cutoff, extra)


if __name__ == "__main__":
    raise SystemExit(main())
