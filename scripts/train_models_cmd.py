"""Entrena y congela los modelos operacionales (mejor modelo por familia)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    steps = [
        [PY, str(ROOT / "scripts/train_models.py")],
        [PY, str(ROOT / "src/entrenar_modelos_operacionales.py")],
    ]
    for command in steps:
        print(f"\n=== {' '.join(command[-1:])} ===")
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())