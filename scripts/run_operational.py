"""Ejecuta la proyeccion operacional sobre la ultima carpeta SNN disponible.

Cada corrida genera exclusivamente la semana ISO siguiente e inmutable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    command = [PY, str(ROOT / "src/proyeccion_vision.py")]
    if len(sys.argv) > 1:
        command += sys.argv[1:]
    print(f"=== {' '.join(command)} ===")
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())