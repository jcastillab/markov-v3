"""Compatibilidad: usa ``scripts/modelos.py entrenar``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    command = [PY, str(ROOT / "scripts" / "modelos.py"), "entrenar"]
    command += sys.argv[1:]
    return __import__("subprocess").run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
