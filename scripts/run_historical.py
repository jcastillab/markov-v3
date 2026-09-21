"""Compatibilidad: usa la interfaz publica ``scripts/modelos.py todo``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    command = [sys.executable, str(ROOT / "scripts" / "modelos.py"), "todo"]
    command += sys.argv[1:]
    return __import__("subprocess").run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
