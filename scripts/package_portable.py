"""Genera un ZIP portable con todo lo necesario para levantar el dashboard.

Incluye codigo, configuracion, datos de entrada y artefactos de ``outputs``
(modelos, datasets, evaluaciones y corridas) para que el tablero funcione en
otro equipo sin reentrenar. Excluye el entorno virtual, el historial de git y
los archivos comprimidos auxiliares.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "markov_v3"

INCLUDE_DIRS = [
    "config",
    "src",
    "scripts",
    "tests",
    "docs",
    "data/raw",
    "data/external",
    "data/vision",
    "outputs",
]
INCLUDE_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "AGENTS.md",
    ".gitignore",
]
EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".lock", ".zip")
EXCLUDE_PREFIXES = ("~$",)

LEEME = """\
MARKOV V3 - PAQUETE PORTABLE DEL DASHBOARD
==========================================

Requisitos en el equipo nuevo
-----------------------------
1. Windows de 64 bits.
2. Python 3.12 (https://www.python.org/downloads/), marcando la casilla
   "Add python.exe to PATH" durante la instalacion.

Instalacion (una sola vez)
--------------------------
Abre una terminal en esta carpeta (la que contiene este archivo) y ejecuta:

    py -3.12 -m venv .venv
    .\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
    .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

Levantar el tablero
-------------------
    .\\.venv\\Scripts\\python.exe scripts\\run_dashboard.py

Luego abre en el navegador: http://localhost:8501

Notas
-----
- El primer arranque reconstruye el cache de conteos de vision (unos minutos);
  los siguientes tardan ~30 s si no cambio ningun dato.
- NO copies ni reutilices la carpeta .venv de otro equipo.
- Para actualizar datos: reemplaza data/raw/auxiliares/Roses.xlsx, agrega la
  semana nueva en data/vision/Resultados/ y vuelve a correr run_dashboard.py.
"""


def _excluded(path: Path) -> bool:
    if path.name in EXCLUDE_DIR_NAMES:
        return True
    if path.name.endswith(EXCLUDE_SUFFIXES):
        return True
    if any(path.name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return True
    return False


def collect_entries(root: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for relative in INCLUDE_DIRS:
        base = root / relative
        if not base.is_dir():
            continue
        for current, dirs, files in os.walk(base):
            dirs[:] = [name for name in dirs if name not in EXCLUDE_DIR_NAMES]
            current_path = Path(current)
            for name in files:
                path = current_path / name
                if _excluded(path):
                    continue
                entries.append((path, f"{PACKAGE_ROOT}/{path.relative_to(root).as_posix()}"))
    for relative in INCLUDE_FILES:
        path = root / relative
        if path.is_file() and not _excluded(path):
            entries.append((path, f"{PACKAGE_ROOT}/{relative}"))
    return entries


def build_zip(root: Path, output: Path) -> tuple[int, int]:
    entries = collect_entries(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, arcname in entries:
            archive.write(path, arcname)
        archive.writestr(f"{PACKAGE_ROOT}/LEEME_PRIMERO.txt", LEEME)
    return len(entries) + 1, output.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Ruta del ZIP (por defecto markov_v3_portable.zip en la raiz)")
    args = parser.parse_args()
    output = args.output or (ROOT / "markov_v3_portable.zip")
    print("Empaquetando proyecto portable...")
    count, size = build_zip(ROOT, output)
    print(f"Archivos: {count:,}")
    print(f"ZIP: {output} ({size / (1024 * 1024):.1f} MB)")
    print(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
