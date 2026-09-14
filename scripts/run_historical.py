"""Orquesta el pipeline historico completo: fases 0-8 y seleccion de modelos."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    "src/fase0_auditoria.py",
    "src/fase0_contratos.py",
    "src/fase1_pipeline.py",
    "src/fase2_m3.py",
    "src/fase3_p32.py",
    "src/fase4_podas.py",
    "src/fase5_clima.py",
    "src/fase6_supervisado.py",
    "src/evaluacion_hyperparametros.py",
    "src/fase7_bayes.py",
    "src/evaluacion_rolling.py",
    "src/fase8_comparacion.py",
]


def main() -> int:
    for step in STEPS:
        print(f"\n=== {step} ===")
        result = subprocess.run([PY, str(ROOT / step)], cwd=ROOT)
        if result.returncode != 0:
            print(f"FALLO en {step}", file=sys.stderr)
            return result.returncode
    print("\nPipeline historico completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())