"""Orquestacion compacta del pipeline de modelos.

Las fases historicas siguen siendo modulos internos para preservar trazabilidad
y tests. Este modulo define la interfaz publica que deben usar operadores y
otros repositorios.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


STAGES: dict[str, tuple[str, ...]] = {
    "consolidar": (
        "src/fase0_auditoria.py",
        "src/fase0_contratos.py",
        "src/fase1_pipeline.py",
        "src/fase2_m3.py",
        "src/fase3_p32.py",
        "src/fase4_podas.py",
        "src/fase5_clima.py",
    ),
    "evaluar": (
        "src/fase6_supervisado.py",
        "src/evaluacion_hyperparametros.py",
        "src/fase7_bayes.py",
        "src/evaluacion_rolling.py",
        "src/fase8_comparacion.py",
    ),
    "entrenar": (
        "scripts/train_models.py",
        "src/entrenar_modelos_operacionales.py",
    ),
    "operacional": ("scripts/run_operational.py",),
    "dashboard": (
        "src/evaluacion_entrada.py",
        "scripts/run_operational.py",
    ),
}


def build_stage_commands(stage: str, root: Path, cutoff: str | None = None,
                         extra_args: Iterable[str] = ()) -> list[list[str]]:
    """Construye comandos sin ejecutarlos, facilitando tests y auditoria."""
    if stage not in STAGES:
        raise ValueError(f"Etapa desconocida: {stage}. Opciones: {sorted(STAGES)}")
    commands = []
    for relative in STAGES[stage]:
        command = [sys.executable, str(root / relative)]
        if cutoff and stage == "entrenar":
            command += ["--cutoff", cutoff]
        if stage in {"operacional", "dashboard"}:
            command += list(extra_args)
        commands.append(command)
    return commands


def run_stage(stage: str, root: Path, cutoff: str | None = None,
              extra_args: Iterable[str] = ()) -> int:
    """Ejecuta una etapa y detiene el pipeline ante el primer error."""
    for command in build_stage_commands(stage, root, cutoff, extra_args):
        print(f"\n=== {' '.join(command)} ===", flush=True)
        result = subprocess.run(command, cwd=root)
        if result.returncode != 0:
            print(f"FALLO en {command[-1]}", file=sys.stderr)
            return result.returncode
    return 0
