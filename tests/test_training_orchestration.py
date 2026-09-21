from pathlib import Path

import pytest

from src.training.orchestration import build_stage_commands


def test_public_stages_are_small_and_explicit():
    root = Path("project")
    commands = build_stage_commands("entrenar", root, "2026-08-09")

    assert len(commands) == 2
    assert commands[0][-2:] == ["--cutoff", "2026-08-09"]
    assert commands[1][-2:] == ["--cutoff", "2026-08-09"]


def test_operational_arguments_are_forwarded():
    commands = build_stage_commands("operacional", Path("project"),
                                    extra_args=("--semana", "S36"))

    assert commands == [[
        commands[0][0], str(Path("project") / "scripts" / "run_operational.py"),
        "--semana", "S36"
    ]]


def test_unknown_stage_fails_clearly():
    with pytest.raises(ValueError, match="Etapa desconocida"):
        build_stage_commands("inexistente", Path("project"))


def test_dashboard_rebuilds_external_input_and_operational_run():
    commands = build_stage_commands("dashboard", Path("project"))

    assert commands[0][1].endswith("src\\evaluacion_entrada.py")
    assert commands[1][1].endswith("scripts\\run_operational.py")
