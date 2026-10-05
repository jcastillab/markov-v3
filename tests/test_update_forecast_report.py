import numpy as np
import pandas as pd
import sys
import types
import zipfile

from scripts.package_portable import build_zip
from scripts.update_forecast_report import _build_reports, _update_excel_tables


def test_report_keeps_future_week_without_inventing_real_values():
    weekly = pd.DataFrame([{
        "modelo": "RF_OPT_FENO_ENTRADA",
        "finca": "ALMER",
        "bloque": "1",
        "fecha_origen": "2026-09-21",
        "semana_proyeccion": 202640,
        "real": np.nan,
        "proyectado": 1234.5,
        "dias": 7,
        "dias_reales": 0,
        "estado_evaluacion": "NO EVALUABLE",
    }])

    farm, block = _build_reports(weekly)

    assert farm.loc[0, "semana_proyeccion"] == 202640
    assert block.loc[0, "semana_proyeccion"] == 202640
    assert farm.loc[0, "estado_evaluacion"] == "NO EVALUABLE"
    assert block.loc[0, "estado_evaluacion"] == "NO EVALUABLE"
    assert pd.isna(farm.loc[0, "real"])
    assert pd.isna(block.loc[0, "real"])
    assert farm.loc[0, "proyectado"] == 1234.5
    assert farm.loc[0, "indicador"] == "SIN REAL"


def test_report_table_update_uses_python_excel_automation(monkeypatch, tmp_path):
    class Range:
        Value2 = None

    class PivotTables:
        Count = 0

    class Sheet:
        def __init__(self):
            self.ListObjects = types.SimpleNamespace(
                Item=lambda _name: types.SimpleNamespace(
                    ListColumns=types.SimpleNamespace(Count=1), Resize=lambda _range: None
                )
            )

        @staticmethod
        def Cells(*_args):
            return object()

        @staticmethod
        def Range(*_args):
            return Range()

        @staticmethod
        def PivotTables():
            return PivotTables()

    sheets = {name: Sheet() for name in (
        "COMPARACION_SEMANAL", "COMPARACION_SEMANAL_BLOQUE"
    )}

    class Worksheets:
        def Item(self, name):
            return sheets[name]

        def __iter__(self):
            return iter(sheets.values())

    workbook = types.SimpleNamespace(
        Worksheets=Worksheets(), Save=lambda: None,
        Close=lambda _save: setattr(workbook, "closed", True),
    )
    excel = types.SimpleNamespace(Workbooks=types.SimpleNamespace(
        Open=lambda _path: workbook
    ), Quit=lambda: setattr(excel, "quit", True))
    client = types.ModuleType("win32com.client")
    client.DispatchEx = lambda _name: excel
    package = types.ModuleType("win32com")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "win32com", package)
    monkeypatch.setitem(sys.modules, "win32com.client", client)

    _update_excel_tables(tmp_path / "report.xlsx", {
        "COMPARACION_SEMANAL": pd.DataFrame({"dato": [1, 2]}),
        "COMPARACION_SEMANAL_BLOQUE": pd.DataFrame({"dato": [3]}),
    })

    assert workbook.closed
    assert excel.quit


def test_portable_package_does_not_contain_powershell_files(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "entry.py").write_text("print('ok')", encoding="utf-8")
    output = tmp_path / "portable.zip"

    build_zip(project, output)

    with zipfile.ZipFile(output) as package:
        names = package.namelist()
        assert not any(name.lower().endswith((".ps1", ".psm1", ".psd1")) for name in names)
        readme = package.read("markov_v3/LEEME_PRIMERO.txt").decode("utf-8")
        assert "powershell" not in readme.lower()
