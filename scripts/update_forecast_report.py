"""Actualiza las tablas fuente del informe Excel conservando pivots y segmentadores.

La actualizacion usa la automatizacion COM de Excel desde Python (Excel debe
estar instalado en Windows). El archivo plantilla original no se modifica; el
informe vigente se escribe en ``outputs/informe pronosticos.xlsx``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT / "pronosticos_reentrenados_S17_S38_bloque.xlsx"
MODELS = {
    "RF_H1_H7_FENO_ENTRADA",
    "RF_OPT_FENO_ENTRADA",
    "GLM_NB_FENO_CLIMA_ENTRADA",
    "NB_JERARQUICO_COVARIABLES_ENTRADA",
}


TABLE_MAPPINGS = (
    ("COMPARACION_SEMANAL", "Table1"),
    ("COMPARACION_SEMANAL_BLOQUE", "Table2"),
)


def _update_excel_tables(excel_path: Path, sources: dict[str, pd.DataFrame]) -> None:
    """Actualiza las tablas y pivots via Excel COM, sin generar scripts de shell."""
    try:
        from win32com.client import DispatchEx
    except ImportError as exc:
        raise RuntimeError(
            "La actualizacion del informe requiere Excel y pywin32. "
            "Instala las dependencias con: python -m pip install -r requirements.txt"
        ) from exc

    excel = target = None
    saved = False
    try:
        excel = DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        target = excel.Workbooks.Open(str(excel_path.resolve()))
        for sheet_name, table_name in TABLE_MAPPINGS:
            frame = sources[sheet_name]
            sheet = target.Worksheets.Item(sheet_name)
            table = sheet.ListObjects.Item(table_name)
            row_count, column_count = len(frame) + 1, len(frame.columns)
            if column_count != table.ListColumns.Count:
                raise ValueError(
                    f"Columnas incompatibles en {sheet_name}: "
                    f"fuente={column_count} tabla={table.ListColumns.Count}"
                )
            table.Resize(sheet.Range(sheet.Cells(1, 1), sheet.Cells(row_count, column_count)))
            if len(frame):
                # Excel COM requiere un escalar para una celda y una matriz
                # bidimensional para rangos de varias celdas.
                values = tuple(
                    tuple(
                        None if pd.isna(value) else
                        value.item() if hasattr(value, "item") else value
                        for value in row
                    )
                    for row in frame.itertuples(index=False, name=None)
                )
                destination = sheet.Range(
                    sheet.Cells(2, 1), sheet.Cells(row_count, column_count)
                )
                destination.Value2 = values[0][0] if len(values) == 1 and column_count == 1 else values
        for sheet in target.Worksheets:
            pivots = sheet.PivotTables()
            for index in range(1, pivots.Count + 1):
                pivots.Item(index).RefreshTable()
        target.Save()
        saved = True
    finally:
        if target is not None:
            target.Close(saved)
        if excel is not None:
            excel.Quit()


def _build_reports(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce las dos formas tabulares que alimentan las tablas dinámicas."""
    if {"modelo", "finca", "bloque", "fecha_origen", "semana_proyeccion",
        "real", "proyectado"}.difference(weekly.columns):
        raise ValueError("El CSV semanal de entrada no contiene las columnas requeridas")
    weekly = weekly[weekly["modelo"].isin(MODELS)].copy()
    weekly["semana_proyeccion"] = pd.to_numeric(
        weekly["semana_proyeccion"], errors="coerce").astype("Int64")
    weekly["fecha_origen"] = pd.to_datetime(weekly["fecha_origen"], errors="coerce")
    if weekly.empty:
        raise ValueError("No hay filas de los modelos del informe en la evaluación actual")
    # Los informes establecidos seleccionan el origen representativo por
    # modelo/bloque/semana; las filas ya vienen agregadas y se evalúan aquí.
    from src.evaluacion_entrada import _status, _select_weekly_entry_origins
    selected = _select_weekly_entry_origins(weekly)
    selected["diferencia"] = selected["proyectado"] - selected["real"]
    selected["razon_proyectado_real"] = selected["proyectado"] / selected["real"].where(selected["real"].ne(0))
    selected["desviacion_pct"] = selected["razon_proyectado_real"] - 1
    selected["indicador"] = selected["razon_proyectado_real"].map(_status)
    selected.loc[selected["estado_evaluacion"].eq("NO EVALUABLE"),
                 ["real", "diferencia", "desviacion_pct", "indicador"]] = [pd.NA, pd.NA, pd.NA, "SIN REAL"]
    selected["modelo_nombre"] = selected["modelo"].map({
        "RF_H1_H7_FENO_ENTRADA": "RF H1-H7 FENO",
        "RF_OPT_FENO_ENTRADA": "RF global OPT",
        "GLM_NB_FENO_CLIMA_ENTRADA": "GLM NB FENO + clima",
        "NB_JERARQUICO_COVARIABLES_ENTRADA": "Bayes NB - covariables",
    })
    block = selected[["modelo", "modelo_nombre", "finca", "bloque", "semana_proyeccion",
                      "estado_evaluacion", "real", "proyectado", "diferencia",
                      "desviacion_pct", "indicador"]].copy()

    farm = (selected.groupby(["modelo", "finca", "semana_proyeccion"], as_index=False,
                             dropna=False)
            .agg(real=("real", "sum"), proyectado=("proyectado", "sum"),
                 dias=("dias", "sum"), dias_reales=("dias_reales", "sum")))
    farm["estado_evaluacion"] = "NO EVALUABLE"
    farm.loc[farm.dias_reales.eq(farm.dias), "estado_evaluacion"] = "COMPLETA"
    farm.loc[farm.dias_reales.gt(0) & farm.dias_reales.lt(farm.dias), "estado_evaluacion"] = "PARCIAL"
    farm["diferencia"] = farm["proyectado"] - farm["real"]
    farm["razon_proyectado_real"] = farm["proyectado"] / farm["real"].where(farm["real"].ne(0))
    farm["desviacion_pct"] = farm["razon_proyectado_real"] - 1
    farm["indicador"] = farm["razon_proyectado_real"].map(_status)
    farm.loc[farm.estado_evaluacion.eq("NO EVALUABLE"),
             ["real", "diferencia", "desviacion_pct", "indicador"]] = [pd.NA, pd.NA, pd.NA, "SIN REAL"]
    farm["modelo_nombre"] = farm["modelo"].map(selected.drop_duplicates("modelo").set_index("modelo")["modelo_nombre"])
    farm = farm[["modelo", "modelo_nombre", "finca", "semana_proyeccion", "estado_evaluacion",
                 "real", "proyectado", "diferencia", "desviacion_pct", "indicador"]]
    return farm, block


def run(root: Path = ROOT) -> Path:
    from src.canonical import load_config, pipeline_config_path

    cfg = load_config(pipeline_config_path(root))
    template = root / TEMPLATE.relative_to(ROOT)
    profile_outputs = root / cfg["paths"]["outputs"]
    weekly_csv = profile_outputs / "evaluation" / "predictions_entrada_semanales.csv"
    output = profile_outputs / "informe pronosticos.xlsx"
    if not template.is_file():
        raise FileNotFoundError(f"No se encontró el libro plantilla: {template}")
    if not weekly_csv.is_file():
        raise FileNotFoundError(f"No se encontró evaluación del dashboard: {weekly_csv}")
    weekly = pd.read_csv(weekly_csv, parse_dates=["fecha_origen"])
    farm, block = _build_reports(weekly)
    maximum_week = int(pd.to_numeric(block["semana_proyeccion"]).max())
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, output)
    _update_excel_tables(output, {
        "COMPARACION_SEMANAL": farm,
        "COMPARACION_SEMANAL_BLOQUE": block,
    })
    print(f"Informe actualizado: {output}")
    print(f"Última semana incluida: {maximum_week}")
    print(f"Filas por finca: {len(farm)}; por finca-bloque: {len(block)}")
    return output


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
