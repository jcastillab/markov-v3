"""Preparacion de poblaciones de validacion para el dashboard."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from evaluation.metrics import metrics
except ModuleNotFoundError:
    from src.evaluation.metrics import metrics


WINDOW_KEYS = ["modelo", "split", "finca", "bloque", "fecha_origen", "semana_proyeccion"]


def load_latest_operational_run(root: Path, runs_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Carga la ultima corrida inmutable sin depender de artefactos de validacion."""
    runs = root / runs_path
    latest = runs / "latest.json"
    if not latest.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    pointer = json.loads(latest.read_text(encoding="utf-8"))
    run_id = pointer.get("run_id", "")
    if not re.fullmatch(r"s\d{1,2}-[0-9a-f]{12}", run_id):
        raise ValueError("Run ID operacional invalido")
    run_dir = runs / run_id
    daily_path = run_dir / "predicciones_diarias.parquet"
    weekly_path = run_dir / "predicciones_semanales.parquet"
    manifest_path = run_dir / "manifest.json"
    if not daily_path.exists() or not weekly_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Corrida operacional incompleta: {run_dir}")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != pointer.get("manifest_sha256"):
        raise ValueError("El hash del manifest operacional no coincide con latest")
    manifest = json.loads(manifest_bytes)
    if manifest.get("run_id") != run_id:
        raise ValueError("El manifest no corresponde al puntero operacional")
    dependency_hash = hashlib.sha256(json.dumps(
        manifest.get("dependencies", {}), sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    expected_run_id = f"{manifest.get('semana_entrada', '').lower()}-{dependency_hash[:12]}"
    if expected_run_id != run_id:
        raise ValueError("Las dependencias no corresponden al run ID operacional")
    expected_outputs = {
        "comparacion_modelos.xlsx", "conteos_consolidados.parquet",
        "predicciones_diarias.parquet", "predicciones_semanales.parquet",
        "qa.csv", "videos_validos.parquet",
    }
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict) or set(output_hashes) != expected_outputs:
        raise ValueError("Manifest de salidas operacionales incompleto")
    for name, expected in output_hashes.items():
        path = run_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Integridad invalida en corrida operacional: {path}")
    daily = pd.read_parquet(daily_path)
    weekly = pd.read_parquet(weekly_path)
    return daily, weekly, manifest


def operational_model_comparison(weekly: pd.DataFrame) -> pd.DataFrame:
    """Pivota solo las claves observadas; no genera productos cartesianos."""
    if weekly.empty:
        return weekly.copy()
    index = ["finca", "bloque", "fecha_origen", "semana_proyeccion"]
    comparison = weekly.pivot(index=index, columns="modelo", values="proyectado").reset_index()
    comparison.columns.name = None
    return comparison


def weekly_status(ratio: float | None) -> str:
    """Clasifica la razon semanal segun los rangos operativos de la compania."""
    if ratio is None or not np.isfinite(ratio):
        return "SIN REAL"
    if 0.93 <= ratio <= 1.07:
        return "ACIERTO"
    if 0.90 <= ratio < 0.93 or 1.07 < ratio <= 1.10:
        return "CERCA"
    return "NO ACIERTO"


def load_validation_predictions(root: Path, prediction_files: dict[str, str], split: str) -> pd.DataFrame:
    """Carga exclusivamente trazas producidas sobre una poblacion de validacion."""
    frames = []
    for model, relative_path in prediction_files.items():
        path = root / relative_path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        pred_col = next((col for col in ("pred_bloque", "pred", "proyectado") if col in frame), None)
        required = {"finca", "bloque", "fecha_origen", "fecha_objetivo", "horizonte_dia", "real"}
        if pred_col is None or not required.issubset(frame.columns):
            continue
        frame = frame.rename(columns={pred_col: "proyectado_modelo"}).copy()
        frame["fecha_origen"] = pd.to_datetime(frame["fecha_origen"])
        frame["fecha_objetivo"] = pd.to_datetime(frame["fecha_objetivo"])
        if "semana_proyeccion" not in frame:
            iso = frame["fecha_objetivo"].dt.isocalendar()
            frame["semana_proyeccion"] = iso.year * 100 + iso.week
        # Los productores antiguos no persistian esta columna. El archivo ya
        # pertenece al split solicitado por el dashboard; solo falta verificar
        # que la ventana tenga los siete dias y sus reales observados.
        if "estado_ventana" not in frame:
            frame["estado_ventana"] = "VALIDA"
        else:
            frame["estado_ventana"] = frame["estado_ventana"].fillna("VALIDA")
        frame["modelo"] = model
        frame["split"] = split
        frame["proyectado_modelo"] = pd.to_numeric(frame["proyectado_modelo"], errors="coerce")
        frame["real"] = pd.to_numeric(frame["real"], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def complete_windows(daily: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna ventanas bloque-semana completas y sus filas diarias."""
    expected = set(range(1, horizon_days + 1))
    rows = []
    for key, group in daily.groupby(WINDOW_KEYS, dropna=False, sort=False):
        horizons = set(pd.to_numeric(group["horizonte_dia"], errors="coerce").dropna().astype(int))
        source_valid = "estado_ventana" not in group or group["estado_ventana"].eq("VALIDA").all()
        dates = pd.to_datetime(group["fecha_objetivo"].dropna()).sort_values()
        consecutive = (len(dates) == horizon_days and (dates.iloc[-1] - dates.iloc[0]).days == horizon_days - 1)
        is_complete = (horizons == expected and consecutive and group["real"].notna().sum() == horizon_days
                       and group["proyectado_modelo"].notna().sum() == horizon_days and source_valid)
        if not is_complete:
            continue
        row = dict(zip(WINDOW_KEYS, key))
        row.update(real=group["real"].sum(), proyectado_modelo=group["proyectado_modelo"].sum(),
                   dias=horizon_days)
        rows.append(row)
    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return weekly, daily.iloc[0:0].copy()
    valid_keys = weekly[WINDOW_KEYS].drop_duplicates()
    complete_daily = daily.merge(valid_keys, on=WINDOW_KEYS, how="inner", validate="many_to_one")
    complete_daily["error_abs"] = (complete_daily["proyectado_modelo"] - complete_daily["real"]).abs()
    return weekly, complete_daily


def model_scores(weekly: pd.DataFrame) -> pd.DataFrame:
    """Calcula cada modelo por separado sobre ventanas semanales completas."""
    rows = []
    for model, group in weekly.groupby("modelo"):
        score = metrics(group["real"], group["proyectado_modelo"])
        rows.append({"modelo": model, "ventanas": len(group), **score})
    return pd.DataFrame(rows).sort_values("wape") if rows else pd.DataFrame()


def operational_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Agrega la lectura operacional por semana sobre ventanas completas."""
    if weekly.empty:
        return weekly.copy()
    result = weekly.copy()
    result["real"] = pd.to_numeric(result["real"], errors="coerce")
    result["proyectado_modelo"] = pd.to_numeric(result["proyectado_modelo"], errors="coerce")
    result["diferencia"] = result["proyectado_modelo"] - result["real"]
    result["diferencia_abs"] = result["diferencia"].abs()
    result["ratio_proyectado_real"] = np.where(
        result["real"].ne(0), result["proyectado_modelo"] / result["real"], np.nan)
    result["desviacion_pct"] = result["ratio_proyectado_real"] - 1
    result["indicador"] = result["ratio_proyectado_real"].map(weekly_status)
    return result


def operational_weekly_selected(daily: pd.DataFrame, by_block: bool = False) -> pd.DataFrame:
    """Selecciona una ventana por finca-semana, cercana al lunes.

    La tabla diaria conserva todas las fechas de origen. Esta vista solo elige
    una ventana representativa para el indicador operativo y deja la fecha
    seleccionada visible para auditoria. Con ``by_block=True`` conserva el
    nivel finca-bloque-semana para segmentaciones.
    """
    if daily.empty:
        return daily.copy()
    required = {"modelo", "split", "finca", "bloque", "fecha_origen",
                "fecha_objetivo", "semana_proyeccion", "real", "proyectado_modelo"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Faltan columnas para seleccionar ventana operativa: {sorted(missing)}")

    candidates = []
    # Cada bloque elige su origen mas cercano; despues se suman los bloques
    # para obtener el dato global de finca-semana.
    selection_keys = ["modelo", "split", "finca", "bloque", "semana_proyeccion"]
    candidate_keys = selection_keys + ["fecha_origen"]
    for key, group in daily.groupby(candidate_keys, dropna=False, sort=False):
        objective_dates = pd.to_datetime(group["fecha_objetivo"], errors="coerce").dropna()
        origin = pd.Timestamp(group["fecha_origen"].iloc[0]).normalize()
        if objective_dates.empty or pd.isna(origin):
            continue
        week_start = objective_dates.min().normalize().to_period("W-SUN").start_time.normalize()
        candidates.append({**dict(zip(candidate_keys, key)),
                           "inicio_semana": week_start,
                           "distancia_inicio_dias": abs((origin - week_start).days)})
    if not candidates:
        return pd.DataFrame()
    candidate_frame = pd.DataFrame(candidates)
    selected = (candidate_frame.sort_values(selection_keys + ["distancia_inicio_dias", "fecha_origen"])
                .drop_duplicates(selection_keys, keep="first"))
    selected_keys = selected[candidate_keys + ["inicio_semana", "distancia_inicio_dias"]]
    chosen = daily.merge(selected_keys, on=candidate_keys, how="inner", validate="many_to_one")
    output_keys = selection_keys if by_block else ["modelo", "split", "finca", "semana_proyeccion"]
    if by_block:
        result = (chosen.groupby(output_keys + ["fecha_origen", "inicio_semana", "distancia_inicio_dias"],
                                 as_index=False, dropna=False)
                  .agg(real=("real", "sum"), proyectado_modelo=("proyectado_modelo", "sum"),
                       dias=("fecha_objetivo", "nunique")))
        result["fecha_origen_seleccionada"] = result["fecha_origen"]
    else:
        def origin_label(values):
            dates = sorted(pd.to_datetime(values).dt.strftime("%Y-%m-%d").unique())
            return dates[0] if len(dates) == 1 else ", ".join(dates)

        result = (chosen.groupby(output_keys, as_index=False, dropna=False)
                  .agg(real=("real", "sum"), proyectado_modelo=("proyectado_modelo", "sum"),
                       dias=("fecha_objetivo", "nunique"),
                       fecha_origen_seleccionada=("fecha_origen", origin_label),
                       inicio_semana=("inicio_semana", "first"),
                       distancia_inicio_dias=("distancia_inicio_dias", "min")))
    return operational_weekly(result)


def operational_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    """Cuenta y mide aciertos operativos por modelo."""
    if weekly.empty:
        return pd.DataFrame()
    detail = operational_weekly(weekly)
    summary = detail.groupby("modelo", as_index=False).agg(
        semanas_evaluables=("indicador", lambda values: int(values.isin(["ACIERTO", "CERCA", "NO ACIERTO"]).sum())),
        aciertos=("indicador", lambda values: int(values.eq("ACIERTO").sum())),
        cerca=("indicador", lambda values: int(values.eq("CERCA").sum())),
        no_aciertos=("indicador", lambda values: int(values.eq("NO ACIERTO").sum())),
        sin_real=("indicador", lambda values: int(values.eq("SIN REAL").sum())),
    )
    summary["pct_acierto"] = np.where(summary.semanas_evaluables.gt(0),
                                       summary.aciertos / summary.semanas_evaluables, np.nan)
    summary["pct_acierto_o_cerca"] = np.where(summary.semanas_evaluables.gt(0),
                                               (summary.aciertos + summary.cerca) /
                                               summary.semanas_evaluables, np.nan)
    return summary.sort_values(["pct_acierto", "pct_acierto_o_cerca"], ascending=False)
