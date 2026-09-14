"""Proyeccion M3 de corte a partir de los conteos de Vision-artificial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from canonical import (active_beds_by_date, canonical_block,
                           canonical_farm, load_bed_validity, load_config)
    from models.m3 import fit_m3, load_traditional_intervals, simulate
    from models.operational import (build_operational_windows, load_operational_history,
                                    load_operational_models, score_operational_models,
                                    validate_scoring_cutoff)
    from models.supervised import build_supervised_dataset
except ModuleNotFoundError:
    from src.canonical import (active_beds_by_date, canonical_block,
                               canonical_farm, load_bed_validity, load_config)
    from src.models.m3 import fit_m3, load_traditional_intervals, simulate
    from src.models.operational import (build_operational_windows, load_operational_history,
                                        load_operational_models, score_operational_models,
                                        validate_scoring_cutoff)
    from src.models.supervised import build_supervised_dataset


COUNT_COLUMNS = ["conteo_RC", "conteo_SS", "conteo_AP", "conteo_CO", "conteo_total"]
REQUIRED_COLUMNS = {"semana", "finca", "bloque", "fecha", *COUNT_COLUMNS}
VIDEO_METADATA_COLUMNS = ["duracion_s", "fps", "ancho", "alto"]
VIDEO_COLUMNS = [
    "source_path", "archivo_video", "semana_carpeta", "semana_declarada",
    "finca", "finca_declarada", "bloque", "bloque_declarado", "fecha_conteo",
    "semana_iso", "muestra_idx", "conteo_rc", "conteo_ss", "conteo_ap",
    "conteo_corte_inmediato_visual", "conteo_total", "veredicto_qc",
]
QA_COLUMNS = ["source_path", "estado", "motivo", "detalle"]


def _week_number(value: str) -> int | None:
    match = re.fullmatch(r"S(\d{1,2})", str(value).strip().upper())
    return int(match.group(1)) if match else None


def _read_single_row(path: Path) -> pd.Series:
    first_line = path.open(encoding="utf-8-sig").readline()
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    frame = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
    if len(frame) != 1:
        raise ValueError(f"se esperaba una fila y se encontraron {len(frame)}")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"faltan columnas: {sorted(missing)}")
    return frame.iloc[0]


def _parse_vision_date(value: object) -> pd.Timestamp | pd.NaT:
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?", text):
        return pd.to_datetime(text, yearfirst=True, errors="coerce").normalize()
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return parsed.normalize() if not pd.isna(parsed) else pd.NaT


def _qa(path: Path, estado: str, motivo: str, detalle: object = "") -> dict:
    return {"source_path": str(path), "estado": estado, "motivo": motivo,
            "detalle": "" if pd.isna(detalle) else str(detalle)}


def load_vision_videos(results_root: Path, cfg: dict,
                       week: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee un CSV por cama y devuelve observaciones aceptadas y trazabilidad QA."""
    results_root = Path(results_root)
    selected_week = week.upper() if week else None
    search_root = results_root / selected_week if selected_week else results_root
    if not search_root.is_dir():
        raise FileNotFoundError(f"No existe la entrada visual: {search_root}")

    aliases = cfg["farm_aliases"]
    target_farms = set(cfg["project"]["target_farms"])
    vision_cfg = cfg["vision"]
    accepted: list[dict] = []
    qa_rows: list[dict] = []

    for path in search_root.rglob("*.csv"):
        try:
            relative = path.relative_to(results_root)
        except ValueError:
            qa_rows.append(_qa(path, "RECHAZADO", "RUTA_FUERA_DE_RAIZ"))
            continue
        if len(relative.parts) != 4:
            qa_rows.append(_qa(path, "RECHAZADO", "ESTRUCTURA_RUTA_INVALIDA", relative))
            continue
        path_week, path_farm, path_block, _ = relative.parts
        if selected_week and path_week.upper() != selected_week:
            continue
        if _week_number(path_week) is None:
            qa_rows.append(_qa(path, "RECHAZADO", "SEMANA_RUTA_INVALIDA", path_week))
            continue

        farm = canonical_farm(path_farm, aliases)
        block = canonical_block(path_block)
        if farm not in target_farms:
            continue

        try:
            row = _read_single_row(path)
        except Exception as exc:
            qa_rows.append(_qa(path, "RECHAZADO", "CSV_INVALIDO", exc))
            continue

        date = _parse_vision_date(row["fecha"])
        if pd.isna(date):
            qa_rows.append(_qa(path, "RECHAZADO", "FECHA_INVALIDA", row["fecha"]))
            continue
        counts = pd.to_numeric(row[COUNT_COLUMNS], errors="coerce")
        if (counts.isna().any() or not np.isfinite(counts).all() or (counts < 0).any()
                or not np.allclose(counts, np.round(counts))):
            qa_rows.append(_qa(path, "RECHAZADO", "CONTEOS_INVALIDOS"))
            continue
        counts = counts.astype("int64")
        state_sum = int(counts[["conteo_RC", "conteo_SS", "conteo_AP", "conteo_CO"]].sum())
        if state_sum != int(counts["conteo_total"]):
            qa_rows.append(_qa(path, "RECHAZADO", "TOTAL_NO_COINCIDE", state_sum))
            continue

        if vision_cfg["require_valid_video_metadata"]:
            metadata = pd.to_numeric(row.reindex(VIDEO_METADATA_COLUMNS), errors="coerce")
            if metadata.isna().any() or not np.isfinite(metadata).all() or (metadata <= 0).any():
                qa_rows.append(_qa(path, "RECHAZADO", "METADATA_VIDEO_INVALIDA"))
                continue

        verdict = str(row.get("veredicto_qc", "")).strip().lower()
        if vision_cfg["reject_bad_qc"] and verdict == "mal tomado":
            qa_rows.append(_qa(path, "RECHAZADO", "QC_MAL_TOMADO"))
            continue
        if not verdict or verdict == "nan":
            qa_rows.append(_qa(path, "ADVERTENCIA", "QC_NO_INFORMADO"))

        declared_farm = canonical_farm(row["finca"], aliases)
        declared_block = canonical_block(row["bloque"])
        declared_week = str(row["semana"]).strip().upper()
        mismatches = []
        if declared_farm != farm:
            mismatches.append(f"finca={row['finca']}")
        if declared_block != block:
            mismatches.append(f"bloque={row['bloque']}")
        if declared_week != path_week.upper():
            mismatches.append(f"semana={row['semana']}")
        if mismatches:
            state = "RECHAZADO" if vision_cfg["reject_identity_mismatch"] else "ADVERTENCIA"
            qa_rows.append(_qa(path, state, "METADATA_DIFIERE_DE_RUTA", ", ".join(mismatches)))
            if state == "RECHAZADO":
                continue

        iso = date.isocalendar()
        if int(iso.week) != _week_number(path_week):
            qa_rows.append(_qa(path, "RECHAZADO", "FECHA_NO_COINCIDE_CON_SEMANA",
                               f"{date.date()} no pertenece a {path_week}"))
            continue
        accepted.append({
            "source_path": str(path),
            "archivo_video": str(row.get("archivo_original", path.stem)),
            "semana_carpeta": path_week.upper(),
            "semana_declarada": declared_week,
            "finca": farm,
            "finca_declarada": str(row["finca"]),
            "bloque": block,
            "bloque_declarado": str(row["bloque"]),
            "fecha_conteo": date,
            "semana_iso": int(iso.year * 100 + iso.week),
            "muestra_idx": str(row.get("muestra_idx", "")),
            "conteo_rc": int(counts["conteo_RC"]),
            "conteo_ss": int(counts["conteo_SS"]),
            "conteo_ap": int(counts["conteo_AP"]),
            "conteo_corte_inmediato_visual": int(counts["conteo_CO"]),
            "conteo_total": int(counts["conteo_total"]),
            "veredicto_qc": "" if verdict == "nan" else verdict,
        })

    videos = pd.DataFrame(accepted, columns=VIDEO_COLUMNS)
    qa = pd.DataFrame(qa_rows, columns=QA_COLUMNS)
    return videos, qa


def aggregate_vision_counts(videos: pd.DataFrame, beds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega camas visuales y calcula expansion al bloque con vigencia temporal."""
    if videos.empty:
        return pd.DataFrame(), pd.DataFrame(columns=QA_COLUMNS)
    counts = (videos.groupby(["finca", "bloque", "fecha_conteo", "semana_iso"], as_index=False)
              .agg(conteo_rc=("conteo_rc", "sum"), conteo_ss=("conteo_ss", "sum"),
                   conteo_ap=("conteo_ap", "sum"),
                   conteo_corte_inmediato_visual=("conteo_corte_inmediato_visual", "sum"),
                   conteo_total=("conteo_total", "sum"),
                   camas_muestreadas=("source_path", "nunique")))
    active = active_beds_by_date(beds, counts["fecha_conteo"]).rename(
        columns={"fecha": "fecha_conteo"})
    counts = counts.merge(active, on=["finca", "bloque", "fecha_conteo"], how="left",
                          validate="one_to_one")
    qa_rows = []
    missing = counts["camas_activas"].isna() | counts["camas_activas"].eq(0)
    for row in counts[missing].itertuples():
        qa_rows.append(_qa(Path(""), "RECHAZADO", "SIN_CAMAS_ACTIVAS",
                           f"{row.finca}/{row.bloque}/{row.fecha_conteo.date()}"))
    counts["factor_extrapolacion"] = counts["camas_activas"] / counts["camas_muestreadas"]
    over = counts["camas_muestreadas"] > counts["camas_activas"]
    for row in counts[over.fillna(False)].itertuples():
        qa_rows.append(_qa(Path(""), "ADVERTENCIA", "MUESTREADAS_SUPERAN_ACTIVAS",
                           f"{row.finca}/{row.bloque}: {row.camas_muestreadas}>{row.camas_activas}"))
    counts.loc[over, "factor_extrapolacion"] = 1.0
    counts["cobertura_muestreo"] = counts["camas_muestreadas"] / counts["camas_activas"]
    return counts[~missing].reset_index(drop=True), pd.DataFrame(qa_rows, columns=QA_COLUMNS)


def _period_for_origin(origin: pd.Timestamp, periods: list[dict]) -> str:
    for period in periods:
        start = pd.Timestamp(period["fecha_inicio_vigencia"])
        end = pd.Timestamp(period["fecha_fin_vigencia"])
        if start <= origin <= end:
            return period["name"]
    raise ValueError(f"Fecha {origin.date()} fuera de las vigencias M3 configuradas")


def project_vision_counts(counts: pd.DataFrame, intervals: pd.DataFrame,
                          cfg: dict, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proyecta el ultimo conteo valido de cada bloque y semana."""
    origins = (counts.sort_values("fecha_conteo")
               .groupby(["finca", "bloque", "semana_iso"], as_index=False).tail(1))
    rows = []
    horizon = int(cfg["forecast"]["horizon_days"])
    for origin in origins.itertuples():
        date = pd.Timestamp(origin.fecha_conteo)
        period = _period_for_origin(date, cfg["m3"]["periods"])
        matrix = fit_m3(intervals, origin.finca, period, max_date=date)
        x0 = np.array([origin.conteo_rc, origin.conteo_ss, origin.conteo_ap], dtype=float)
        first_target = date.to_period("W-SUN").end_time.normalize() + pd.Timedelta(days=1)
        targets = pd.date_range(first_target, periods=horizon, freq="D")
        max_lead = int((targets[-1] - date).days)
        simulation = simulate(matrix, x0, max_lead, alpha).set_index("dia_simulado")
        for h, target in enumerate(targets, 1):
            lead = int((target - date).days)
            pred_sample = float(simulation.loc[lead, "PC_dia_muestra"])
            rows.append({
                "modelo": "E00_M3_BASE_VISION", "finca": origin.finca,
                "bloque": origin.bloque, "fecha_origen": date,
                "fecha_objetivo": target, "horizonte_dia": h, "lead_dias": lead,
                "semana_proyeccion": int(target.isocalendar().year * 100 + target.isocalendar().week),
                "conteo_rc_t0": origin.conteo_rc, "conteo_ss_t0": origin.conteo_ss,
                "conteo_ap_t0": origin.conteo_ap,
                "conteo_corte_inmediato_visual_t0": origin.conteo_corte_inmediato_visual,
                "camas_muestreadas_t0": origin.camas_muestreadas,
                "camas_activas_t0": origin.camas_activas,
                "factor_extrapolacion_t0": origin.factor_extrapolacion,
                "pred_muestra": pred_sample,
                "pred_bloque": pred_sample * origin.factor_extrapolacion,
                "alpha_ingreso_rc": alpha, "periodo": period, "causal": True,
                "fecha_max_dato_modelo": date,
            })
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, pd.DataFrame()
    weekly = (daily.groupby(["modelo", "finca", "bloque", "fecha_origen",
                             "semana_proyeccion"], as_index=False)
              .agg(pred_bloque_semana=("pred_bloque", "sum"),
                   camas_muestreadas=("camas_muestreadas_t0", "first"),
                   camas_activas=("camas_activas_t0", "first"),
                   factor_extrapolacion=("factor_extrapolacion_t0", "first")))
    return daily, weekly


def _input_sha256(results_root: Path, week: str) -> str:
    digest = hashlib.sha256()
    for path in sorted((results_root / week).rglob("*.csv")):
        digest.update(path.relative_to(results_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependency_hashes(root: Path, results_root: Path, week: str,
                       cfg: dict, model_manifest: dict) -> dict[str, str]:
    raw = root / cfg["paths"]["raw"]
    history = root / cfg["paths"]["external"] / cfg["vision"]["operational_history_path"]
    paths = {
        "historial_cortes": history,
        "plano_siembra": raw / cfg["sources"]["plano_siembra"],
        "codigo_proyeccion": root / "src" / "proyeccion_vision.py",
        "codigo_operacional": root / "src" / "models" / "operational.py",
        "codigo_supervisado": root / "src" / "models" / "supervised.py",
        "codigo_m3": root / "src" / "models" / "m3.py",
        "codigo_canonico": root / "src" / "canonical.py",
    }
    for period in cfg["m3"]["periods"]:
        paths[f"m3_{period['name']}"] = raw / cfg["sources"][period["source_key"]]
    hashes = {name: _file_sha256(path) if path.exists() else "MISSING"
              for name, path in paths.items()}
    hashes["entrada_visual"] = _input_sha256(results_root, week)
    hashes["modelo_rf"] = model_manifest["artifact_sha256"]
    hashes["manifest_modelo"] = model_manifest["manifest_sha256"]
    hashes["configuracion"] = hashlib.sha256(json.dumps(
        cfg, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")).hexdigest()
    return hashes


def _operational_run_id(week: str, dependencies: dict[str, str]) -> str:
    digest = hashlib.sha256(json.dumps(
        dependencies, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"{week.lower()}-{digest[:12]}"


def _weekly_predictions(daily: pd.DataFrame) -> pd.DataFrame:
    keys = ["modelo", "finca", "bloque", "fecha_origen", "semana_proyeccion"]
    aggregations = {
        "proyectado": ("proyectado", lambda values: values.sum(min_count=1)),
        "dias_proyectados": ("proyectado", "count"),
        "estado_modelo": ("estado_modelo", "first"), "motivo": ("motivo", "first"),
    }
    for column in ("familia", "features", "causal", "run_id", "training_cutoff",
                   "model_artifact_sha256", "model_data_cutoff"):
        if column in daily:
            aggregations[column] = (column, "first")
    weekly = (daily.groupby(keys, as_index=False, dropna=False)
              .agg(**aggregations))
    weekly.loc[weekly["dias_proyectados"].ne(7), "proyectado"] = np.nan
    return weekly


def _comparison(weekly: pd.DataFrame) -> pd.DataFrame:
    index = ["finca", "bloque", "fecha_origen", "semana_proyeccion"]
    comparison = weekly.pivot(index=index, columns="modelo", values="proyectado").reset_index()
    comparison.columns.name = None
    comparison["modelo_elegido"] = ""
    comparison["valor_reportado"] = np.nan
    return comparison


def _validate_existing_run(run_dir: Path) -> None:
    required = {"comparacion_modelos.xlsx", "conteos_consolidados.parquet", "manifest.json",
                "predicciones_diarias.parquet", "predicciones_semanales.parquet", "qa.csv",
                "videos_validos.parquet"}
    missing = sorted(name for name in required if not (run_dir / name).exists())
    if missing:
        raise ValueError(f"Corrida operacional incompleta {run_dir}: {missing}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_dir.name:
        raise ValueError(f"El manifest no corresponde a {run_dir}")
    expected_run_id = _operational_run_id(
        manifest.get("semana_entrada", ""), manifest.get("dependencies", {}))
    if expected_run_id != run_dir.name:
        raise ValueError(f"Las dependencias no corresponden a {run_dir}")
    output_hashes = manifest.get("output_sha256")
    expected_outputs = required.difference({"manifest.json"})
    if not isinstance(output_hashes, dict) or set(output_hashes) != expected_outputs:
        raise ValueError(f"Manifest de salidas incompleto en {run_dir}")
    for name, expected in output_hashes.items():
        if _file_sha256(run_dir / name) != expected:
            raise ValueError(f"Hash invalido para {run_dir / name}")


def _validate_single_iso_year(videos: pd.DataFrame, week: str) -> None:
    if videos["semana_iso"].floordiv(100).nunique() != 1:
        raise ValueError(f"La carpeta {week} contiene conteos de mas de un ano ISO")


def _next_week_iso(date: pd.Timestamp) -> int:
    """Devuelve la semana ISO objetivo del lunes siguiente al origen."""
    monday = pd.Timestamp(date).to_period("W-SUN").end_time.normalize() + pd.Timedelta(days=1)
    iso = monday.isocalendar()
    return int(iso.year * 100 + iso.week)


def _validate_next_week_prediction(daily: pd.DataFrame, videos: pd.DataFrame,
                                  week: str) -> int:
    """Evita mezclar semanas: una corrida solo puede proyectar la siguiente."""
    expected = {_next_week_iso(date) for date in pd.to_datetime(videos["fecha_conteo"])}
    actual = set(pd.to_numeric(daily["semana_proyeccion"], errors="coerce").dropna().astype(int))
    if len(expected) != 1 or actual != expected:
        raise ValueError(
            f"La corrida {week} no proyecta exactamente la semana siguiente: "
            f"esperada={sorted(expected)}, encontrada={sorted(actual)}"
        )
    return next(iter(expected))


def _publish_latest(runs: Path, run_id: str) -> None:
    manifest_hash = _file_sha256(runs / run_id / "manifest.json")
    latest_temporary = runs / f".latest-{os.getpid()}.tmp"
    latest_temporary.write_text(json.dumps(
        {"run_id": run_id, "manifest_sha256": manifest_hash}, indent=2), encoding="utf-8")
    latest_temporary.replace(runs / "latest.json")


def _latest_week(results_root: Path) -> str:
    weeks = []
    for directory in results_root.iterdir():
        if not directory.is_dir() or _week_number(directory.name) is None:
            continue
        dates = []
        csv_paths = list(directory.rglob("*.csv"))
        for path in csv_paths:
            match = re.search(r"(20\d{6})", path.stem)
            if match:
                parsed = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
                if not pd.isna(parsed):
                    dates.append(parsed)
        if not dates:
            for path in csv_paths[:100]:
                try:
                    parsed = _parse_vision_date(_read_single_row(path)["fecha"])
                except Exception:
                    continue
                if not pd.isna(parsed):
                    dates.append(parsed)
        if dates:
            weeks.append((max(dates), directory.name.upper()))
    if not weeks:
        raise ValueError(f"No se encontraron carpetas SNN en {results_root}")
    return max(weeks)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semana", help="Semana visual SNN; por defecto usa la ultima disponible")
    parser.add_argument("--input", type=Path, help="Raiz alternativa de data/vision/Resultados")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    if cfg["vision"].get("one_video_per_bed") is not True:
        raise ValueError("La proyeccion requiere confirmar vision.one_video_per_bed=true")
    if cfg["vision"].get("conteo_co_semantics") != "CORTE_INMEDIATO_VISUAL":
        raise ValueError("La semantica de conteo_CO visual no esta confirmada")
    raw = root / cfg["paths"]["raw"]
    results_root = args.input or root / cfg["paths"]["vision"] / cfg["vision"]["results_path"]
    week = args.semana.upper() if args.semana else _latest_week(results_root)
    alpha = float(cfg["m3"]["baseline_ingress"])

    videos, qa = load_vision_videos(results_root, cfg, week)
    if videos.empty:
        raise ValueError(f"No hay videos validos para {week}")
    _validate_single_iso_year(videos, week)
    registry = root / cfg["operational"]["model_registry"]
    model_pointer = root / cfg["operational"]["active_model_pointer"]
    bundle, model_manifest = load_operational_models(registry, model_pointer)
    if model_manifest.get("models") != cfg["operational"]["models"]:
        raise ValueError("La configuracion operacional difiere del artefacto entrenado")
    if model_manifest.get("conteo_co_semantics") != cfg["vision"]["conteo_co_semantics"]:
        raise ValueError("La semantica CO difiere entre entrenamiento y scoring")
    dependencies = _dependency_hashes(root, results_root, week, cfg, model_manifest)
    run_id = _operational_run_id(week, dependencies)
    runs = root / cfg["operational"]["runs_path"]
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / run_id
    if run_dir.exists():
        _validate_existing_run(run_dir)
        _publish_latest(runs, run_id)
        print(f"Corrida ya existente: {run_dir}")
        return
    beds = load_bed_validity(raw, cfg)
    counts, aggregate_qa = aggregate_vision_counts(videos, beds)
    qa = pd.concat([qa, aggregate_qa], ignore_index=True)
    if counts.empty:
        raise ValueError(f"No hay conteos con camas activas para {week}")
    intervals = load_traditional_intervals(raw, cfg)
    m3_daily, _ = project_vision_counts(counts, intervals, cfg, alpha)
    m3_daily = m3_daily.rename(columns={"pred_bloque": "proyectado"})
    m3_daily["modelo"] = cfg["operational"]["models"]["m3"]["name"]
    m3_daily["estado_modelo"] = "DISPONIBLE"
    m3_daily["motivo"] = ""

    history_path = root / cfg["paths"]["external"] / cfg["vision"]["operational_history_path"]
    history = load_operational_history(history_path, cfg)
    windows = build_operational_windows(counts, int(cfg["forecast"]["horizon_days"]))
    feature_frame = build_supervised_dataset(
        windows, history, intervals, cfg, include_incomplete=True)
    validate_scoring_cutoff(feature_frame, model_manifest)
    rf_daily = score_operational_models(feature_frame, bundle, cfg)
    m3_daily["model_data_cutoff"] = m3_daily["fecha_max_dato_modelo"]
    rf_daily["model_data_cutoff"] = pd.Timestamp(model_manifest["training_cutoff"])
    output_columns = ["modelo", "finca", "bloque", "fecha_origen", "fecha_objetivo",
                      "semana_proyeccion", "horizonte_dia", "proyectado",
                      "estado_modelo", "motivo", "model_data_cutoff"]
    daily = pd.concat([m3_daily[output_columns], rf_daily[output_columns]], ignore_index=True)
    model_metadata = {
        cfg["operational"]["models"]["m3"]["name"]: ("M3", "FENO_M3"),
        cfg["operational"]["models"]["rf_feno"]["name"]: ("RF", "FENO"),
        cfg["operational"]["models"]["rf_h1_h7"]["name"]: ("RF_HORIZON", "FENO"),
    }
    daily["familia"] = daily["modelo"].map(lambda value: model_metadata[value][0])
    daily["features"] = daily["modelo"].map(lambda value: model_metadata[value][1])
    daily["causal"] = True
    daily["run_id"] = run_id
    daily["training_cutoff"] = np.where(
        daily["familia"].eq("M3"), None, model_manifest["training_cutoff"])
    daily["model_artifact_sha256"] = np.where(
        daily["familia"].eq("M3"), None, model_manifest["artifact_sha256"])
    prediction_key = ["modelo", "finca", "bloque", "fecha_origen", "fecha_objetivo", "horizonte_dia"]
    if daily.duplicated(prediction_key).any():
        raise ValueError("La prediccion operacional tiene claves duplicadas")
    weekly = _weekly_predictions(daily)
    target_week = _validate_next_week_prediction(daily, videos, week)
    comparison = _comparison(weekly)

    run_manifest = {
        "run_id": run_id, "semana_entrada": week,
        "semana_objetivo": target_week,
        "prediction_scope": "ONLY_NEXT_ISO_WEEK",
        "input_sha256": dependencies["entrada_visual"],
        "model_artifact_sha256": model_manifest["artifact_sha256"],
        "model_manifest_sha256": model_manifest["manifest_sha256"],
        "training_cutoff": model_manifest["training_cutoff"],
        "models": [cfg["operational"]["models"][key]["name"]
                   for key in ("m3", "rf_feno", "rf_h1_h7")],
        "camas_validas": int(len(videos)), "origenes": int(len(comparison)),
        "predictions_are_immutable": True, "dependencies": dependencies,
        "conteo_co_semantics": cfg["vision"]["conteo_co_semantics"],
    }
    with tempfile.TemporaryDirectory(dir=runs, prefix=f".{run_id}-") as temporary:
        temporary_dir = Path(temporary)
        daily.to_parquet(temporary_dir / "predicciones_diarias.parquet", index=False)
        weekly.to_parquet(temporary_dir / "predicciones_semanales.parquet", index=False)
        counts.to_parquet(temporary_dir / "conteos_consolidados.parquet", index=False)
        videos.to_parquet(temporary_dir / "videos_validos.parquet", index=False)
        qa.to_csv(temporary_dir / "qa.csv", index=False)
        operational_report = temporary_dir / "comparacion_modelos.xlsx"
        with pd.ExcelWriter(operational_report, engine="openpyxl") as writer:
            comparison.to_excel(writer, sheet_name="Comparacion", index=False)
            weekly.to_excel(writer, sheet_name="Semanal", index=False)
            daily.to_excel(writer, sheet_name="Diaria", index=False)
            counts.to_excel(writer, sheet_name="Conteos", index=False)
            qa.to_excel(writer, sheet_name="QA", index=False)
        output_names = ["comparacion_modelos.xlsx", "conteos_consolidados.parquet",
                        "predicciones_diarias.parquet", "predicciones_semanales.parquet",
                        "qa.csv", "videos_validos.parquet"]
        run_manifest["output_sha256"] = {
            name: _file_sha256(temporary_dir / name) for name in output_names}
        (temporary_dir / "manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary_dir.replace(run_dir)
    _publish_latest(runs, run_id)
    operational_report = run_dir / "comparacion_modelos.xlsx"
    print(f"Semana procesada: {week}")
    print(f"Camas visuales validas: {len(videos):,}")
    print(f"Origenes proyectados: {len(comparison):,}")
    print(f"Reporte operacional: {operational_report}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
