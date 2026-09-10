"""Entrenamiento y scoring de modelos fenologicos operacionales."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

try:
    from canonical import canonical_block, canonical_farm, parse_date
    from models.supervised import feature_groups
except ModuleNotFoundError:
    from src.canonical import canonical_block, canonical_farm, parse_date
    from src.models.supervised import feature_groups


def feno_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = set(frame.select_dtypes(include=[np.number]).columns)
    return list(dict.fromkeys(
        column for column in feature_groups(frame)["FENO"]
        if column in numeric and column != "target"
    ))


def training_rows(frame: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Selecciona solo targets conocidos hasta el cutoff de entrenamiento."""
    data = frame.copy()
    data["fecha_objetivo"] = pd.to_datetime(data["fecha_objetivo"])
    data = data[data["target"].notna()].copy()
    if cutoff is not None:
        data = data[data["fecha_objetivo"] <= pd.Timestamp(cutoff)].copy()
    if data.empty:
        raise ValueError("No hay targets historicos para entrenar modelos operacionales")
    return data


def _eligible_history(frame: pd.DataFrame, minimum: int) -> pd.Series:
    core = ["RC_t0", "SS_t0", "AP_t0", "TOTAL_t0", "M3_pred_bloque",
            "camas_activas", "camas_muestreadas", "factor_extrapolacion"]
    core_valid = frame.reindex(columns=core).notna().all(axis=1)
    observed = (pd.to_numeric(frame["corte_dias_observados_28d"], errors="coerce")
                if "corte_dias_observados_28d" in frame
                else pd.Series(np.nan, index=frame.index))
    return core_valid & observed.ge(minimum)


def train_operational_models(frame: pd.DataFrame, cfg: dict,
                             cutoff: pd.Timestamp | None = None) -> tuple[dict, dict]:
    train = training_rows(frame, cutoff)
    total_before_policy = len(train)
    minimum = int(cfg["operational"]["minimum_history_days_28d"])
    train = train[_eligible_history(train, minimum)].copy()
    if train.empty:
        raise ValueError("No hay filas con historial suficiente para entrenar")
    features = feno_feature_columns(train)
    if not features:
        raise ValueError("No se encontraron features FENO para entrenamiento")
    x = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train["target"].astype(float)
    specs = cfg["operational"]["models"]

    pooled = RandomForestRegressor(**specs["rf_feno"]["hyperparameters"])
    pooled.fit(x, y)
    horizons = {}
    for horizon in range(1, int(cfg["forecast"]["horizon_days"]) + 1):
        current = train["horizonte_dia"].eq(horizon)
        if not current.any():
            raise ValueError(f"No hay filas de entrenamiento para H{horizon}")
        estimator = RandomForestRegressor(**specs["rf_h1_h7"]["hyperparameters"])
        estimator.fit(x.loc[current], y.loc[current])
        horizons[horizon] = estimator

    max_target = pd.Timestamp(train["fecha_objetivo"].max())
    training_hash = hashlib.sha256(
        pd.util.hash_pandas_object(train, index=True).values.tobytes()).hexdigest()
    bundle = {"features": features, "pooled": pooled, "horizons": horizons}
    manifest = {
        "training_cutoff": max_target.strftime("%Y-%m-%d"),
        "training_rows": int(len(train)),
        "training_rows_excluded_by_history_policy": int(total_before_policy - len(train)),
        "training_origins": int(train[["finca", "bloque", "fecha_origen"]].drop_duplicates().shape[0]),
        "features": features,
        "models": specs,
        "target": "corte_real_dia",
        "conteo_co_semantics": cfg["vision"]["conteo_co_semantics"],
        "training_data_sha256": training_hash,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "causal_training_filter": "fecha_objetivo <= training_cutoff; M3 usa solo trayectorias terminadas en t0",
    }
    return bundle, manifest


def save_operational_models(bundle: dict, manifest: dict,
                            registry: Path, pointer_path: Path) -> tuple[Path, Path]:
    """Persiste un artefacto content-addressed y actualiza el puntero activo."""
    registry.mkdir(parents=True, exist_ok=True)
    temporary_artifact = registry / f".modelos_feno-{os.getpid()}.tmp"
    joblib.dump(bundle, temporary_artifact)
    artifact_hash = hashlib.sha256(temporary_artifact.read_bytes()).hexdigest()
    frozen_manifest = {**manifest, "artifact_sha256": artifact_hash,
                       "artifact_file": "modelos_feno.joblib"}
    manifest_payload = json.dumps(
        frozen_manifest, indent=2, ensure_ascii=False).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    version_hash = hashlib.sha256((artifact_hash + manifest_hash).encode("ascii")).hexdigest()
    version_dir = registry / version_hash[:16]
    version_dir.mkdir(exist_ok=True)
    artifact_path = version_dir / "modelos_feno.joblib"
    manifest_path = version_dir / "manifest.json"
    if artifact_path.exists() and hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact_hash:
        raise ValueError(f"Colision de artefacto operacional en {version_dir}")
    if artifact_path.exists():
        temporary_artifact.unlink()
    else:
        temporary_artifact.replace(artifact_path)
    if manifest_path.exists() and hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
        raise ValueError(f"Manifest distinto para version operacional {version_dir}")
    if not manifest_path.exists():
        temporary_manifest = version_dir / f".manifest-{os.getpid()}.tmp"
        temporary_manifest.write_bytes(manifest_payload)
        temporary_manifest.replace(manifest_path)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_pointer = pointer_path.with_name(f".{pointer_path.name}-{os.getpid()}.tmp")
    temporary_pointer.write_text(json.dumps(
        {"version": version_dir.name, "manifest_sha256": manifest_hash}, indent=2), encoding="utf-8")
    temporary_pointer.replace(pointer_path)
    return artifact_path, manifest_path


def load_operational_models(registry: Path, pointer_path: Path) -> tuple[dict, dict]:
    if not pointer_path.exists():
        raise FileNotFoundError(
            "Faltan modelos operacionales. Ejecute src/entrenar_modelos_operacionales.py"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    version = pointer["version"]
    if not re.fullmatch(r"[0-9a-f]{16}", version):
        raise ValueError("Version operacional invalida en el puntero activo")
    artifact_path = registry / version / "modelos_feno.joblib"
    manifest_path = registry / version / "manifest.json"
    if not artifact_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Version operacional incompleta: {registry / version}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != pointer.get("manifest_sha256"):
        raise ValueError("El hash del manifest operacional no coincide con el puntero activo")
    current_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if current_hash != manifest.get("artifact_sha256"):
        raise ValueError("El hash del artefacto operacional no coincide con su manifest")
    expected_version = hashlib.sha256((current_hash + manifest_hash).encode("ascii")).hexdigest()[:16]
    if version != expected_version:
        raise ValueError("La version operacional no corresponde a artefacto y manifest")
    return joblib.load(artifact_path), {**manifest, "manifest_sha256": manifest_hash,
                                        "version": version}


def validate_scoring_cutoff(frame: pd.DataFrame, manifest: dict) -> None:
    origins = pd.to_datetime(frame["fecha_origen"], errors="coerce")
    cutoff = pd.Timestamp(manifest["training_cutoff"])
    if origins.isna().any():
        raise ValueError("Hay fechas de origen invalidas en el scoring operacional")
    if origins.min() <= cutoff:
        raise ValueError(
            f"El origen {origins.min().date()} no es posterior al cutoff de entrenamiento {cutoff.date()}"
        )


def score_operational_models(frame: pd.DataFrame, bundle: dict, cfg: dict) -> pd.DataFrame:
    """Puntua RF pooled y H1-H7; faltantes historicos se bloquean, no se imputan."""
    required = list(bundle["features"])
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Faltan features operacionales: {missing}")
    minimum = int(cfg["operational"]["minimum_history_days_28d"])
    eligible = _eligible_history(frame, minimum)
    x = frame[required].replace([np.inf, -np.inf], np.nan).fillna(0)
    base = frame[["finca", "bloque", "fecha_origen", "fecha_objetivo",
                  "semana_proyeccion", "horizonte_dia"]].copy()
    outputs = []
    specs = cfg["operational"]["models"]
    for key, estimators in (("rf_feno", bundle["pooled"]), ("rf_h1_h7", bundle["horizons"])):
        result = base.copy()
        result["modelo"] = specs[key]["name"]
        result["proyectado"] = np.nan
        if key == "rf_feno":
            if eligible.any():
                result.loc[eligible, "proyectado"] = np.maximum(0, estimators.predict(x.loc[eligible]))
        else:
            for horizon, estimator in estimators.items():
                current = eligible & frame["horizonte_dia"].eq(int(horizon))
                if current.any():
                    result.loc[current, "proyectado"] = np.maximum(0, estimator.predict(x.loc[current]))
        result["estado_modelo"] = np.where(eligible, "DISPONIBLE", "NO_DISPONIBLE")
        result["motivo"] = np.where(eligible, "", "HISTORIAL_CORTES_INSUFICIENTE")
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def load_operational_history(path: Path, cfg: dict) -> pd.DataFrame:
    """Carga cortes diarios disponibles para construir rezagos causales."""
    if not path.exists():
        return pd.DataFrame(columns=["finca", "bloque", "fecha", "corte_comercial_real"])
    source = pd.read_excel(path)
    required = {"Finca", "Bloque", "Fecha", "Cantidad"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Faltan columnas en historial operacional: {sorted(missing)}")
    if "Variedad" in source:
        source = source[source["Variedad"].astype(str).str.upper().eq(cfg["project"]["variety"])]
    source = source.copy()
    source["finca"] = source["Finca"].map(lambda value: canonical_farm(value, cfg["farm_aliases"]))
    source["bloque"] = source["Bloque"].map(canonical_block)
    source["fecha"] = parse_date(source["Fecha"])
    source["corte_comercial_real"] = pd.to_numeric(source["Cantidad"], errors="coerce")
    source = source[source["finca"].isin(cfg["project"]["target_farms"]) & source["fecha"].notna()]
    return (source.groupby(["finca", "bloque", "fecha"], as_index=False)
            .agg(corte_comercial_real=("corte_comercial_real",
                                       lambda values: values.sum(min_count=1))))


def build_operational_windows(counts: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Construye H1-H7 para el ultimo conteo disponible de cada bloque/semana."""
    origins = (counts.sort_values("fecha_conteo")
               .groupby(["finca", "bloque", "semana_iso"], as_index=False).tail(1))
    rows = []
    for origin in origins.itertuples():
        first_target = (pd.Timestamp(origin.fecha_conteo).to_period("W-SUN").end_time.normalize()
                        + pd.Timedelta(days=1))
        for horizon, target in enumerate(pd.date_range(first_target, periods=horizon_days), 1):
            rows.append({
                "finca": origin.finca, "bloque": origin.bloque,
                "fecha_origen": origin.fecha_conteo, "semana_origen": origin.semana_iso,
                "semana_objetivo": int(target.isocalendar().year * 100 + target.isocalendar().week),
                "semana_proyeccion": int(target.isocalendar().year * 100 + target.isocalendar().week),
                "fecha_objetivo": target, "horizonte_dia": horizon,
                "conteo_RC_t0": origin.conteo_rc, "conteo_SS_t0": origin.conteo_ss,
                "conteo_AP_t0": origin.conteo_ap,
                "conteo_CO_t0": origin.conteo_corte_inmediato_visual,
                "conteo_total_t0": origin.conteo_total,
                "camas_muestreadas_t0": origin.camas_muestreadas,
                "camas_activas_t0": origin.camas_activas,
                "factor_extrapolacion_t0": origin.factor_extrapolacion,
                "corte_real_dia": np.nan, "corte_real_semana": np.nan,
                "estado_ventana": "PENDIENTE_REAL",
            })
    return pd.DataFrame(rows)
