"""Construccion de la capa canonica de datos de Fase 1."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_farm(value: object, aliases: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None
    raw = re.sub(r"\s+", " ", str(value).strip()).upper()
    normalized = {re.sub(r"\s+", " ", str(k).strip()).upper(): v
                  for k, v in aliases.items()}
    return normalized.get(raw)


def canonical_block(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.normalize()
    return pd.to_datetime(series, dayfirst=True, errors="coerce").dt.normalize()


def parse_iso_week(value: object, year: int = 2026) -> int | None:
    if pd.isna(value):
        return None
    match = re.fullmatch(r"S(\d{1,2})", str(value).strip().upper())
    return year * 100 + int(match.group(1)) if match else None


def load_conteos(raw: Path, aliases: dict, target_farms: list[str]) -> pd.DataFrame:
    df = pd.read_excel(raw / "conteos_vs_cortes_multifinca.xlsx")
    df["finca"] = df["Finca"].map(lambda x: canonical_farm(x, aliases))
    df["bloque"] = df["Bloque"].map(canonical_block)
    df["fecha"] = parse_date(df["Fecha"])
    df["semana_iso"] = pd.to_numeric(df["semana"], errors="coerce").astype("Int64")
    df = df[df["finca"].isin(target_farms) &
            df["Variedad"].astype(str).str.upper().eq("FREEDOM")].copy()
    rename = {
        "Cantidad": "corte_comercial_real",
        "conteo_RC": "conteo_RC", "conteo_SS": "conteo_SS",
        "conteo_AP": "conteo_AP", "conteo_CO": "conteo_CO",
        "conteo_total": "conteo_total",
    }
    source_cols = ["finca", "bloque", "fecha", "semana_iso", "Cantidad",
                   "conteo_RC", "conteo_SS", "conteo_AP", "conteo_CO",
                   "conteo_total"]
    out = df[source_cols].rename(columns=rename)
    if out.duplicated(["finca", "bloque", "fecha"]).any():
        raise ValueError("conteos tiene claves finca+bloque+fecha duplicadas")
    return out


def load_sampled_beds(raw: Path, aliases: dict, target_farms: list[str]) -> pd.DataFrame:
    df = pd.read_excel(raw / "camas_muestreadas_semana.xlsx")
    df = df[df["Semana"].astype(str).str.fullmatch(r"S\d+", na=False)].copy()
    df["finca"] = df["Finca"].map(lambda x: canonical_farm(x, aliases))
    df["bloque"] = df["Bloque"].map(canonical_block)
    df["semana_iso"] = df["Semana"].map(parse_iso_week).astype("Int64")
    df["camas_muestreadas"] = pd.to_numeric(df["Cantidad_csv"], errors="coerce")
    out = df[df["finca"].isin(target_farms)][
        ["finca", "bloque", "semana_iso", "camas_muestreadas"]].copy()
    if out.duplicated(["finca", "bloque", "semana_iso"]).any():
        raise ValueError("camas_muestreadas tiene claves duplicadas")
    return out


def load_bed_validity(raw: Path, aliases: dict, target_farms: list[str]) -> pd.DataFrame:
    df = pd.read_excel(raw / "plano_siembra.xlsx")
    df["finca"] = df["Finca"].map(lambda x: canonical_farm(x, aliases))
    df["bloque"] = df["Bloque"].map(canonical_block)
    df["cama_id"] = df["Cama"].map(canonical_block)
    df["variedad"] = df["Variedad"].astype(str).str.strip().str.upper()
    df["fecha_siembra"] = parse_date(df["Fecha Siembra"])
    df["fecha_erradicacion"] = parse_date(df["Fecha Erradicacion"])
    for col in ["Plantas", "Area Sembrada"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df[df["finca"].isin(target_farms) & df["variedad"].eq("FREEDOM")].copy()
    out = out.rename(columns={"Plantas": "plantas", "Area Sembrada": "area_sembrada"})
    out = out[["finca", "bloque", "cama_id", "variedad", "fecha_siembra",
               "fecha_erradicacion", "plantas", "area_sembrada", "Estado"]]
    out = out.rename(columns={"Estado": "estado"})
    if out.duplicated(["finca", "bloque", "cama_id"]).any():
        raise ValueError("plano tiene mas de una vigencia por cama; dividir vigencias")
    return out


def active_beds_by_date(beds: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    keys = beds[["finca", "bloque"]].drop_duplicates()
    grid = keys.assign(_key=1).merge(
        pd.DataFrame({"fecha": pd.Series(dates.dropna().unique())}).assign(_key=1),
        on="_key").drop(columns="_key")
    merged = grid.merge(beds, on=["finca", "bloque"], how="left")
    active = (merged["fecha_siembra"].le(merged["fecha"]) &
              (merged["fecha_erradicacion"].isna() |
               merged["fecha_erradicacion"].ge(merged["fecha"])))
    merged["_active"] = active
    merged.loc[~active, ["plantas", "area_sembrada"]] = np.nan
    out = (merged.groupby(["finca", "bloque", "fecha"], as_index=False)
           .agg(camas_activas=("_active", "sum"), plantas_activas=("plantas", "sum"),
                area_activa=("area_sembrada", "sum")))
    out["camas_activas"] = out["camas_activas"].astype("int64")
    return out


def load_pruning(raw: Path, aliases: dict, target_farms: list[str]) -> pd.DataFrame:
    df = pd.read_excel(raw / "Podas 10.xlsx")
    df["finca"] = df["Finca"].map(lambda x: canonical_farm(x, aliases))
    df["bloque"] = df["Block"].map(canonical_block)
    df["fecha"] = parse_date(df["Fecha"])
    df["variedad"] = df["Variedad"].astype(str).str.strip().str.upper()
    df["destino"] = df["Destino"].astype(str).str.strip().str.upper()
    df["cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)
    df = df[df["finca"].isin(target_farms) & df["variedad"].eq("FREEDOM")]
    out = (df.assign(poda_corte=np.where(df["destino"].eq("CORTE"), df["cantidad"], 0),
                     poda_alineamiento=np.where(df["destino"].eq("ALINEAMIENTO"), df["cantidad"], 0))
           .groupby(["finca", "bloque", "fecha"], as_index=False)
           .agg(poda_corte=("poda_corte", "sum"),
                poda_alineamiento=("poda_alineamiento", "sum")))
    out["poda_total"] = out["poda_corte"] + out["poda_alineamiento"]
    return out


def build_fact_bloque_dia(cfg: dict, raw: Path) -> pd.DataFrame:
    farms = cfg["project"]["target_farms"]
    aliases = cfg["farm_aliases"]
    counts = load_conteos(raw, aliases, farms)
    sampled = load_sampled_beds(raw, aliases, farms)
    beds = load_bed_validity(raw, aliases, farms)
    pruning = load_pruning(raw, aliases, farms)

    fact = counts.sort_values(["finca", "bloque", "fecha"]).copy()
    active = active_beds_by_date(beds, fact["fecha"])
    fact = fact.merge(active, on=["finca", "bloque", "fecha"], how="left", validate="one_to_one")
    fact = fact.merge(sampled, on=["finca", "bloque", "semana_iso"], how="left", validate="many_to_one")
    fact = fact.merge(pruning, on=["finca", "bloque", "fecha"], how="left", validate="one_to_one")
    for col in ["poda_corte", "poda_alineamiento", "poda_total"]:
        fact[col] = fact[col].fillna(0.0)
    fact["es_fecha_conteo"] = fact["conteo_total"].notna()
    fact["cobertura_muestreo"] = fact["camas_muestreadas"] / fact["camas_activas"].replace(0, np.nan)
    fact["factor_extrapolacion"] = (
        fact["camas_activas"] / fact["camas_muestreadas"].replace(0, np.nan)
    )
    fact.loc[fact["camas_muestreadas"] > fact["camas_activas"], "factor_extrapolacion"] = 1.0

    # Se transporta solo la procedencia del ultimo conteo, nunca el conteo sin metadata.
    origins = (fact[fact["es_fecha_conteo"]]
               .sort_values("fecha")
               .groupby(["finca", "bloque", "semana_iso"], as_index=False)
               .tail(1)
               [["finca", "bloque", "semana_iso", "fecha", "conteo_RC", "conteo_SS",
                 "conteo_AP", "conteo_CO", "conteo_total"]]
               .rename(columns={"fecha": "fecha_conteo_origen",
                                "conteo_RC": "conteo_RC_origen", "conteo_SS": "conteo_SS_origen",
                                "conteo_AP": "conteo_AP_origen", "conteo_CO": "conteo_CO_origen",
                                "conteo_total": "conteo_total_origen"}))
    fact = fact.merge(origins, on=["finca", "bloque", "semana_iso"], how="left", validate="many_to_one")
    fact["dias_desde_conteo"] = (fact["fecha"] - fact["fecha_conteo_origen"]).dt.days
    fact["semana_objetivo"] = fact["semana_iso"] + 1
    fact = fact[["finca", "bloque", "fecha", "semana_iso", "semana_objetivo",
                 "corte_comercial_real", "conteo_RC", "conteo_SS", "conteo_AP",
                 "conteo_CO", "conteo_total", "es_fecha_conteo", "fecha_conteo_origen",
                 "dias_desde_conteo", "conteo_RC_origen", "conteo_SS_origen",
                 "conteo_AP_origen", "conteo_CO_origen", "conteo_total_origen",
                 "camas_activas", "camas_muestreadas", "factor_extrapolacion",
                 "cobertura_muestreo", "plantas_activas", "area_activa", "poda_alineamiento",
                 "poda_corte", "poda_total"]]
    return fact.sort_values(["finca", "bloque", "fecha"]).reset_index(drop=True)


def build_forecast_windows(fact: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    origins = (fact[fact["es_fecha_conteo"]].sort_values("fecha")
               .groupby(["finca", "bloque", "semana_iso"], as_index=False).tail(1).copy())
    origins["fecha_inicio_objetivo"] = origins["fecha"].dt.to_period("W-SUN").dt.end_time.dt.normalize() + pd.Timedelta(days=1)
    origins["fecha_fin_objetivo"] = origins["fecha_inicio_objetivo"] + pd.Timedelta(days=horizon_days - 1)
    rows = []
    for _, row in origins.iterrows():
        dates = pd.date_range(row["fecha_inicio_objetivo"], periods=horizon_days, freq="D")
        for h, date in enumerate(dates, 1):
            rows.append({"finca": row["finca"], "bloque": row["bloque"],
                         "fecha_origen": row["fecha_conteo_origen"],
                         "semana_origen": row["semana_iso"],
                          "semana_objetivo": int(date.isocalendar().year * 100 + date.isocalendar().week),
                          "semana_proyeccion": int(date.isocalendar().year * 100 + date.isocalendar().week),
                         "fecha_objetivo": date, "horizonte_dia": h,
                         "conteo_RC_t0": row["conteo_RC_origen"],
                         "conteo_SS_t0": row["conteo_SS_origen"],
                         "conteo_AP_t0": row["conteo_AP_origen"],
                         "conteo_CO_t0": row["conteo_CO_origen"],
                         "conteo_total_t0": row["conteo_total_origen"],
                         "camas_muestreadas_t0": row["camas_muestreadas"],
                         "camas_activas_t0": row["camas_activas"],
                         "factor_extrapolacion_t0": row["factor_extrapolacion"],
                         "corte_real_dia": np.nan, "corte_real_semana": np.nan})
    windows = pd.DataFrame(rows)
    real = fact[["finca", "bloque", "fecha", "corte_comercial_real"]].rename(
        columns={"fecha": "fecha_objetivo", "corte_comercial_real": "corte_real_dia"})
    windows = windows.drop(columns="corte_real_dia").merge(
        real, on=["finca", "bloque", "fecha_objetivo"], how="left", validate="one_to_one")
    weekly = (windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)
              .agg(corte_real_semana=("corte_real_dia", "sum"), n_dias_reales=("corte_real_dia", "count")))
    windows = windows.drop(columns="corte_real_semana").merge(
        weekly, on=["finca", "bloque", "fecha_origen"], how="left", validate="many_to_one")
    windows["dias_reales_disponibles"] = windows["n_dias_reales"]
    windows["ventana_evaluable"] = windows["dias_reales_disponibles"].eq(horizon_days)
    window_state = np.select(
        [windows["dias_reales_disponibles"].eq(0), windows["ventana_evaluable"]],
        ["PENDIENTE_REAL", "VALIDA"], default="PARCIAL")
    windows["estado_ventana"] = window_state
    windows["motivo_no_evaluable"] = np.where(
        windows["ventana_evaluable"], None, "faltan_dias_reales_en_horizonte")
    return windows.drop(columns="n_dias_reales")


def build_dimensions(cfg: dict, raw: Path, fact: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construye las dimensiones maestras usadas por los datasets."""
    aliases = cfg["farm_aliases"]
    farms = cfg["project"]["target_farms"]
    farm_rows = []
    for source_name, source_value in aliases.items():
        if source_value in farms:
            farm_rows.append({"finca_id": source_value, "finca_canonica": source_value,
                              "nombre_fuente": source_name, "fuente": "farm_aliases"})
    dim_finca = pd.DataFrame(farm_rows).drop_duplicates().sort_values(
        ["finca_canonica", "nombre_fuente"])
    dim_bloque = fact[["finca", "bloque"]].drop_duplicates().rename(
        columns={"finca": "finca_id", "bloque": "bloque_id"})
    dim_bloque["bloque_raw"] = dim_bloque["bloque_id"]
    dim_bloque = dim_bloque[["finca_id", "bloque_id", "bloque_raw"]].sort_values(
        ["finca_id", "bloque_id"])
    dates = pd.date_range(fact["fecha"].min(), fact["fecha"].max(), freq="D")
    iso = dates.isocalendar()
    dim_fecha = pd.DataFrame({
        "fecha": dates,
        "anio": dates.year,
        "mes": dates.month,
        "semana_iso": iso.year * 100 + iso.week,
        "dia_semana": dates.dayofweek,
        "dia_anio": dates.dayofyear,
    })
    dim_fecha["inicio_semana"] = dim_fecha["fecha"] - pd.to_timedelta(
        dim_fecha["dia_semana"], unit="D")
    dim_fecha["fin_semana"] = dim_fecha["inicio_semana"] + pd.Timedelta(days=6)
    dim_cama = load_bed_validity(raw, aliases, farms).rename(
        columns={"finca": "finca_id", "bloque": "bloque_id"})
    return {"dim_finca": dim_finca.reset_index(drop=True),
            "dim_bloque": dim_bloque.reset_index(drop=True),
            "dim_fecha": dim_fecha,
            "dim_cama_vigencia": dim_cama.reset_index(drop=True)}


def build_qa_join_coverage(fact: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Resumen QA de cardinalidad y cobertura de la capa canónica."""
    candidates = windows[["finca", "bloque", "fecha_origen"]].drop_duplicates()
    evals = windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)[
        "ventana_evaluable"].first()
    result = (candidates.merge(evals, on=["finca", "bloque", "fecha_origen"],
                                validate="one_to_one")
              .assign(n_filas_fact=1))
    result["hay_muestreo_t0"] = result.merge(
        fact[["finca", "bloque", "fecha", "camas_muestreadas"]].rename(
            columns={"fecha": "fecha_origen"}),
        on=["finca", "bloque", "fecha_origen"], how="left")["camas_muestreadas"].notna().to_numpy()
    return result
