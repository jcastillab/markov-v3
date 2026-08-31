"""Normalizacion climatica y features causales de Fase 5."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _column(df: pd.DataFrame, fragment: str) -> str:
    matches = [c for c in df.columns if fragment.lower() in str(c).lower()]
    if not matches:
        raise KeyError(f"No se encontro columna climatica: {fragment}")
    return matches[0]


def load_hourly_climate(raw, files: list[str], station_id: int) -> tuple[pd.DataFrame, dict]:
    frames = [pd.read_excel(raw / filename) for filename in files]
    df = pd.concat(frames, ignore_index=True)
    station_col = _column(df, "IdEstacion")
    date_col = _column(df, "FechaHora")
    df = df[pd.to_numeric(df[station_col], errors="coerce").eq(station_id)].copy()
    df["fecha_hora"] = pd.to_datetime(df[date_col], dayfirst=True,
                                       format="mixed", errors="coerce")
    before = len(df)
    df = df.dropna(subset=["fecha_hora"])
    invalid = before - len(df)
    duplicates = int(df.duplicated("fecha_hora").sum())
    df = df.sort_values("fecha_hora").drop_duplicates("fecha_hora", keep="first")
    qa = {"filas_fuente": before, "filas_invalidas_fecha": invalid,
          "duplicados_eliminados": duplicates}
    return df, qa


def build_daily_climate(hourly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h = hourly.copy()
    h["fecha"] = h["fecha_hora"].dt.normalize()
    names = {"rain": _column(h, "Lluvia"), "temp": _column(h, "Temperatura ("),
             "tmax": _column(h, "TemperaturaMaxima"), "tmin": _column(h, "TemperaturaMinima"),
             "rh": _column(h, "Humedad"), "wind": _column(h, "Vel_Viento"),
             "et0": _column(h, "Evapotranspiracion"), "dew": _column(h, "Punto_Rocio"),
             "uv": _column(h, "UV"), "rad": _column(h, "Radiacion_Solar"),
             "pressure": _column(h, "Presion Atmosferica")}
    for key, col in names.items():
        h[key] = pd.to_numeric(h[col], errors="coerce")
    sat = 0.6108 * np.exp(17.27 * h["temp"] / (h["temp"] + 237.3))
    h["vpd_kpa"] = sat * (1.0 - h["rh"] / 100.0)
    h["daylight"] = h["rad"].fillna(0).gt(0)
    h = h.sort_values("fecha_hora")
    next_time = h["fecha_hora"].shift(-1)
    seconds = (next_time - h["fecha_hora"]).dt.total_seconds()
    typical = seconds[(seconds > 0) & (seconds <= 7200)].median()
    h["seconds_interval"] = seconds.where((seconds > 0) & (seconds <= 7200), typical).fillna(3600)
    h["solar_mj"] = h["rad"] * h["seconds_interval"] / 1e6
    h["dli_par_est"] = h["solar_mj"] * 1e6 * float(cfg["climate"]["par_fraction_of_shortwave"]) \
        * float(cfg["climate"]["umol_par_per_joule"]) / 1e6
    grouped = h.groupby("fecha")
    daily = grouped.agg(temp_mean_C=("temp", "mean"), temp_min_C=("temp", "min"),
        temp_max_C=("temp", "max"), rh_mean_pct=("rh", "mean"), rh_min_pct=("rh", "min"),
        rh_max_pct=("rh", "max"), rain_mm=("rain", "sum"), rain_hours=("rain", lambda x: x.gt(0).sum()),
        wind_mean_kmh=("wind", "mean"), wind_max_kmh=("wind", "max"), et0_mm=("et0", "sum"),
        dewpoint_mean_C=("dew", "mean"), uv_mean=("uv", "mean"), uv_max=("uv", "max"),
        radiation_mean_W_m2=("rad", "mean"), radiation_max_W_m2=("rad", "max"),
        pressure_mean_hPa=("pressure", "mean"), vpd_mean_kPa=("vpd_kpa", "mean"),
        vpd_max_kPa=("vpd_kpa", "max"), vpd_min_kPa=("vpd_kpa", "min"),
        vpd_daylight_mean_kPa=("vpd_kpa", lambda x: x[h.loc[x.index, "daylight"]].mean()),
        hours_vpd_gt_1=("vpd_kpa", lambda x: x.gt(1).sum()),
        hours_vpd_gt_1_5=("vpd_kpa", lambda x: x.gt(1.5).sum()),
        hours_vpd_lt_0_3=("vpd_kpa", lambda x: x.lt(0.3).sum()),
        solar_MJ_m2_d=("solar_mj", "sum"), DLI_PAR_est_mol_m2_d=("dli_par_est", "sum"))
    daily = daily.reset_index()
    for base in cfg["climate"]["gdd_bases"]:
        label = str(base).replace(".", "_")
        daily[f"gdd{label}_daily"] = ((daily.temp_max_C + daily.temp_min_C) / 2 - base).clip(lower=0)
    daily["clima_fuente"] = cfg["climate"]["fuente"]
    daily["microclima_invernadero_disponible"] = cfg["climate"]["microclima_invernadero_disponible"]
    return daily


def build_climate_features(daily: pd.DataFrame, origins: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = daily.set_index("fecha").sort_index()
    windows = cfg["climate"]["causal_windows_days"]
    rows = []
    for _, origin in origins.iterrows():
        date = pd.Timestamp(origin["fecha_origen"]).normalize()
        hist = d.loc[:date]
        row = {"finca": origin["finca"], "bloque": origin["bloque"], "fecha_origen": date}
        for width in windows:
            part = hist.tail(width)
            for col, label in (("temp_mean_C", "temp_mean_C"),
                               ("vpd_mean_kPa", "vpd_mean_kPa"),
                               ("DLI_PAR_est_mol_m2_d", "DLI_PAR_est_mol_m2_d"),
                               ("gdd5_0_daily", "gdd5_0")):
                row[f"{label}_mean_{width}d"] = part[col].mean()
            row[f"rain_sum_{width}d"] = part["rain_mm"].sum()
            row[f"gdd5_0_sum_{width}d"] = part["gdd5_0_daily"].sum()
            row[f"gdd5_2_sum_{width}d"] = part["gdd5_2_daily"].sum()
            row[f"DLI_sum_{width}d"] = part["DLI_PAR_est_mol_m2_d"].sum()
            row[f"et0_sum_{width}d"] = part["et0_mm"].sum()
        row["gdd5_0_7d_minus_28d_mean"] = row["gdd5_0_mean_7d"] - row["gdd5_0_mean_28d"]
        row["DLI_x_temp_7d"] = row["DLI_sum_7d"] * row["temp_mean_C_mean_7d"]
        rows.append(row)
    return pd.DataFrame(rows)
