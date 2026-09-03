"""Baseline M3 por exposiciones observadas y grupos de vigencia.

La matriz se almacena con orientacion Q[destino, origen]. Las probabilidades
se calculan sobre exposiciones, no como promedio simple de trayectorias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("RC", "SS", "AP")
EVENTS = {
    "RC": {"RC": "STAY", "SS": "ADVANCE", "L": "LOSS"},
    "SS": {"SS": "STAY", "AP": "ADVANCE", "L": "LOSS"},
    "AP": {"AP": "STAY", "PC": "CUT", "L": "LOSS"},
}


def _norm_code(value: object) -> str:
    return " ".join(str(value).strip().upper().split())


def load_traditional_intervals(raw: Path, cfg: dict) -> pd.DataFrame:
    """Construye exposiciones diarias de M3_EXPOSICIONES_GRUPO.

    La fuente fenologica se observa por fechas de conteo, no necesariamente
    todos los dias. Por eso una trayectoria completa aporta ``duracion - 1``
    exposiciones STAY y una exposicion de salida (avance, corte o perdida).
    """
    mapping = {_norm_code(k): v for k, v in
               cfg["fenologia"]["micro_to_macro"].items()}
    aliases = {_norm_code(k): _norm_code(v) for k, v in
               cfg["fenologia"]["canonical_aliases"].items()}
    rows: list[dict] = []
    farm_aliases = {_norm_code(k): v for k, v in cfg["farm_aliases"].items()}
    for period_cfg in cfg["m3"]["periods"]:
        period = period_cfg["name"]
        filename = cfg["sources"][period_cfg["source_key"]]
        sheets = period_cfg["sheets"]
        xl = pd.ExcelFile(raw / filename)
        for sheet in sheets:
            if sheet not in xl.sheet_names:
                continue
            df = xl.parse(sheet)
            stem_cols = [c for c in df.columns if isinstance(c, int) or str(c).isdigit()]
            long = df[["FINCA", "FECHA"] + stem_cols].copy()
            long["fecha"] = pd.to_datetime(long["FECHA"], errors="coerce").dt.normalize()
            long["finca"] = long["FINCA"].map(_norm_code).map(farm_aliases).fillna(
                long["FINCA"].map(_norm_code))
            for stem in stem_cols:
                # Cada columna de tallo puede repetirse entre cohortes/fincas;
                # nunca se debe concatenar la trayectoria de dos cohortes.
                for cohort, cohort_rows in long.groupby(["FINCA", "finca"], dropna=False):
                    obs = cohort_rows[["finca", "fecha", stem]].rename(columns={stem: "codigo_raw"})
                    obs["codigo_raw"] = obs["codigo_raw"].map(_norm_code)
                    obs["codigo_canonico"] = obs["codigo_raw"].map(lambda x: aliases.get(x, x))
                    obs["macroestado"] = obs["codigo_canonico"].map(mapping)
                    obs = obs.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
                    if obs.empty:
                        continue
                    states = obs["macroestado"].tolist()
                    rc_index = next((i for i, value in enumerate(states) if value == "RC"), None)
                    terminal_index = next(
                        (i for i in range((rc_index or 0) + 1, len(states))
                         if states[i] in {"PC", "L"}), None)
                    if rc_index is None or terminal_index is None:
                        continue
                    duration = int((obs.loc[terminal_index, "fecha"] -
                                     obs.loc[rc_index, "fecha"]).days)
                    if duration <= 0:
                        continue
                    trajectory = f"{sheet}|{cohort[0]}|{stem}"
                    for state in STATES:
                        start = next((i for i in range(rc_index, terminal_index + 1)
                                      if states[i] == state), None)
                        if start is None:
                            continue
                        destinations = set(STATES[STATES.index(state) + 1:]) | {"PC", "L"}
                        exit_index = next((i for i in range(start + 1, terminal_index + 1)
                                           if states[i] in destinations), None)
                        if exit_index is None:
                            continue
                        state_duration = int((obs.loc[exit_index, "fecha"] -
                                              obs.loc[start, "fecha"]).days)
                        if state_duration <= 0:
                            continue
                        source = state
                        destination = states[exit_index]
                        event = EVENTS[source].get(destination)
                        if event is None:
                            continue
                        source_raw = obs.loc[start, "codigo_raw"]
                        destination_raw = obs.loc[exit_index, "codigo_raw"]
                        for offset in range(state_duration):
                            is_exit = offset == state_duration - 1
                            rows.append({
                                "finca": obs.loc[start, "finca"], "periodo": period,
                                "fecha": obs.loc[start, "fecha"] + pd.Timedelta(days=offset),
                                "estado_origen": source,
                                "estado_destino": destination if is_exit else source,
                                "evento": event if is_exit else "STAY",
                                "codigo_raw_origen": source_raw,
                                "codigo_raw_destino": destination_raw if is_exit else source_raw,
                                "delta_dias": 1, "duracion_grupo": duration,
                                "tallo": str(stem), "fuente": filename, "hoja": sheet,
                                "id_trayectoria": trajectory,
                            })
    return pd.DataFrame(rows)


def _period_for_date(date: pd.Timestamp, periods: list[dict]) -> str:
    for period in periods:
        start = pd.Timestamp(period["fecha_inicio_vigencia"])
        end = pd.Timestamp(period["fecha_fin_vigencia"])
        if start <= date <= end:
            return period["name"]
    return periods[-1]["name"]


@dataclass(frozen=True)
class M3Matrix:
    finca: str
    periodo: str
    Q: np.ndarray
    r: np.ndarray
    p: np.ndarray
    audit: pd.DataFrame


def fit_m3(intervals: pd.DataFrame, finca: str, periodo: str,
           max_date: pd.Timestamp | None = None) -> M3Matrix:
    """Ajusta M3 por grupos de duración, con fallback por estado."""
    data = intervals[(intervals["finca"] == finca) & (intervals["periodo"] == periodo)].copy()
    global_data = intervals[intervals["periodo"] == periodo].copy()
    historical_data = intervals.copy()
    if max_date is not None:
        data = data[data["fecha"] <= max_date]
        global_data = global_data[global_data["fecha"] <= max_date]
        historical_data = historical_data[historical_data["fecha"] <= max_date]
    if global_data.empty:
        global_data = historical_data.copy()
    if data.empty:
        data = global_data.copy()
    groups = data["duracion_grupo"].dropna().unique().tolist() if "duracion_grupo" in data else []
    if not groups:
        groups = [None]
    weights = []
    for group in groups:
        subset = data if group is None else data[data["duracion_grupo"].eq(group)]
        weights.append(float(subset["id_trayectoria"].nunique()) if "id_trayectoria" in subset else float(len(subset)))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() if weights.sum() else np.ones(len(groups)) / len(groups)
    Q = np.zeros((3, 3), dtype=float); r = np.zeros(3, dtype=float); p = np.zeros(3, dtype=float)
    audit = []
    for group, weight in zip(groups, weights):
        subset_group = data if group is None else data[data["duracion_grupo"].eq(group)]
        for j, source in enumerate(STATES):
            subset = subset_group[subset_group["estado_origen"].eq(source)]
            provenance = "GRUPO_DURACION"
            if subset.empty:
                subset = global_data[global_data["estado_origen"].eq(source)]
                provenance = "GLOBAL_PERIODO_ESTADO_FALLBACK"
            if subset.empty:
                subset = historical_data[historical_data["estado_origen"].eq(source)]
                provenance = "GLOBAL_HISTORICO_ESTADO_FALLBACK"
            counts = subset["evento"].value_counts().to_dict()
            n = int(len(subset))
            if n == 0:
                raise ValueError(f"Sin exposiciones para {periodo}/{source}, incluso en fallback global")
            for dest, event in EVENTS[source].items():
                value = float(counts.get(event, 0)) / n
                if dest in STATES: Q[STATES.index(dest), j] += weight * value
                elif dest == "PC": r[j] += weight * value
                elif dest == "L": p[j] += weight * value
                audit.append({"finca": finca, "periodo": periodo, "duracion_grupo": group,
                              "estado_origen": source, "estado_destino": dest,
                              "n_exposiciones": n, "n_eventos": int(counts.get(event, 0)),
                              "probabilidad": value, "peso_grupo": weight,
                              "procedencia": provenance, "fecha_max_dato_modelo": max_date})
    sums = Q.sum(axis=0) + r + p
    if not np.allclose(sums, np.ones(3), atol=1e-10):
        raise ValueError(f"Las columnas de M3 no suman 1: {sums}")
    return M3Matrix(finca, periodo, Q, r, p, pd.DataFrame(audit))


def simulate(matrix: M3Matrix, x0: np.ndarray, days: int,
             alpha_ingreso_rc: float) -> pd.DataFrame:
    """Propaga x_(t+1)=Qx_t+ingreso y devuelve cortes diarios muestrales."""
    x = np.asarray(x0, dtype=float).copy()
    rows = []
    for day in range(1, days + 1):
        cut = float(matrix.r @ x)
        loss = float(matrix.p @ x)
        x = matrix.Q @ x
        x[0] += alpha_ingreso_rc * x0[0]
        rows.append({"dia_simulado": day, "RC": x[0], "SS": x[1], "AP": x[2],
                     "PC_dia_muestra": cut, "L_dia_muestra": loss,
                     "masa_activa": x.sum()})
    return pd.DataFrame(rows)
