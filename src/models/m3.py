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
    """Normaliza Abril/Junio/Julio a intervalos diarios consecutivos."""
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
                obs = long[["finca", "fecha", stem]].rename(columns={stem: "codigo_raw"})
                obs["codigo_raw"] = obs["codigo_raw"].map(_norm_code)
                obs["codigo_canonico"] = obs["codigo_raw"].map(
                    lambda x: aliases.get(x, x))
                obs["macroestado"] = obs["codigo_canonico"].map(mapping)
                obs = obs.sort_values("fecha")
                obs["fecha_siguiente"] = obs["fecha"].shift(-1)
                obs["codigo_siguiente"] = obs["codigo_canonico"].shift(-1)
                obs["macro_siguiente"] = obs["macroestado"].shift(-1)
                obs["delta_dias"] = (obs["fecha_siguiente"] - obs["fecha"]).dt.days
                for _, r in obs.iloc[:-1].iterrows():
                    source = r["macroestado"]
                    destination = r["macro_siguiente"]
                    if pd.isna(source) or pd.isna(destination) or r["delta_dias"] != 1:
                        continue
                    event = EVENTS.get(source, {}).get(destination)
                    if event is None:
                        continue
                    rows.append({"finca": r["finca"], "periodo": period,
                                 "fecha": r["fecha"], "estado_origen": source,
                                 "estado_destino": destination, "evento": event,
                                 "codigo_raw_origen": r["codigo_raw"],
                                 "codigo_raw_destino": r["codigo_siguiente"],
                                 "delta_dias": int(r["delta_dias"]),
                                 "fuente": filename, "hoja": sheet,
                                 "id_trayectoria": f"{sheet}|{r['finca']}|{stem}"})
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
    """Ajusta una matriz causal; usa fallback global si falta soporte."""
    data = intervals[(intervals["finca"] == finca) &
                     (intervals["periodo"] == periodo)]
    provenance = "FINCA_PERIODO"
    if max_date is not None:
        data = data[data["fecha"] <= max_date]
    if data.empty:
        data = intervals[(intervals["periodo"] == periodo) &
                         (intervals["fecha"] <= max_date if max_date is not None else True)]
        provenance = "PERIODO_GLOBAL_FALLBACK"
    if data.empty:
        data = intervals[intervals["fecha"] <= max_date] if max_date is not None else intervals
        provenance = "GLOBAL_FALLBACK"
    Q = np.zeros((3, 3), dtype=float)
    r = np.zeros(3, dtype=float)
    p = np.zeros(3, dtype=float)
    audit = []
    for j, source in enumerate(STATES):
        subset = data[data["estado_origen"] == source]
        counts = subset["evento"].value_counts().to_dict()
        n = int(len(subset))
        denom = max(n, 1)
        for dest, event in EVENTS[source].items():
            value = float(counts.get(event, 0)) / denom
            if dest in STATES:
                Q[STATES.index(dest), j] = value
            elif dest == "PC":
                r[j] = value
            elif dest == "L":
                p[j] = value
            audit.append({"finca": finca, "periodo": periodo,
                          "estado_origen": source, "estado_destino": dest,
                          "n_exposiciones": n, "n_eventos": int(counts.get(event, 0)),
                          "probabilidad": value, "procedencia": provenance,
                          "fecha_max_dato_modelo": max_date})
    # Empty rows are kept stochastic and auditable rather than inventing data.
    for j in range(3):
        if Q[:, j].sum() + r[j] + p[j] == 0:
            Q[j, j] = 1.0
            audit.append({"finca": finca, "periodo": periodo,
                          "estado_origen": STATES[j], "estado_destino": STATES[j],
                          "n_exposiciones": 0, "n_eventos": 0,
                          "probabilidad": 1.0, "procedencia": "IDENTIDAD_SIN_SOPORTE",
                          "fecha_max_dato_modelo": max_date})
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
