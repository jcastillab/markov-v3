"""Ingesta P32 y matrices empiricas de transicion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .m3 import EVENTS, M3Matrix, STATES, _norm_code
except ImportError:  # ejecucion directa desde src/
    from models.m3 import EVENTS, M3Matrix, STATES, _norm_code


def _code_maps(cfg: dict) -> tuple[dict, dict, set[str]]:
    micro_to_macro = {_norm_code(k): v for k, v in cfg["fenologia"]["micro_to_macro"].items()}
    aliases = {_norm_code(k): _norm_code(v) for k, v in cfg["fenologia"]["canonical_aliases"].items()}
    pending = {_norm_code(x) for x in cfg["fenologia"]["codes_pending"]}
    pending.add("SP")
    return micro_to_macro, aliases, pending


def _normalize_observation(raw: object, maps: tuple[dict, dict, set[str]]) -> tuple[str | None, str | None, bool, str | None]:
    micro_to_macro, aliases, pending = maps
    raw_code = _norm_code(raw)
    canonical = aliases.get(raw_code, raw_code)
    if raw_code in pending or canonical in pending:
        if raw_code == "SP":
            return "SP_SIN_FRACCION", "AP", False, "CODIGO_PENDIENTE_SP"
        return canonical, micro_to_macro.get(canonical), False, "CODIGO_PENDIENTE"
    macro = micro_to_macro.get(canonical)
    if macro is None:
        return canonical, None, False, "CODIGO_DESCONOCIDO"
    return canonical, macro, True, None


def _date_columns(raw: pd.DataFrame, sheet: str) -> list[tuple[int, pd.Timestamp, int | None]]:
    result = []
    for col in range(raw.shape[1]):
        value = raw.iat[0, col]
        date = pd.to_datetime(value, errors="coerce")
        if pd.isna(date):
            continue
        size_col = col + 1 if sheet == "Garbanzo" and col + 1 < raw.shape[1] else None
        result.append((col, pd.Timestamp(date).normalize(), size_col))
    return result


def load_p32_observations(raw_path: Path, cfg: dict) -> pd.DataFrame:
    """Convierte las cuatro hojas P32 de formato ancho a formato largo."""
    maps = _code_maps(cfg)
    rows: list[dict] = []
    xl = pd.ExcelFile(raw_path)
    for sheet in ["Garbanzo", "Rayando 1", "Separando S", "Definiendo P"]:
        raw = xl.parse(sheet, header=None)
        date_cols = _date_columns(raw, sheet)
        label_col, initial_state_col, bed_col = 0, 1, 3 if sheet == "Garbanzo" else 2
        first_data = next((i for i in range(1, len(raw)) if pd.notna(raw.iat[i, label_col])), len(raw))
        for i in range(first_data, len(raw)):
            label = raw.iat[i, label_col]
            if pd.isna(label):
                continue
            for col, date, size_col in date_cols:
                raw_value = raw.iat[i, col]
                if pd.isna(raw_value) or str(raw_value).strip() == "":
                    continue
                micro, macro, valid, reason = _normalize_observation(raw_value, maps)
                size = raw.iat[i, size_col] if size_col is not None else np.nan
                rows.append({"grupo": sheet, "etiqueta": str(label).strip(),
                             "cama": str(raw.iat[i, bed_col]).strip() if pd.notna(raw.iat[i, bed_col]) else None,
                             "fecha": date, "estado_raw": str(raw_value).strip(),
                             "estado_micro": micro, "estado_macro": macro,
                             "tamano_mm": pd.to_numeric(size, errors="coerce"),
                             "valido_codigo": valid, "motivo_codigo": reason})
    out = pd.DataFrame(rows).sort_values(["grupo", "etiqueta", "fecha"]).reset_index(drop=True)
    return out


def build_p32_intervals(observations: pd.DataFrame) -> pd.DataFrame:
    """Crea pares de fechas consecutivas y clasifica exclusiones."""
    rows: list[dict] = []
    for (group, label), track in observations.groupby(["grupo", "etiqueta"], sort=False):
        track = track.sort_values("fecha")
        for left, right in zip(track.iloc[:-1].to_dict("records"), track.iloc[1:].to_dict("records")):
            delta = (right["fecha"] - left["fecha"]).days
            event = EVENTS.get(left["estado_macro"], {}).get(right["estado_macro"])
            if delta != 1:
                reason = "GAP_NO_DIARIO"
            elif not left["valido_codigo"] or not right["valido_codigo"]:
                left_reason = left["motivo_codigo"] if pd.notna(left["motivo_codigo"]) else None
                right_reason = right["motivo_codigo"] if pd.notna(right["motivo_codigo"]) else None
                reason = left_reason or right_reason or "CODIGO_PENDIENTE"
            elif event is None:
                reason = "TRANSICION_NO_CANONICA"
            else:
                reason = None
            rows.append({"grupo": group, "etiqueta": label, "cama": left["cama"],
                         "fecha_t": left["fecha"], "fecha_t1": right["fecha"],
                         "estado_raw_t": left["estado_raw"], "estado_raw_t1": right["estado_raw"],
                         "estado_micro_t": left["estado_micro"], "estado_micro_t1": right["estado_micro"],
                         "estado_macro_t": left["estado_macro"], "estado_macro_t1": right["estado_macro"],
                         "tamano_mm_t": left["tamano_mm"], "tamano_mm_t1": right["tamano_mm"],
                         "delta_dias": delta, "evento": event, "valido": reason is None,
                         "motivo_exclusion": reason})
    return pd.DataFrame(rows)


def _counts(intervals: pd.DataFrame, max_date: pd.Timestamp | None = None) -> dict[tuple[str, str], int]:
    data = intervals[intervals["valido"]].copy()
    if max_date is not None:
        data = data[data["fecha_t"] <= max_date]
    return {(state, event): int(((data["estado_macro_t"] == state) & (data["evento"] == event)).sum())
            for state in STATES for event in set(EVENTS[state].values())}


def fit_p32_matrix(intervals: pd.DataFrame, prior: M3Matrix | None = None,
                   tau: float = 0.0, max_date: pd.Timestamp | None = None) -> M3Matrix:
    """Matriz P32 pooled; REG aplica shrinkage hacia M3 con tau."""
    counts = _counts(intervals, max_date)
    Q = np.zeros((3, 3), dtype=float)
    r = np.zeros(3, dtype=float)
    p = np.zeros(3, dtype=float)
    audit = []
    for j, state in enumerate(STATES):
        events = list(EVENTS[state].items())
        n = sum(counts[(state, event)] for _, event in events)
        denom = n + tau
        for dest, event in events:
            prior_prob = 0.0
            if prior is not None:
                if dest in STATES:
                    prior_prob = prior.Q[STATES.index(dest), j]
                elif dest == "PC":
                    prior_prob = prior.r[j]
                else:
                    prior_prob = prior.p[j]
            prob = ((counts[(state, event)] + tau * prior_prob) / denom
                    if denom else 1.0 / len(events))
            if dest in STATES:
                Q[STATES.index(dest), j] = prob
            elif dest == "PC":
                r[j] = prob
            else:
                p[j] = prob
            audit.append({"finca": "P32_POOLED", "periodo": "P32",
                          "estado_origen": state, "estado_destino": dest,
                          "n_exposiciones": n, "n_eventos": counts[(state, event)],
                          "probabilidad": prob, "procedencia": "P32_REG" if tau else "P32_RAW",
                          "tau": tau, "fecha_max_dato_modelo": max_date})
    return M3Matrix("P32_POOLED", "P32", Q, r, p, pd.DataFrame(audit))
