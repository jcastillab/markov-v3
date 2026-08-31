"""Semi Markov corregido: masa por estado y bin de edad."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .m3 import EVENTS, STATES, M3Matrix
except ImportError:  # ejecucion directa desde src/
    from models.m3 import EVENTS, STATES, M3Matrix


def estimate_age_distributions(intervals: pd.DataFrame, max_bin: int = 3) -> dict[str, np.ndarray]:
    """Estima P(edad|estado) sin concentrar todo x0 en edad cero.

    La edad es la duracion observada dentro del episodio P32; un episodio que
    inicia en mitad del estado es tratado como censura izquierda y no inventa
    su edad previa. Por eso el estimador usa solo exposiciones consecutivas.
    """
    valid = intervals[intervals["valido"]].sort_values(["grupo", "etiqueta", "fecha_t"])
    values: dict[str, list[int]] = {state: [] for state in STATES}
    for (_, _), track in valid.groupby(["grupo", "etiqueta"]):
        run_state = None
        run_age = 0
        for row in track.itertuples():
            state = row.estado_macro_t
            if state != run_state:
                run_state, run_age = state, 0
            values[state].append(min(run_age, max_bin))
            run_age += 1
    result = {}
    for state in STATES:
        counts = np.bincount(values[state], minlength=max_bin + 1).astype(float)
        result[state] = counts / counts.sum() if counts.sum() else np.ones(max_bin + 1) / (max_bin + 1)
    return result


def _with_age_bins(intervals: pd.DataFrame, max_bin: int) -> pd.DataFrame:
    data = intervals[intervals["valido"]].sort_values(["grupo", "etiqueta", "fecha_t"]).copy()
    age_bins = []
    previous = {}
    for row in data.itertuples():
        key = (row.grupo, row.etiqueta)
        age = previous.get(key, (None, -1))
        current = age[1] + 1 if age[0] == row.estado_macro_t else 0
        previous[key] = (row.estado_macro_t, current)
        age_bins.append(min(current, max_bin))
    data["bin_edad"] = age_bins
    return data


def competitive_hazards(intervals: pd.DataFrame, m3: M3Matrix,
                        max_bin: int = 3, tau: float = 1.0,
                        tau_loss: float = 10.0) -> dict[tuple[str, int], dict[str, float]]:
    """Estima hazards; la suma STAY/ADVANCE/CUT/LOSS siempre es uno."""
    valid = _with_age_bins(intervals, max_bin)
    result = {}
    for j, state in enumerate(STATES):
        state_data = valid[valid["estado_macro_t"] == state]
        loss_n = int((state_data["evento"] == "LOSS").sum())
        loss = (loss_n + tau_loss * m3.p[j]) / (len(state_data) + tau_loss)
        nonloss = state_data[state_data["evento"] != "LOSS"]
        for age in range(max_bin + 1):
            # Age bins are estimated from the ordered exposure; sparse bins use M3.
            data = nonloss[nonloss["bin_edad"] == age]
            if data.empty:
                data = nonloss
            counts = {event: int((data["evento"] == event).sum()) for event in set(EVENTS[state].values()) if event != "LOSS"}
            base = {event: (m3.Q[STATES.index(dest), j] if dest in STATES else m3.r[j])
                    for dest, event in EVENTS[state].items() if event != "LOSS"}
            base_total = sum(base.values())
            if base_total:
                base = {event: value / base_total for event, value in base.items()}
            denom = sum(counts.values()) + tau
            if denom == 0:
                conditional = base.copy()
            else:
                conditional = {event: (counts.get(event, 0) + tau * base[event]) / denom
                               for event in base}
            hazards = {"LOSS": float(loss)}
            for event, prob in conditional.items():
                hazards[event] = float((1.0 - loss) * prob)
            total = sum(hazards.values())
            result[(state, age)] = {key: value / total for key, value in hazards.items()}
    return result


class SemiMarkov:
    def __init__(self, hazards, age_distributions, max_bin: int = 3):
        self.hazards = hazards
        self.age_distributions = age_distributions
        self.max_bin = max_bin

    def initial_mass(self, x0: np.ndarray) -> dict[tuple[str, int], float]:
        mass = {}
        for j, state in enumerate(STATES):
            for age, probability in enumerate(self.age_distributions[state]):
                mass[(state, age)] = float(x0[j] * probability)
        return mass

    def simulate(self, x0: np.ndarray, days: int, alpha_ingreso_rc: float) -> pd.DataFrame:
        mass = self.initial_mass(x0)
        rows = []
        for day in range(1, days + 1):
            next_mass = {(state, age): 0.0 for state in STATES for age in range(self.max_bin + 1)}
            cut = 0.0
            loss = 0.0
            for (state, age), amount in mass.items():
                hazards = self.hazards[(state, age)]
                stay = hazards.get("STAY", 0.0)
                next_mass[(state, min(age + 1, self.max_bin))] += amount * stay
                if state == "RC":
                    next_mass[("SS", 0)] += amount * hazards.get("ADVANCE", 0.0)
                elif state == "SS":
                    next_mass[("AP", 0)] += amount * hazards.get("ADVANCE", 0.0)
                else:
                    cut += amount * hazards.get("CUT", 0.0)
                loss += amount * hazards.get("LOSS", 0.0)
            next_mass[("RC", 0)] += alpha_ingreso_rc * x0[0]
            mass = next_mass
            rows.append({"dia_simulado": day, "RC": sum(mass[("RC", a)] for a in range(self.max_bin + 1)),
                         "SS": sum(mass[("SS", a)] for a in range(self.max_bin + 1)),
                         "AP": sum(mass[("AP", a)] for a in range(self.max_bin + 1)),
                         "PC_dia_muestra": cut, "L_dia_muestra": loss,
                         "masa_activa": sum(mass.values())})
        return pd.DataFrame(rows)
