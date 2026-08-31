"""Modelos bayesianos conjugados y auditables de Fase 7."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from models.m3 import EVENTS, STATES
except ModuleNotFoundError:
    from src.models.m3 import EVENTS, STATES


class DirichletM3:
    """Posterior Dirichlet por fila de transición, centrado en M3."""

    def __init__(self, intervals, m3, prior_strength=5.0, seed=42):
        self.m3, self.rng = m3, np.random.default_rng(seed)
        self.alpha = {}
        for source in STATES:
            j = STATES.index(source)
            advance = 1 if source != "AP" else None
            prior = [m3.Q[j, j], m3.Q[advance, j] if advance is not None else 0.0,
                     m3.r[j], m3.p[j]]
            prior = np.asarray(prior, float)
            prior = np.maximum(prior, 1e-6)
            prior = prior / prior.sum() * prior_strength
            data = intervals[intervals.estado_origen.eq(source)]
            counts = [int((data.evento == "STAY").sum()),
                      int((data.evento == "ADVANCE").sum()),
                      int((data.evento == "CUT").sum()),
                      int((data.evento == "LOSS").sum())]
            self.alpha[source] = prior + np.asarray(counts, float)

    def draw_matrix(self):
        q, r, loss = np.zeros((3, 3)), np.zeros(3), np.zeros(3)
        for source in STATES:
            values = self.rng.dirichlet(self.alpha[source])
            j = STATES.index(source)
            q[j, j] = values[0] + (values[2] if source != "AP" else 0.0)
            if source == "AP":
                r[j] = values[1] + values[2]
            else:
                q[STATES.index("SS" if source == "RC" else "AP"), j] = values[1]
            loss[j] = values[3]
        return q, r, loss

    def posterior_summary(self, draws=1000):
        values = {source: np.array([self.rng.dirichlet(self.alpha[source]) for _ in range(draws)])
                  for source in STATES}
        rows = []
        for source, samples in values.items():
            for event_i, event in enumerate(("STAY", "ADVANCE", "CUT", "LOSS")):
                x = samples[:, event_i]
                rows.append({"estado_origen": source, "evento": event,
                             "media": x.mean(), "mediana": np.quantile(x, .5),
                             "intervalo_80_low": np.quantile(x, .10), "intervalo_80_high": np.quantile(x, .90),
                             "intervalo_95_low": np.quantile(x, .025), "intervalo_95_high": np.quantile(x, .975)})
        return pd.DataFrame(rows)


class HierarchicalNB:
    """Pooling conjugado Gamma-Poisson por finca, bloque y horizonte."""

    def __init__(self, shrinkage=10.0):
        self.shrinkage = shrinkage

    def fit(self, frame):
        self.global_mean = float(frame.target.mean())
        self.stats = (frame.groupby(["finca", "bloque", "horizonte_dia"])["target"]
                      .agg(["sum", "count"]).reset_index())
        return self

    def predict(self, frame):
        out = frame[["finca", "bloque", "horizonte_dia"]].merge(
            self.stats, on=["finca", "bloque", "horizonte_dia"], how="left")
        return ((out["sum"].fillna(0) + self.shrinkage * self.global_mean) /
                (out["count"].fillna(0) + self.shrinkage)).to_numpy()

    def predictive_interval(self, frame, draws=100, seed=42):
        mean = self.predict(frame)
        rng = np.random.default_rng(seed)
        shape = 1.0 / max(self.shrinkage / max(self.global_mean, 1e-6), 1e-6)
        rate_mean = np.maximum(mean, 0)
        lambdas = rng.gamma(shape, rate_mean / shape, size=(draws, len(mean)))
        samples = rng.poisson(lambdas)
        return np.quantile(samples, [.1, .9, .025, .975], axis=0)
