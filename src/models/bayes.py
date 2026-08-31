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
        self.events = {}
        for source in STATES:
            j = STATES.index(source)
            if source == "RC":
                event_names = ("STAY", "ADVANCE", "LOSS")
                prior = [m3.Q[j, j], m3.Q[STATES.index("SS"), j], m3.p[j]]
            elif source == "SS":
                event_names = ("STAY", "ADVANCE", "LOSS")
                prior = [m3.Q[j, j], m3.Q[STATES.index("AP"), j], m3.p[j]]
            else:
                event_names = ("STAY", "CUT", "LOSS")
                prior = [m3.Q[j, j], m3.r[j], m3.p[j]]
            prior = np.asarray(prior, float)
            prior = np.maximum(prior, 1e-6)
            prior = prior / prior.sum() * prior_strength
            data = intervals[intervals.estado_origen.eq(source)]
            counts = [int((data.evento == event).sum()) for event in event_names]
            self.events[source] = event_names
            self.alpha[source] = prior + np.asarray(counts, float)

    def draw_matrix(self):
        q, r, loss = np.zeros((3, 3)), np.zeros(3), np.zeros(3)
        for source in STATES:
            values = self.rng.dirichlet(self.alpha[source])
            j = STATES.index(source)
            if source == "AP":
                q[j, j], r[j], loss[j] = values
            else:
                q[j, j], advance, loss[j] = values
                destination = "SS" if source == "RC" else "AP"
                q[STATES.index(destination), j] = advance
        return q, r, loss

    def posterior_summary(self, draws=1000):
        values = {source: np.array([self.rng.dirichlet(self.alpha[source]) for _ in range(draws)])
                  for source in STATES}
        rows = []
        for source, samples in values.items():
            for event_i, event in enumerate(self.events[source]):
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


class CovariateHierarchicalNB:
    """NB empírico con offset de exposición y pooling progresivo.

    La tasa se regulariza global -> finca -> bloque -> horizonte. Las
    covariables ajustan log-linealmente esa tasa con un prior ridge; el
    intervalo predictivo agrega incertidumbre Gamma-Poisson.
    """

    def __init__(self, shrinkage=10.0, ridge=1.0, exposure="camas_activas"):
        self.shrinkage, self.ridge, self.exposure = shrinkage, ridge, exposure

    @staticmethod
    def _rates(frame, keys, exposure):
        grouped = frame.groupby(keys).agg(target=("target", "sum"), exposure=(exposure, "sum"))
        return {(key if isinstance(key, tuple) else (key,)): (float(row.target), float(row.exposure))
                for key, row in grouped.iterrows()}

    def fit(self, frame, features=()):
        data = frame.copy()
        data[self.exposure] = pd.to_numeric(data[self.exposure], errors="coerce").fillna(0).clip(lower=1)
        data["log_exposure"] = np.log(data[self.exposure])
        self.global_rate = float(data.target.sum() / data[self.exposure].sum())
        self.farm = self._rates(data, ["finca"], self.exposure)
        self.block = self._rates(data, ["finca", "bloque"], self.exposure)
        self.group = self._rates(data, ["finca", "bloque", "horizonte_dia"], self.exposure)
        self.features = [c for c in features if c in data and c not in {"target", self.exposure}]
        self.feature_mean = data[self.features].apply(pd.to_numeric, errors="coerce").mean() if self.features else pd.Series(dtype=float)
        self.feature_scale = data[self.features].apply(pd.to_numeric, errors="coerce").std().replace(0, 1) if self.features else pd.Series(dtype=float)
        baseline = self._baseline(data)
        if self.features:
            x = ((data[self.features].apply(pd.to_numeric, errors="coerce") - self.feature_mean) /
                 self.feature_scale).fillna(0).to_numpy(float)
            x = np.column_stack([np.ones(len(x)), x])
            y = np.log((data.target.to_numpy(float) + .5) / (baseline + .5))
            penalty = np.eye(x.shape[1]) * self.ridge; penalty[0, 0] = 0
            self.coef_ = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            fitted = baseline * np.exp(np.clip(x @ self.coef_, -3, 3))
        else:
            self.coef_ = np.zeros(1)
            fitted = baseline
        y = data.target.to_numpy(float)
        mu = np.maximum(fitted, 1e-6)
        self.alpha = max(float(np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2)), 0.01)
        return self

    def _baseline(self, frame):
        values = []
        for row in frame.itertuples(index=False):
            farm = self.farm.get((row.finca,), (0.0, 0.0))
            farm_rate = (farm[0] + self.shrinkage * self.global_rate) / (farm[1] + self.shrinkage)
            block = self.block.get((row.finca, row.bloque), (0.0, 0.0))
            block_rate = (block[0] + self.shrinkage * farm_rate) / (block[1] + self.shrinkage)
            group = self.group.get((row.finca, row.bloque, row.horizonte_dia), (0.0, 0.0))
            rate = (group[0] + self.shrinkage * block_rate) / (group[1] + self.shrinkage)
            values.append(rate * max(float(getattr(row, self.exposure)), 1.0))
        return np.asarray(values)

    def predict(self, frame):
        baseline = self._baseline(frame)
        if not self.features:
            return baseline
        data = frame.copy()
        data[self.exposure] = pd.to_numeric(data[self.exposure], errors="coerce").fillna(0).clip(lower=1)
        data["log_exposure"] = np.log(data[self.exposure])
        x = ((data[self.features].apply(pd.to_numeric, errors="coerce") - self.feature_mean) /
             self.feature_scale).fillna(0).to_numpy(float)
        x = np.column_stack([np.ones(len(x)), x])
        return baseline * np.exp(np.clip(x @ self.coef_, -3, 3))

    def predictive_interval(self, frame, draws=100, seed=42):
        mean = self.predict(frame)
        rng = np.random.default_rng(seed)
        shape = 1 / self.alpha
        lambdas = rng.gamma(shape, mean / shape, size=(draws, len(mean)))
        samples = rng.poisson(lambdas)
        return np.quantile(samples, [.1, .9, .025, .975], axis=0)
