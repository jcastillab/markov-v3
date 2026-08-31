"""Fase 7: M3 Dirichlet-Multinomial y NB jerarquico."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import load_config
from evaluation.metrics import metrics
from models.bayes import DirichletM3, HierarchicalNB
from models.m3 import _period_for_date, fit_m3, simulate


def main():
    root = Path(__file__).resolve().parents[1]; cfg = load_config(root / "config" / "pipeline.yaml")
    out = root / cfg["paths"]["outputs"]; datasets, evaluation, models = [out / x for x in ("datasets", "evaluation", "models")]
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    intervals = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    frame = pd.read_parquet(datasets / "dataset_supervisado_diario.parquet")
    origins = (windows.groupby(["finca", "bloque", "fecha_origen"], as_index=False)
               .size().drop(columns="size").sort_values("fecha_origen"))
    n = max(cfg["evaluation"]["min_train_windows"], int(len(origins) * .60))
    valid_keys = set(map(tuple, origins.iloc[n:].itertuples(index=False, name=None)))
    valid = frame[[tuple(x) in valid_keys for x in zip(frame.finca, frame.bloque, frame.fecha_origen)]]
    train = frame[[tuple(x) not in valid_keys for x in zip(frame.finca, frame.bloque, frame.fecha_origen)]]
    # NB jerarquico: efectos finca-bloque-horizonte con pooling hacia la media global.
    nb = HierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"]).fit(train)
    pred_nb = nb.predict(valid)
    nb_metrics = metrics(valid.target, pd.Series(pred_nb))
    intervals_nb = nb.predictive_interval(valid, cfg["bayes"]["posterior_draws"], cfg["bayes"]["seed"])
    coverage80 = np.mean((valid.target.to_numpy() >= intervals_nb[0]) & (valid.target.to_numpy() <= intervals_nb[1]))
    coverage95 = np.mean((valid.target.to_numpy() >= intervals_nb[2]) & (valid.target.to_numpy() <= intervals_nb[3]))
    rows = [{"experiment_id": "NB_JERARQUICO", "split": "VALIDATION", "causal": True,
             "coverage_interval_80": coverage80, "coverage_interval_95": coverage95,
             "ancho_medio_intervalo": float(np.mean(intervals_nb[1] - intervals_nb[0])), **nb_metrics}]
    # Posterior Dirichlet por origen, con matriz M3 causal como centro del prior.
    pred, lows, highs, real = [], [], [], []
    for _, row in valid.iterrows():
        origin = pd.Timestamp(row.fecha_origen); period = _period_for_date(origin, cfg["m3"]["periods"])
        prior = fit_m3(intervals, row.finca, period, origin)
        data = intervals[(intervals.finca == row.finca) & (intervals.periodo == period) & (intervals.fecha <= origin)]
        posterior = DirichletM3(data, prior, cfg["bayes"]["dirichlet_prior_strength"], cfg["bayes"]["seed"])
        x0 = np.array([row.RC_t0, row.SS_t0, row.AP_t0]); lead = (pd.Timestamp(row.fecha_objetivo) - origin).days
        samples = []
        for _ in range(cfg["bayes"]["posterior_draws"]):
            q, r, loss = posterior.draw_matrix()
            from models.m3 import M3Matrix
            matrix = M3Matrix(row.finca, period, q, r, loss, prior.audit)
            samples.append(simulate(matrix, x0, lead, cfg["m3"]["baseline_ingress"]).iloc[-1].PC_dia_muestra * row.factor_extrapolacion)
        pred.append(float(np.mean(samples))); lows.append(float(np.quantile(samples, .1))); highs.append(float(np.quantile(samples, .9))); real.append(row.target)
    pred, lows, highs, real = map(np.asarray, (pred, lows, highs, real))
    d_metrics = metrics(pd.Series(real), pd.Series(pred))
    rows.append({"experiment_id": "M3_DIRICHLET_MULTINOMIAL", "split": "VALIDATION", "causal": True,
                 "coverage_interval_80": np.mean((real >= lows) & (real <= highs)),
                 "coverage_interval_95": np.nan, "ancho_medio_intervalo": float(np.mean(highs - lows)), **d_metrics})
    trace = valid[["finca", "bloque", "fecha_origen", "fecha_objetivo", "horizonte_dia"]].reset_index(drop=True)
    trace.assign(real=real, pred=pred, low80=lows, high80=highs).to_csv(
        evaluation / "predictions_m3_dirichlet.csv", index=False)
    trace.assign(real=valid.target.to_numpy(), pred=pred_nb,
                  low80=intervals_nb[0], high80=intervals_nb[1],
                  low95=intervals_nb[2], high95=intervals_nb[3]).to_csv(
        evaluation / "predictions_nb_jerarquico.csv", index=False)
    pd.DataFrame(rows).to_csv(evaluation / "metrics_fase7_bayes.csv", index=False)
    posterior.posterior_summary(cfg["bayes"]["posterior_draws"]).to_csv(models / "dirichlet_posterior_summary.csv", index=False)
    (models / "bayes_manifest.json").write_text(json.dumps({"phase": 7, "causal": True,
        "posterior_draws": cfg["bayes"]["posterior_draws"], "models": [r["experiment_id"] for r in rows]}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
