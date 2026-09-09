"""Fase 7: M3 Dirichlet-Multinomial y NB jerarquico."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import load_config
from evaluation.metrics import metrics
from evaluation.split import validation_start
from models.bayes import CovariateHierarchicalNB, DirichletM3, HierarchicalNB
from models.m3 import _period_for_date, fit_m3, simulate


def weekly_intervals(trace, samples, model):
    """Agrega draws por ventana antes de calcular cuantiles semanales."""
    keys = ["finca", "bloque", "fecha_origen", "semana_proyeccion"]
    rows = []
    for key, positions in trace.groupby(keys, dropna=False).indices.items():
        positions = np.asarray(list(positions), dtype=int)
        total_draws = samples[:, positions].sum(axis=1)
        row = dict(zip(keys, key))
        row.update(model=model, real=float(trace.iloc[positions].real.sum()),
                   pred=float(total_draws.mean()), low80=float(np.quantile(total_draws, .1)),
                   high80=float(np.quantile(total_draws, .9)), low95=float(np.quantile(total_draws, .025)),
                   high95=float(np.quantile(total_draws, .975)), n_dias=len(positions))
        row["coverage80"] = float(row["low80"] <= row["real"] <= row["high80"])
        row["coverage95"] = float(row["low95"] <= row["real"] <= row["high95"])
        row["width80"] = row["high80"] - row["low80"]
        row["width95"] = row["high95"] - row["low95"]
        rows.append(row)
    return pd.DataFrame(rows)


def posterior_draws(trace, samples, model):
    """Persiste draws diarios para agregar intervalos sin sumar cuantiles."""
    keys = ["finca", "bloque", "fecha_origen", "semana_proyeccion"]
    base = trace[keys].reset_index(drop=True)
    draws, rows = samples.shape
    repeated = base.loc[base.index.repeat(draws)].reset_index(drop=True)
    repeated["model"] = model
    repeated["draw"] = np.tile(np.arange(draws), rows)
    repeated["sample"] = samples.T.reshape(-1)
    return repeated


def main():
    root = Path(__file__).resolve().parents[1]; cfg = load_config(root / "config" / "pipeline.yaml")
    out = root / cfg["paths"]["outputs"]; datasets, evaluation, models = [out / x for x in ("datasets", "evaluation", "models")]
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    fact = pd.read_parquet(datasets / "fact_bloque_dia.parquet")
    intervals = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    frame = pd.read_parquet(datasets / "dataset_supervisado_diario.parquet")
    frame["semana_del_año"] = pd.to_datetime(frame["fecha_origen"]).dt.isocalendar().week.astype(float)
    frame["periodo_ABRIL_JULIO"] = (pd.to_datetime(frame["fecha_origen"]).dt.month >= 7).astype(float)
    origin_dates = sorted(pd.to_datetime(windows["fecha_origen"]).unique())
    cutoff = validation_start(origin_dates, cfg)
    origin = pd.to_datetime(frame["fecha_origen"])
    objective = pd.to_datetime(frame["fecha_objetivo"])
    valid = frame[origin >= cutoff].copy()
    # Un objetivo posterior al cutoff no es observable al entrenar el modelo.
    train = frame[(origin < cutoff) & (objective < cutoff)].copy()
    # NB jerarquico: efectos finca-bloque-horizonte con pooling hacia la media global.
    nb = HierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"]).fit(train)
    pred_nb = nb.predict(valid)
    nb_metrics = metrics(valid.target, pd.Series(pred_nb))
    samples_nb = nb.predictive_samples(valid, cfg["bayes"]["posterior_draws"], cfg["bayes"]["seed"])
    intervals_nb = np.quantile(samples_nb, [.1, .9, .025, .975], axis=0)
    coverage80 = np.mean((valid.target.to_numpy() >= intervals_nb[0]) & (valid.target.to_numpy() <= intervals_nb[1]))
    coverage95 = np.mean((valid.target.to_numpy() >= intervals_nb[2]) & (valid.target.to_numpy() <= intervals_nb[3]))
    rows = [{"experiment_id": "NB_JERARQUICO", "split": "VALIDATION", "causal": True,
             "coverage_interval_80": coverage80, "coverage_interval_95": coverage95,
              "ancho_medio_intervalo": float(np.mean(intervals_nb[1] - intervals_nb[0])), **nb_metrics}]
    bayes_features = ["RC_t0", "SS_t0", "AP_t0", "p_RC", "p_SS", "p_AP", "TOTAL_t0",
                      "factor_extrapolacion", "corte_lag_1d", "corte_lag_2d", "corte_lag_3d",
                      "corte_sum_3d", "corte_sum_7d", "corte_sum_14d", "corte_mean_7d",
                      "log_exposure",
                      "horizonte_dia", "semana_del_año", "periodo_ABRIL_JULIO"]
    cov_nb = CovariateHierarchicalNB(cfg["bayes"]["hierarchical_shrinkage"],
                                     cfg["bayes"].get("covariate_ridge", 1.0)).fit(train, bayes_features)
    pred_cov = cov_nb.predict(valid)
    samples_cov = cov_nb.predictive_samples(valid, cfg["bayes"]["posterior_draws"], cfg["bayes"]["seed"])
    intervals_cov = np.quantile(samples_cov, [.1, .9, .025, .975], axis=0)
    cov_metrics = metrics(valid.target, pd.Series(pred_cov))
    rows.append({"experiment_id": "NB_JERARQUICO_COVARIABLES", "split": "VALIDATION", "causal": True,
                 "coverage_interval_80": np.mean((valid.target.to_numpy() >= intervals_cov[0]) &
                                                   (valid.target.to_numpy() <= intervals_cov[1])),
                 "coverage_interval_95": np.mean((valid.target.to_numpy() >= intervals_cov[2]) &
                                                   (valid.target.to_numpy() <= intervals_cov[3])),
                 "ancho_medio_intervalo": float(np.mean(intervals_cov[1] - intervals_cov[0])), **cov_metrics})
    # Posterior Dirichlet por origen, con matriz M3 causal como centro del prior.
    pred, lows, highs, lows95, highs95, real, summaries, dirichlet_samples = [], [], [], [], [], [], [], []
    for row_i, (_, row) in enumerate(valid.iterrows()):
        origin = pd.Timestamp(row.fecha_origen); period = _period_for_date(origin, cfg["m3"]["periods"])
        prior = fit_m3(intervals, row.finca, period, origin)
        data = intervals[(intervals.finca == row.finca) & (intervals.periodo == period) & (intervals.fecha <= origin)]
        posterior = DirichletM3(data, prior, cfg["bayes"]["dirichlet_prior_strength"],
                                cfg["bayes"]["seed"] + row_i)
        x0 = np.array([row.RC_t0, row.SS_t0, row.AP_t0]); lead = (pd.Timestamp(row.fecha_objetivo) - origin).days
        samples = []
        for _ in range(cfg["bayes"]["posterior_draws"]):
            q, r, loss = posterior.draw_matrix()
            from models.m3 import M3Matrix
            matrix = M3Matrix(row.finca, period, q, r, loss, prior.audit)
            samples.append(simulate(matrix, x0, lead, cfg["m3"]["baseline_ingress"]).iloc[-1].PC_dia_muestra * row.factor_extrapolacion)
        pred.append(float(np.mean(samples))); lows.append(float(np.quantile(samples, .1)))
        highs.append(float(np.quantile(samples, .9)))
        lows95.append(float(np.quantile(samples, .025)))
        highs95.append(float(np.quantile(samples, .975))); real.append(row.target)
        dirichlet_samples.append(samples)
        summary = posterior.posterior_summary(cfg["bayes"]["posterior_draws"])
        summary["finca"], summary["periodo"], summary["fecha_origen"] = row.finca, period, origin
        summaries.append(summary)
    pred, lows, highs, lows95, highs95, real = map(
        np.asarray, (pred, lows, highs, lows95, highs95, real))
    d_metrics = metrics(pd.Series(real), pd.Series(pred))
    rows.append({"experiment_id": "M3_DIRICHLET_MULTINOMIAL", "split": "VALIDATION", "causal": True,
                 "coverage_interval_80": np.mean((real >= lows) & (real <= highs)),
                  "coverage_interval_95": np.mean((real >= lows95) & (real <= highs95)),
                  "ancho_medio_intervalo": float(np.mean(highs - lows)), **d_metrics})
    trace = valid[["finca", "bloque", "fecha_origen", "fecha_objetivo", "semana_proyeccion", "horizonte_dia"]].reset_index(drop=True)
    trace.assign(real=real, pred=pred, low80=lows, high80=highs,
                 low95=lows95, high95=highs95).to_csv(
        evaluation / "predictions_m3_dirichlet.csv", index=False)
    trace.assign(real=valid.target.to_numpy(), pred=pred_nb,
                  low80=intervals_nb[0], high80=intervals_nb[1],
                  low95=intervals_nb[2], high95=intervals_nb[3]).to_csv(
        evaluation / "predictions_nb_jerarquico.csv", index=False)
    trace.assign(real=valid.target.to_numpy(), pred=pred_cov,
                 low80=intervals_cov[0], high80=intervals_cov[1],
                 low95=intervals_cov[2], high95=intervals_cov[3]).to_csv(
        evaluation / "predictions_nb_jerarquico_covariables.csv", index=False)
    weekly = pd.concat([
        weekly_intervals(trace.assign(real=valid.target.to_numpy()), samples_nb, "NB_JERARQUICO"),
        weekly_intervals(trace.assign(real=valid.target.to_numpy()), samples_cov, "NB_JERARQUICO_COVARIABLES"),
        weekly_intervals(trace.assign(real=real), np.column_stack(dirichlet_samples), "M3_DIRICHLET_MULTINOMIAL"),
    ], ignore_index=True)
    weekly.to_csv(evaluation / "bayes_weekly_intervals.csv", index=False)
    pd.concat([
        posterior_draws(trace, samples_nb, "NB_JERARQUICO"),
        posterior_draws(trace, samples_cov, "NB_JERARQUICO_COVARIABLES"),
        posterior_draws(trace, np.column_stack(dirichlet_samples), "M3_DIRICHLET_MULTINOMIAL"),
    ], ignore_index=True).to_csv(evaluation / "bayes_posterior_draws.csv", index=False)
    pd.DataFrame(rows).to_csv(evaluation / "metrics_fase7_bayes.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(models / "dirichlet_posterior_summary.csv", index=False)
    (models / "bayes_manifest.json").write_text(json.dumps({"phase": 7, "causal": True,
        "posterior_draws": cfg["bayes"]["posterior_draws"], "models": [r["experiment_id"] for r in rows]}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
