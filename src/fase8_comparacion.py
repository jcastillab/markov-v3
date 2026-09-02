"""Fase 8: ranking final, bootstrap y seleccion de champion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap(path, reps, seed):
    frame = pd.read_csv(path)
    y, p = frame["real"].to_numpy(float), frame["pred"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        ix = rng.integers(0, len(y), len(y))
        values.append(np.abs(p[ix] - y[ix]).sum() / max(np.abs(y[ix]).sum(), 1e-12))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def main():
    root = Path(__file__).resolve().parents[1]; cfg = __import__("canonical").load_config(root / "config/pipeline.yaml")
    evaluation = root / cfg["paths"]["outputs"] / "evaluation"
    files = ["metrics_fase2_m3.csv", "metrics_fase4_podas.csv", "metrics_fase5_clima.csv",
             "metrics_fase6_supervisado.csv", "metrics_fase7_bayes.csv",
             "metrics_rf_ablation_m3.csv"]
    frames = [pd.read_csv(evaluation / f) for f in files]
    all_metrics = pd.concat(frames, ignore_index=True, sort=False)
    causal = all_metrics[all_metrics.causal.astype(bool)].copy()
    causal["scope_comparable"] = causal.n.eq(714) & causal.split.eq("VALIDATION")
    causal["ranking_exclusion"] = np.where(causal.scope_comparable, "", "poblacion_o_escala_distinta")
    # Keep the best configured point for each experiment, eliminating grid duplicates.
    ranked = causal[causal.scope_comparable].sort_values("wape").drop_duplicates("experiment_id")
    pred_map = {"E00_M3_BASE": "E00_M3_BASE.csv",
                "RF_DIARIO_POOLED_FENO": "predictions_feno.csv",
                 "RF_H1_H7_FENO": "predictions_rf_h1_h7_feno.csv",
                 "RF_H1_H7_FENO_SIN_M3": "predictions_rf_h1_h7_feno_sin_m3.csv",
                 "RF_RESIDUAL_M3_FENO": "predictions_rf_residual_m3_feno.csv",
                 "NB_JERARQUICO": "predictions_nb_jerarquico.csv",
                 "NB_JERARQUICO_COVARIABLES": "predictions_nb_jerarquico_covariables.csv"}
    ci_low, ci_high = [], []
    for _, row in ranked.iterrows():
        path = pred_map.get(row.experiment_id)
        if path and (evaluation / path).exists():
            low, high = _bootstrap(evaluation / path, cfg["evaluation"]["bootstrap_reps"], cfg["bayes"]["seed"])
        else:
            low, high = np.nan, np.nan
        ci_low.append(low); ci_high.append(high)
    ranked["wape_ci95_low"], ranked["wape_ci95_high"] = ci_low, ci_high
    ranked["decision"] = "challenger"
    ranked.loc[ranked.experiment_id.eq("E00_M3_BASE"), "decision"] = "baseline_obligatorio"
    champion = ranked.iloc[0].experiment_id
    ranked.loc[ranked.experiment_id.eq(champion), "decision"] = "champion_provisional"
    ranked.to_csv(evaluation / "ranking_final.csv", index=False)
    all_metrics.to_csv(evaluation / "metrics_comparacion_final.csv", index=False)
    report = {"champion_provisional": champion, "ranking_population": "VALIDATION causal n=714 diario",
              "excluded_retrospective": int((~all_metrics.causal.astype(bool)).sum()),
              "excluded_noncomparable": int((~causal.scope_comparable).sum()),
              "bootstrap_reps": cfg["evaluation"]["bootstrap_reps"]}
    (evaluation / "champion_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(ranked[["experiment_id", "wape", "wape_ci95_low", "wape_ci95_high", "decision"]].to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
