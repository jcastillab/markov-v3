"""Fase 8: ranking final, bootstrap y seleccion de champion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap(path, reps, seed):
    frame = pd.read_csv(path)
    pred_column = next(column for column in ("pred", "proyectado", "pred_bloque") if column in frame)
    valid = frame["real"].notna() & frame[pred_column].notna()
    frame = frame.loc[valid].reset_index(drop=True)
    cluster_columns = [column for column in ("finca", "bloque", "fecha_origen") if column in frame]
    clusters = [indices.to_numpy() for _, indices in frame.groupby(cluster_columns).groups.items()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        sampled = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        y = frame.loc[indices, "real"].to_numpy(float)
        p = frame.loc[indices, pred_column].to_numpy(float)
        values.append(np.abs(p - y).sum() / max(np.abs(y).sum(), 1e-12))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def main():
    root = Path(__file__).resolve().parents[1]; cfg = __import__("canonical").load_config(root / "config/pipeline.yaml")
    evaluation = root / cfg["paths"]["outputs"] / "evaluation"
    files = ["metrics_fase2_m3.csv", "metrics_fase3_p32.csv", "metrics_fase4_podas.csv", "metrics_fase5_clima.csv",
              "metrics_fase6_supervisado.csv", "metrics_fase7_bayes.csv",
              "metrics_rf_ablation_m3.csv", "metrics_rolling_origin.csv"]
    frames = [pd.read_csv(evaluation / f) for f in files if (evaluation / f).exists()]
    all_metrics = pd.concat(frames, ignore_index=True, sort=False)
    causal_flag = all_metrics["causal"].astype(str).str.lower().eq("true")
    causal = all_metrics[causal_flag].copy()
    champion_split = cfg["evaluation"]["champion_split"]
    target = causal[causal.split.eq(champion_split)]
    if target.empty:
        raise ValueError(f"No hay metricas causales para el split champion {champion_split}")
    population_n = int(target["n"].min())
    if not target["n"].eq(population_n).all():
        raise ValueError("Los modelos rolling no comparten exactamente el mismo tamano de poblacion")
    causal["scope_comparable"] = causal.n.eq(population_n) & causal.split.eq(champion_split)
    causal["ranking_exclusion"] = np.where(causal.scope_comparable, "", "poblacion_o_escala_distinta")
    # Keep the best configured point for each experiment, eliminating grid duplicates.
    ranked = causal[causal.scope_comparable].sort_values("wape").drop_duplicates("experiment_id")
    ci_low, ci_high = [], []
    for _, row in ranked.iterrows():
        path = row.get("prediction_file")
        if pd.notna(path) and (evaluation / path).exists():
            low, high = _bootstrap(evaluation / path, cfg["evaluation"]["bootstrap_reps"], cfg["bayes"]["seed"])
        else:
            low, high = np.nan, np.nan
        ci_low.append(low); ci_high.append(high)
    ranked["wape_ci95_low"], ranked["wape_ci95_high"] = ci_low, ci_high
    ranked["decision"] = "challenger"
    ranked.loc[ranked.experiment_id.str.startswith("E00_M3_BASE"), "decision"] = "baseline_obligatorio"
    champion = ranked.iloc[0].experiment_id
    ranked.loc[ranked.experiment_id.eq(champion), "decision"] = "champion_provisional"
    ranked.to_csv(evaluation / "ranking_final.csv", index=False)
    all_metrics.to_csv(evaluation / "metrics_comparacion_final.csv", index=False)
    report = {"champion_provisional": champion,
              "ranking_population": f"{champion_split} causal n={population_n} diario",
              "selected_model_manifest": "selected_model_manifest.json",
              "excluded_retrospective": int((~causal_flag).sum()),
              "excluded_noncomparable": int((~causal.scope_comparable).sum()),
              "bootstrap_reps": cfg["evaluation"]["bootstrap_reps"]}
    (evaluation / "champion_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Reporte Fase 8 - Comparacion final", "", "## Poblacion comun", "",
             f"El ranking primario usa `{champion_split}` causal con {population_n:,} observaciones diarias comunes entre modelos.",
             "Los experimentos retrospectivos P32 y las poblaciones o escalas distintas quedan excluidos.", "", "## Ranking", "",
             "| Modelo | WAPE | IC bootstrap 95% | Decision |", "|---|---:|---:|---|"]
    for _, row in ranked.iterrows():
        interval = (f"{row.wape_ci95_low:.2%}-{row.wape_ci95_high:.2%}"
                    if pd.notna(row.wape_ci95_low) else "n/d")
        lines.append(f"| {row.experiment_id} | {row.wape:.2%} | {interval} | {row.decision} |")
    lines += ["", "## Decision", "",
              f"`{champion}` es el champion provisional bajo rolling-origin causal.",
              "M3 permanece como baseline obligatorio y referencia mecanistica.",
              "La promocion operacional requiere un periodo futuro congelado independiente.", "",
              "Artefactos: `ranking_final.csv`, `metrics_comparacion_final.csv`, `champion_manifest.json` y `selected_model_manifest.json`."]
    (root / cfg["paths"]["docs"] / "REPORTE_FASE8_COMPARACION_FINAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(ranked[["experiment_id", "wape", "wape_ci95_low", "wape_ci95_high", "decision"]].to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
