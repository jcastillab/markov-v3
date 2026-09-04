"""Fase 3: P32, matrices empiricas y Semi Markov corregido."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import load_config
from evaluation.metrics import metrics
from evaluation.split import temporal_masks
from models.m3 import _period_for_date, fit_m3
from models.p32 import build_p32_intervals, fit_p32_matrix, load_p32_observations
from models.semimarkov import (SemiMarkov, competitive_hazards,
                               estimate_age_distributions)


def _predict(row, intervals, cfg, model, alpha, semimarkov=False):
    origin = pd.Timestamp(row["fecha_origen"])
    target = pd.Timestamp(row["fecha_objetivo"])
    x0 = np.array([row["conteo_RC_t0"], row["conteo_SS_t0"], row["conteo_AP_t0"]], float)
    days = (target - origin).days
    if semimarkov:
        result = model.simulate(x0, days, alpha)
    else:
        result = model_simulate(model, x0, days, alpha)
    last = result.iloc[-1]
    return {"finca": row["finca"], "bloque": row["bloque"],
            "fecha_origen": origin, "fecha_objetivo": target,
             "horizonte_dia": row["horizonte_dia"],
             "semana_proyeccion": row["semana_proyeccion"],
            "pred_bloque": last["PC_dia_muestra"] * row["factor_extrapolacion_t0"],
            "real": row["corte_real_dia"], "causal": False,
            "evaluacion_causal": False, "fecha_max_dato_modelo": pd.Timestamp("2026-08-20")}


def model_simulate(matrix, x0, days, alpha):
    from models.m3 import simulate
    return simulate(matrix, x0, days, alpha)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    raw = root / cfg["paths"]["raw"]
    outputs = root / cfg["paths"]["outputs"]
    datasets, evaluation, models, predictions = [outputs / x for x in
                                                  ("datasets", "evaluation", "models", "predictions")]
    for directory in (datasets, evaluation, models, predictions):
        directory.mkdir(parents=True, exist_ok=True)
    observations = load_p32_observations(raw / cfg["sources"]["fenologia_p32"], cfg)
    intervals = build_p32_intervals(observations)
    observations.to_parquet(datasets / "observaciones_p32.parquet", index=False)
    intervals.to_parquet(datasets / "transition_intervals_p32.parquet", index=False)
    intervals_valid = intervals[intervals["valido"]]
    qa = intervals.groupby("motivo_exclusion", dropna=False).size().reset_index(name="n")
    qa.to_csv(evaluation / "qa_p32_transiciones.csv", index=False)
    windows = pd.read_parquet(datasets / "forecast_windows.parquet")
    traditional = pd.read_parquet(datasets / "transition_intervals_tradicional.parquet")
    _, valid = temporal_masks(windows, cfg, origin_values=windows["fecha_origen"])
    mask = pd.Series(valid, index=windows.index) & windows.ventana_evaluable
    val_windows = windows[mask]
    alpha = float(cfg["m3"]["baseline_ingress"])
    results, metric_rows = [], []
    if len(intervals_valid) == 0:
        raise ValueError("P32 no tiene intervalos validos")
    for exp, tau, semi_states in [("E02_M3_P32_RAW", 0.0, None), ("E03_M3_P32_CANONICO", 0.0, None),
                                   ("E04_M3_P32_REG", 1.0, None), ("E05_SEMIMARKOV_P32_RC", 1.0, ["RC"]),
                                   ("E06_SEMIMARKOV_P32_RC_SS", 1.0, ["RC", "SS"]),
                                   ("E07_SEMIMARKOV_P32_ALL", 1.0, ["RC", "SS", "AP"])]:
        rows = []
        # Un modelo se calcula por origen, no una vez por cada uno de sus 7 dias.
        origin_rows = val_windows.drop_duplicates(["finca", "bloque", "fecha_origen"])
        for _, origin_row in origin_rows.iterrows():
            period = _period_for_date(pd.Timestamp(origin_row["fecha_origen"]), cfg["m3"]["periods"])
            m3_prior = fit_m3(traditional, origin_row["finca"], period, pd.Timestamp(origin_row["fecha_origen"]))
            matrix = fit_p32_matrix(intervals_valid, m3_prior if tau else None, tau=tau)
            if semi_states is None:
                model = matrix
            else:
                hazards = competitive_hazards(intervals_valid, m3_prior, tau=cfg["semimarkov"]["tau"],
                                               max_bin=len(cfg["semimarkov"]["age_bins"]) - 1)
                # Keep P32 hazards only for the selected states; M3 hazards are represented
                # by a matrix-backed Semi Markov approximation for other states.
                if semi_states != ["RC", "SS", "AP"]:
                    # This explicit ablation is conservative: selected states use P32,
                    # unselected states use M3 hazards with age-invariant probabilities.
                    m3_hazards = competitive_hazards(intervals_valid.iloc[0:0], m3_prior, tau=0.0)
                    for key in list(hazards):
                        if key[0] not in semi_states:
                            hazards[key] = m3_hazards[key]
                model = SemiMarkov(hazards, estimate_age_distributions(intervals_valid),
                                   max_bin=len(cfg["semimarkov"]["age_bins"]) - 1)
            key_mask = (val_windows["finca"].eq(origin_row["finca"]) &
                        val_windows["bloque"].eq(origin_row["bloque"]) &
                        val_windows["fecha_origen"].eq(origin_row["fecha_origen"]))
            for _, row in val_windows[key_mask].iterrows():
                out = _predict(row, intervals_valid, cfg, model, alpha, semi_states is not None)
                out["experiment_id"] = exp
                rows.append(out)
        pred = pd.DataFrame(rows)
        pred.to_csv(predictions / f"{exp}.csv", index=False)
        score = metrics(pred["real"], pred["pred_bloque"])
        metric_rows.append({"experiment_id": exp, "split": "VALIDATION",
                            "causal": False, "status": "RETROSPECTIVO_ORACLE_NO_CAUSAL", **score})
        results.append(pred)
    pd.DataFrame(metric_rows).to_csv(evaluation / "metrics_fase3_p32.csv", index=False)
    age = estimate_age_distributions(intervals_valid)
    pd.DataFrame([{"estado": state, "bin_edad": i, "probabilidad": p}
                  for state, values in age.items() for i, p in enumerate(values)]).to_csv(
                      models / "p32_age_distributions.csv", index=False)
    pd.DataFrame([{"estado": state, "bin_edad": age_bin, **hazard}
                  for (state, age_bin), hazard in competitive_hazards(
                      intervals_valid, fit_p32_matrix(intervals_valid)).items()]).to_csv(
                          models / "p32_hazards.csv", index=False)
    (models / "p32_manifest.json").write_text(json.dumps({
        "experiments": ["E02_M3_P32_RAW", "E03_M3_P32_CANONICO", "E04_M3_P32_REG",
                        "E05_SEMIMARKOV_P32_RC", "E06_SEMIMARKOV_P32_RC_SS",
                        "E07_SEMIMARKOV_P32_ALL"],
        "n_observations": len(observations), "n_intervals": len(intervals),
        "n_valid_intervals": len(intervals_valid), "causal": False,
        "label": "RETROSPECTIVO_ORACLE_NO_CAUSAL", "alpha_ingreso_RC": alpha},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Observaciones P32: {len(observations):,}")
    print(f"Intervalos: {len(intervals):,}; validos: {len(intervals_valid):,}")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
