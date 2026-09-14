from pathlib import Path

import numpy as np
import pandas as pd

from src.proyeccion_vision import (aggregate_vision_counts, load_vision_videos,
                                   project_vision_counts, _latest_week,
                                   _parse_vision_date, _validate_next_week_prediction,
                                   _validate_single_iso_year)
from src.models.m3 import M3Matrix


def _cfg() -> dict:
    return {
        "project": {"target_farms": ["ALMER", "LA PRADERA", "SANTA HELENA"]},
        "farm_aliases": {"ALMER": "ALMER", "PRADERA": "LA PRADERA"},
        "vision": {"require_valid_video_metadata": True, "reject_bad_qc": True,
                   "reject_identity_mismatch": True},
        "forecast": {"horizon_days": 7},
        "m3": {"periods": [{"name": "JULIO", "fecha_inicio_vigencia": "2026-07-01",
                              "fecha_fin_vigencia": "2026-12-31"}]},
    }


def _write_video(path: Path, *, rc=1, ss=2, ap=3, co=4, delimiter=",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{
        "archivo_original": f"{path.stem}.MP4", "semana": "S36", "finca": "Almer",
        "bloque": "1", "muestra_idx": 1, "fecha": "2026-09-01",
        "duracion_s": 30, "fps": 29.97, "ancho": 1920, "alto": 1080,
        "veredicto_qc": "", "problemas_qc": "", "conteo_RC": rc,
        "conteo_SS": ss, "conteo_AP": ap, "conteo_CO": co,
        "conteo_total": rc + ss + ap + co,
    }])
    frame.to_csv(path, sep=delimiter, index=False)


def test_each_valid_csv_is_one_sampled_bed(tmp_path):
    root = tmp_path / "Resultados"
    _write_video(root / "S36" / "Almer" / "1" / "a.csv")
    _write_video(root / "S36" / "Almer" / "1" / "b.csv", delimiter=";")
    videos, qa = load_vision_videos(root, _cfg(), "S36")
    beds = pd.DataFrame([{
        "finca": "ALMER", "bloque": "1", "cama_id": str(i), "variedad": "FREEDOM",
        "fecha_siembra": pd.Timestamp("2026-01-01"), "fecha_erradicacion": pd.NaT,
        "plantas": 100, "area_sembrada": 10, "estado": "ACTIVO",
    } for i in range(4)])
    counts, _ = aggregate_vision_counts(videos, beds)

    assert len(videos) == 2
    assert counts.loc[0, "camas_muestreadas"] == 2
    assert counts.loc[0, "camas_activas"] == 4
    assert counts.loc[0, "factor_extrapolacion"] == 2
    assert counts.loc[0, "conteo_ap"] == 6
    assert qa["motivo"].eq("QC_NO_INFORMADO").sum() == 2


def test_visual_co_stays_separate_from_m3_state(monkeypatch):
    counts = pd.DataFrame([{
        "finca": "ALMER", "bloque": "1", "fecha_conteo": pd.Timestamp("2026-09-01"),
        "semana_iso": 202636, "conteo_rc": 10, "conteo_ss": 20, "conteo_ap": 30,
        "conteo_corte_inmediato_visual": 999, "conteo_total": 1059,
        "camas_muestreadas": 2, "camas_activas": 4, "factor_extrapolacion": 2.0,
    }])
    observed = {}

    def fake_fit(*args, **kwargs):
        return M3Matrix("ALMER", "JULIO", np.eye(3), np.zeros(3), np.zeros(3), pd.DataFrame())

    def fake_simulate(matrix, x0, days, alpha):
        observed["x0"] = x0.copy()
        return pd.DataFrame({"dia_simulado": range(1, days + 1),
                             "PC_dia_muestra": np.ones(days)})

    monkeypatch.setattr("src.proyeccion_vision.fit_m3", fake_fit)
    monkeypatch.setattr("src.proyeccion_vision.simulate", fake_simulate)
    daily, weekly = project_vision_counts(counts, pd.DataFrame(), _cfg(), 0.2)

    assert np.array_equal(observed["x0"], np.array([10.0, 20.0, 30.0]))
    assert len(daily) == 7
    assert daily["pred_bloque"].eq(2.0).all()
    assert weekly.loc[0, "pred_bloque_semana"] == 14.0


def test_invalid_video_metadata_is_rejected(tmp_path):
    root = tmp_path / "Resultados"
    path = root / "S36" / "Almer" / "1" / "a.csv"
    _write_video(path)
    frame = pd.read_csv(path)
    frame.loc[0, "fps"] = 0
    frame.to_csv(path, index=False)

    videos, qa = load_vision_videos(root, _cfg(), "S36")

    assert videos.empty
    assert qa["motivo"].tolist() == ["METADATA_VIDEO_INVALIDA"]


def test_iso_date_is_not_reinterpreted_as_day_first():
    assert _parse_vision_date("2026-09-01") == pd.Timestamp("2026-09-01")
    assert _parse_vision_date("01/09/2026") == pd.Timestamp("2026-09-01")


def test_latest_week_uses_capture_date_across_year_boundary(tmp_path):
    older = tmp_path / "S52" / "Almer" / "1" / "DJI_20261224000000_0001.csv"
    newer = tmp_path / "S1" / "Almer" / "1" / "DJI_20270105000000_0001.csv"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("x\n", encoding="utf-8")
    newer.write_text("x\n", encoding="utf-8")
    assert _latest_week(tmp_path) == "S1"


def test_identity_mismatch_is_rejected(tmp_path):
    root = tmp_path / "Resultados"
    path = root / "S36" / "Almer" / "1" / "a.csv"
    _write_video(path)
    frame = pd.read_csv(path)
    frame.loc[0, "bloque"] = 2
    frame.to_csv(path, index=False)

    videos, qa = load_vision_videos(root, _cfg(), "S36")
    assert videos.empty
    assert qa["motivo"].tolist()[-1] == "METADATA_DIFIERE_DE_RUTA"


def test_infinite_count_is_rejected_without_aborting_batch(tmp_path):
    root = tmp_path / "Resultados"
    invalid = root / "S36" / "Almer" / "1" / "a.csv"
    valid = root / "S36" / "Almer" / "1" / "b.csv"
    _write_video(invalid)
    _write_video(valid)
    frame = pd.read_csv(invalid)
    frame["conteo_RC"] = frame["conteo_RC"].astype(float)
    frame.loc[0, "conteo_RC"] = np.inf
    frame.to_csv(invalid, index=False)

    videos, qa = load_vision_videos(root, _cfg(), "S36")
    assert len(videos) == 1
    assert qa["motivo"].eq("CONTEOS_INVALIDOS").any()


def test_mixed_iso_years_in_one_week_are_rejected():
    videos = pd.DataFrame({"semana_iso": [202601, 202701]})
    with np.testing.assert_raises_regex(ValueError, "mas de un ano ISO"):
        _validate_single_iso_year(videos, "S1")


def test_operational_run_only_accepts_the_next_iso_week():
    videos = pd.DataFrame({"fecha_conteo": pd.to_datetime(["2026-09-01"])})
    daily = pd.DataFrame({"semana_proyeccion": [202637, 202637]})
    assert _validate_next_week_prediction(daily, videos, "S36") == 202637


def test_operational_run_rejects_accumulated_weeks():
    videos = pd.DataFrame({"fecha_conteo": pd.to_datetime(["2026-09-01"])})
    daily = pd.DataFrame({"semana_proyeccion": [202636, 202637]})
    with np.testing.assert_raises_regex(ValueError, "exactamente la semana siguiente"):
        _validate_next_week_prediction(daily, videos, "S36")
