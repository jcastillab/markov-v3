import pandas as pd

from src.evaluation.split import holdout_start, selection_masks, temporal_masks


def test_temporal_split_keeps_origin_dates_whole_and_targets_before_cutoff():
    frame = pd.DataFrame({
        "fecha_origen": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02",
                                         "2026-01-02", "2026-01-03", "2026-01-03"]),
        "fecha_objetivo": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-02",
                                           "2026-01-03", "2026-01-03", "2026-01-04"]),
        "target": [1, 1, 1, 1, 1, 1],
    })
    cfg = {"evaluation": {"train_fraction": 0.60, "min_train_windows": 1}}
    train, valid = temporal_masks(frame, cfg)
    assert set(frame.loc[train, "fecha_origen"]) == {pd.Timestamp("2026-01-01")}
    assert set(frame.loc[valid, "fecha_origen"]) == {
        pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")
    }
    assert (frame.loc[train, "fecha_objetivo"] < pd.Timestamp("2026-01-02")).all()


def test_selection_period_is_disjoint_from_final_holdout():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"fecha_origen": dates, "fecha_objetivo": dates,
                          "target": range(10)})
    cfg = {"evaluation": {"train_fraction": 0.60, "selection_fraction": 0.20,
                           "min_train_windows": 1}}
    train, selection = selection_masks(frame, cfg)
    cutoff = holdout_start(dates, cfg)
    assert frame.loc[train, "fecha_origen"].max() < frame.loc[selection, "fecha_origen"].min()
    assert frame.loc[selection, "fecha_origen"].max() < cutoff
    assert (frame.loc[selection, "fecha_objetivo"] < cutoff).all()
    assert (frame.loc[~(train | selection), "fecha_origen"] >= cutoff).all()
