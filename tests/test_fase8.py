from pathlib import Path

import pandas as pd


def test_final_ranking_excludes_noncausal_models():
    path = Path(__file__).resolve().parents[1] / "outputs/evaluation/ranking_final.csv"
    if not path.exists():
        return
    ranking = pd.read_csv(path)
    assert ranking["n"].nunique() == 1
    assert ranking["split"].eq("ROLLING_ORIGIN_COMMON").all()
    assert ranking.iloc[0]["decision"] == "champion_provisional"
