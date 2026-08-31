from pathlib import Path

import pandas as pd


def test_final_ranking_excludes_noncausal_models():
    path = Path(__file__).resolve().parents[1] / "outputs/evaluation/ranking_final.csv"
    if not path.exists():
        return
    ranking = pd.read_csv(path)
    assert ranking["n"].eq(714).all()
    assert ranking["experiment_id"].str.startswith("E02").sum() == 0
    assert ranking.iloc[0]["decision"] == "champion_provisional"
