from pathlib import Path

import numpy as np
import pandas as pd

from src.models.supervised import NegativeBinomialGLM


def test_nb_glm_returns_finite_nonnegative_predictions():
    x = np.array([[1., 0.], [2., 1.], [3., 0.], [4., 1.]])
    y = np.array([2., 4., 5., 8.])
    model = NegativeBinomialGLM(alpha=1.0).fit(x, y)
    pred = model.predict(x)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()


def test_supervised_dataset_artifact_has_contract_keys():
    path = Path(__file__).resolve().parents[1] / "outputs/datasets/dataset_supervisado_diario.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert {"finca", "bloque", "fecha_origen", "fecha_objetivo", "horizonte_dia", "target"}.issubset(df)
