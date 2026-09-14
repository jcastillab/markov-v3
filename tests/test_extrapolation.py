import numpy as np
import pandas as pd
import pytest

from src.modeling.extrapolation import attach_extrapolation


def _frame():
    return pd.DataFrame({
        "finca": ["ALMER", "ALMER"], "bloque": ["1", "1"],
        "semana_iso": [202636, 202636],
        "fecha": pd.to_datetime(["2026-09-01", "2026-09-02"]),
    })


def test_factor_is_active_over_sampled_and_capped_at_one():
    result = attach_extrapolation(
        _frame(),
        pd.DataFrame({"finca": ["ALMER"], "bloque": ["1"],
                      "semana_iso": [202636], "camas_muestreadas": [2]}),
        pd.DataFrame({"finca": ["ALMER", "ALMER"], "bloque": ["1", "1"],
                      "fecha": pd.to_datetime(["2026-09-01", "2026-09-02"]),
                      "camas_activas": [4, 1]}),
    )
    assert result["factor_extrapolacion"].tolist() == [2.0, 1.0]
    assert np.allclose(result["cobertura_muestreo"], [0.5, 2.0])


def test_missing_scale_is_rejected_with_keys():
    with pytest.raises(ValueError, match="202636"):
        attach_extrapolation(
            _frame().iloc[[0]],
            pd.DataFrame(columns=["finca", "bloque", "semana_iso", "camas_muestreadas"]),
            pd.DataFrame(columns=["finca", "bloque", "fecha", "camas_activas"]),
        )
