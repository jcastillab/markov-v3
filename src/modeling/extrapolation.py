"""Expansion de conteos muestreados al bloque completo.

Este modulo no depende de ningun estimador y puede trasladarse junto con un
proyecto nuevo. La ausencia de escala es un error de datos, no un cero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS_WEEK = ["finca", "bloque", "semana_iso"]
KEYS_DATE = ["finca", "bloque", "fecha"]


def attach_extrapolation(frame: pd.DataFrame, sampled: pd.DataFrame,
                         active: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Agrega camas y factor validado a un frame de entrada.

    ``sampled`` debe estar agregado por finca-bloque-semana y ``active`` por
    finca-bloque-fecha. Si una clave no tiene escala válida se lanza un error
    con una muestra de las claves afectadas.
    """
    result = frame.copy()
    result = result.merge(sampled, on=KEYS_WEEK, how="left", validate="many_to_one")
    result = result.merge(active, on=KEYS_DATE, how="left", validate="one_to_one")
    result["camas_muestreadas"] = pd.to_numeric(result["camas_muestreadas"], errors="coerce")
    result["camas_activas"] = pd.to_numeric(result["camas_activas"], errors="coerce")
    invalid = (result["camas_muestreadas"].isna() |
               result["camas_activas"].isna() |
               result["camas_muestreadas"].le(0) |
               result["camas_activas"].le(0))
    if invalid.any() and strict:
        keys = result.loc[invalid, KEYS_DATE + ["semana_iso"]].drop_duplicates().head(20)
        raise ValueError(
            "Faltan camas muestreadas/activas para calcular extrapolacion: "
            f"{keys.to_dict('records')}"
        )
    result["factor_extrapolacion"] = result["camas_activas"] / result["camas_muestreadas"]
    over = result["camas_muestreadas"] > result["camas_activas"]
    result.loc[over, "factor_extrapolacion"] = 1.0
    result["cobertura_muestreo"] = result["camas_muestreadas"] / result["camas_activas"]
    result["escala_disponible"] = ~invalid
    valid_factors = result.loc[~invalid, "factor_extrapolacion"]
    if not np.isfinite(valid_factors).all():
        raise ValueError("El factor de extrapolacion contiene valores no finitos")
    return result
