"""
Blending "Benter Boost": combina las probabilidades del modelo propio con las
probabilidades no-vig del mercado mas sharp (Pinnacle), ponderadas segun que
tan preciso ha sido cada uno historicamente (medido con Brier score).

Nota de honestidad tecnica: la formula EXACTA que uso Benter internamente no
es publica. Esto es nuestra implementacion concreta del mismo principio que
describe Mack -- ponderar segun precision relativa -- usando ponderacion por
error inverso (inverse-error weighting), un metodo estandar y transparente de
combinacion de pronosticos. Si el mercado ha sido historicamente mas preciso
que el modelo, pesa mas; si el modelo aporta senal real, su peso crece.
"""
import pandas as pd


def brier_score_multiclass(probs: pd.DataFrame, outcomes: pd.Series) -> float:
    """
    Brier score para un mercado de 3 resultados (H/D/A). probs debe tener
    columnas prob_home/prob_draw/prob_away. Mas bajo = mejor (0 = perfecto).
    """
    y_home = (outcomes == "H").astype(int)
    y_draw = (outcomes == "D").astype(int)
    y_away = (outcomes == "A").astype(int)

    sq_error = (
        (probs["prob_home"].values - y_home.values) ** 2
        + (probs["prob_draw"].values - y_draw.values) ** 2
        + (probs["prob_away"].values - y_away.values) ** 2
    )
    return sq_error.mean()


def compute_blend_weight(model_brier: float, market_brier: float) -> float:
    """Devuelve el peso del MERCADO (0 a 1) segun error inverso."""
    model_inv = 1 / model_brier
    market_inv = 1 / market_brier
    return market_inv / (model_inv + market_inv)


def blend_probabilities(model_probs: pd.DataFrame, market_probs: pd.DataFrame, market_weight: float) -> pd.DataFrame:
    """Combina linealmente modelo y mercado segun el peso, y renormaliza a que sumen 1."""
    blended = market_weight * market_probs.values + (1 - market_weight) * model_probs.values
    blended = blended / blended.sum(axis=1, keepdims=True)
    return pd.DataFrame(blended, columns=["prob_home", "prob_draw", "prob_away"], index=model_probs.index)