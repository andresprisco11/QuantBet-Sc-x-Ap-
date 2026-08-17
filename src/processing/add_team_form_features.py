"""
Feature engineering: forma reciente por equipo basada en tiros al arco (HST/AST).

MOTIVACION: el modelo (v1/v2/v3) calcula el rating de cada equipo unicamente
a partir de goles marcados/recibidos, ponderados por recencia. Los goles
tienen mucha varianza en una sola muestra -- un equipo puede ganar 1-0
dominando con 18 tiros al arco, o ganar 1-0 de rebote con 2. Los tiros al
arco son un proxy de "calidad subyacente" con menos ruido que el resultado
final (equivalente pobre de un xG propio, sin necesitar datos de pago).

Este script agrega DOS columnas nuevas a matches_clean.csv:
  - home_recent_st_diff: promedio movil de (tiros al arco a favor - en contra)
    del equipo LOCAL en sus ultimos N partidos ANTERIORES a este.
  - away_recent_st_diff: lo mismo para el equipo VISITANTE.

SIN FUGA DE INFORMACION: para el partido que se juega el dia X, el promedio
solo usa partidos con fecha ESTRICTAMENTE anterior a X (shift(1) antes del
rolling). Para el primer partido de un equipo en todo el dataset (sin
historial previo alguno), el valor queda en 0.0 -- un prior neutro, mismo
criterio que ya usamos con PROMOTED_TEAM en poisson_model_v2.py.

Se corre UNA VEZ sobre el dataset completo (no por fold de walk-forward):
la forma reciente de un equipo en una fecha dada es un hecho historico fijo,
no depende de que version del modelo lo vaya a usar despues. Es idempotente
-- correrlo de nuevo simplemente recalcula y sobreescribe las mismas dos
columnas, no duplica nada.

IMPORTANTE: hay que volver a correr este script cada vez que se regenere
matches_clean.csv desde cero (por ejemplo, despues de agregar una temporada
nueva via clean_data.py) -- si no, las columnas de forma reciente quedan
desactualizadas o directamente no existen.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

ROLLING_WINDOW = 5
STAT_FOR_COL = "HST"   # tiros al arco del local, tal cual viene de football-data.co.uk
STAT_AGAINST_COL = "AST"  # tiros al arco del visitante
FEATURE_NAME = "recent_st_diff"


def add_recent_form_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Agrega home_recent_st_diff / away_recent_st_diff a df. No modifica ninguna
    columna existente. Requiere que df tenga HST y AST (football-data.co.uk
    las trae para EPL desde hace muchas temporadas, pero se valida igual).
    """
    missing = [c for c in [STAT_FOR_COL, STAT_AGAINST_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} en el dataset -- football-data.co.uk deberia traerlas "
            f"para EPL, pero si estas usando otra liga/fuente puede que no esten disponibles. "
            f"Revisa el CSV crudo antes de seguir."
        )

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    df["match_id"] = df.index

    # Vista "larga" (una fila por equipo por partido, en vez de una fila por
    # partido) -- necesaria para poder agrupar por equipo y calcular su
    # promedio movil cronologico, sin importar si jugo de local o visitante.
    home_perspective = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "team": df["HomeTeam"],
        "stat_diff": df[STAT_FOR_COL] - df[STAT_AGAINST_COL],
    })
    away_perspective = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "team": df["AwayTeam"],
        "stat_diff": df[STAT_AGAINST_COL] - df[STAT_FOR_COL],
    })
    team_long = pd.concat([home_perspective, away_perspective], ignore_index=True)
    team_long = team_long.sort_values(["team", "Date", "match_id"])

    # shift(1) = excluye el partido actual del promedio -- solo historial estrictamente
    # anterior. rolling(window, min_periods=1) = usa lo que haya disponible (1 a N
    # partidos previos), no exige tener los N completos.
    team_long[FEATURE_NAME] = team_long.groupby("team")["stat_diff"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    # Cold start real: el primerisimo partido de un equipo en todo el dataset no tiene
    # NADA de historial previo -- prior neutro (0.0), mismo criterio que PROMOTED_TEAM.
    team_long[FEATURE_NAME] = team_long[FEATURE_NAME].fillna(0.0)

    # Volver de la vista larga a la vista por partido: para cada match_id, separar
    # cual fila corresponde a la perspectiva del local y cual a la del visitante,
    # via merge explicito por nombre de equipo -- robusto sin importar el orden
    # que haya quedado despues del sort_values de arriba.
    home_side = team_long.merge(
        df[["match_id", "HomeTeam"]], left_on=["match_id", "team"], right_on=["match_id", "HomeTeam"], how="inner"
    )[["match_id", FEATURE_NAME]].rename(columns={FEATURE_NAME: f"home_{FEATURE_NAME}"})

    away_side = team_long.merge(
        df[["match_id", "AwayTeam"]], left_on=["match_id", "team"], right_on=["match_id", "AwayTeam"], how="inner"
    )[["match_id", FEATURE_NAME]].rename(columns={FEATURE_NAME: f"away_{FEATURE_NAME}"})

    df = df.merge(home_side, on="match_id", how="left").merge(away_side, on="match_id", how="left")
    df[f"home_{FEATURE_NAME}"] = df[f"home_{FEATURE_NAME}"].fillna(0.0)
    df[f"away_{FEATURE_NAME}"] = df[f"away_{FEATURE_NAME}"].fillna(0.0)
    df = df.drop(columns=["match_id"])
    return df


def run():
    path = PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv"
    df = pd.read_csv(path)
    print(f"Cargado {path} ({len(df)} partidos)")

    df = add_recent_form_features(df)

    print(f"Agregadas columnas: home_recent_st_diff, away_recent_st_diff (ventana={ROLLING_WINDOW} partidos)")
    print(df[["Date", "HomeTeam", "AwayTeam", "home_recent_st_diff", "away_recent_st_diff"]].tail(10))

    df.to_csv(path, index=False)
    print(f"\nGuardado (sobreescrito) -> {path}")


if __name__ == "__main__":
    run()