"""
Feature engineering: forma reciente por equipo basada en tiros al arco (HST/AST).

MOTIVACION: el modelo (v1/v2/v3) calcula el rating de cada equipo unicamente
a partir de goles marcados/recibidos, ponderados por recencia. Los goles
tienen mucha varianza en una sola muestra -- un equipo puede ganar 1-0
dominando con 18 tiros al arco, o ganar 1-0 de rebote con 2. Los tiros al
arco son un proxy de "calidad subyacente" con menos ruido que el resultado
final (equivalente pobre de un xG propio, sin necesitar datos de pago).

Este script agrega SEIS columnas nuevas a matches_clean.csv, todas basadas
en promedios moviles de los ultimos N=5 partidos ANTERIORES a cada fila
(shift(1) antes del rolling -- cero fuga de informacion). Prior neutro
(0.0) para el primer partido de un equipo en todo el dataset:

  v4 (diferencial neto, usado en poisson_model_v4.py -- se mantiene por
  compatibilidad/referencia, aunque v5 en adelante usa la version separada):
    - home_recent_st_diff / away_recent_st_diff

  v5 (ataque y defensa SEPARADOS -- ver poisson_model_v5.py):
    - home_recent_attack / home_recent_defense: tiros al arco que el
      equipo LOCAL genero / concedio en sus ultimos 5 partidos (jugara
      donde jugara en esos partidos anteriores).
    - away_recent_attack / away_recent_defense: lo mismo para el
      equipo VISITANTE.

  Por que separar ataque de defensa: el diferencial neto (v4) no distingue
  si un equipo rinde mal porque su ataque esta flojo o porque su defensa
  esta regalando tiros -- son dos cosas distintas que un adversario
  responde de forma distinta. Separarlas le da al modelo la capacidad de
  aprender un coeficiente distinto para cada una, en vez de forzar un solo
  numero neto.

Se corre UNA VEZ sobre el dataset completo (no por fold de walk-forward):
la forma reciente de un equipo en una fecha dada es un hecho historico fijo,
no depende de que version del modelo lo vaya a usar despues.

IDEMPOTENTE DE VERDAD: si matches_clean.csv ya tiene alguna de estas 6
columnas de una corrida anterior (por ejemplo, la version vieja de este
mismo script que solo generaba las 2 de v4), se descartan ANTES de
recalcular -- si no, pandas .merge() encuentra nombres de columna
duplicados y los renombra a "..._x"/"..._y" en vez de sobreescribirlos,
rompiendo todo el resto de la funcion con un KeyError silencioso mas
adelante. (Bug real que aparecio la primera vez que se corrio esta
version sobre un CSV ya enriquecido por la version anterior -- corregido
aca.)

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
FEATURE_COLS = ["recent_st_diff", "recent_attack", "recent_defense"]
GENERATED_COLS = [f"{side}_{c}" for side in ("home", "away") for c in FEATURE_COLS]


def add_recent_form_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Agrega las 6 columnas descritas arriba a df. Requiere que df tenga HST
    y AST (football-data.co.uk las trae para EPL desde hace muchas
    temporadas, pero se valida igual).
    """
    missing = [c for c in [STAT_FOR_COL, STAT_AGAINST_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} en el dataset -- football-data.co.uk deberia traerlas "
            f"para EPL, pero si estas usando otra liga/fuente puede que no esten disponibles. "
            f"Revisa el CSV crudo antes de seguir."
        )

    df = df.copy()
    # Idempotencia real: si ya existen (de una corrida anterior, con esta
    # version del script o una vieja), se descartan antes de recalcular --
    # ver nota en el docstring del modulo sobre el bug que esto evita.
    df = df.drop(columns=[c for c in GENERATED_COLS if c in df.columns])

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    df["match_id"] = df.index

    # Vista "larga" (una fila por equipo por partido) -- necesaria para
    # agrupar por equipo y calcular sus promedios moviles cronologicos, sin
    # importar si jugo de local o visitante en cada partido anterior.
    # stat_for/stat_against se guardan SEPARADOS (no solo la diferencia)
    # para poder construir tanto v4 (diferencial neto) como v5 (ataque y
    # defensa separados) desde la misma tabla intermedia.
    home_perspective = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "team": df["HomeTeam"],
        "stat_for": df[STAT_FOR_COL],
        "stat_against": df[STAT_AGAINST_COL],
    })
    away_perspective = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "team": df["AwayTeam"],
        "stat_for": df[STAT_AGAINST_COL],
        "stat_against": df[STAT_FOR_COL],
    })
    team_long = pd.concat([home_perspective, away_perspective], ignore_index=True)
    team_long["stat_diff"] = team_long["stat_for"] - team_long["stat_against"]
    team_long = team_long.sort_values(["team", "Date", "match_id"])

    # shift(1) = excluye el partido actual del promedio -- solo historial estrictamente
    # anterior. rolling(window, min_periods=1) = usa lo que haya disponible (1 a N
    # partidos previos), no exige tener los N completos. Cold start real (primerisimo
    # partido de un equipo en todo el dataset, sin NADA de historial): prior neutro 0.0,
    # mismo criterio que PROMOTED_TEAM.
    raw_to_feature = {"stat_diff": "recent_st_diff", "stat_for": "recent_attack", "stat_against": "recent_defense"}
    for raw_col, out_col in raw_to_feature.items():
        team_long[out_col] = team_long.groupby("team")[raw_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        team_long[out_col] = team_long[out_col].fillna(0.0)

    # Volver de la vista larga a la vista por partido: para cada match_id, separar
    # cual fila corresponde a la perspectiva del local y cual a la del visitante,
    # via merge explicito por nombre de equipo -- robusto sin importar el orden
    # que haya quedado despues del sort_values de arriba.
    home_side = team_long.merge(
        df[["match_id", "HomeTeam"]], left_on=["match_id", "team"], right_on=["match_id", "HomeTeam"], how="inner"
    )[["match_id"] + FEATURE_COLS].rename(columns={c: f"home_{c}" for c in FEATURE_COLS})

    away_side = team_long.merge(
        df[["match_id", "AwayTeam"]], left_on=["match_id", "team"], right_on=["match_id", "AwayTeam"], how="inner"
    )[["match_id"] + FEATURE_COLS].rename(columns={c: f"away_{c}" for c in FEATURE_COLS})

    df = df.merge(home_side, on="match_id", how="left").merge(away_side, on="match_id", how="left")
    for c in GENERATED_COLS:
        df[c] = df[c].fillna(0.0)
    df = df.drop(columns=["match_id"])
    return df


def run():
    path = PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv"
    df = pd.read_csv(path)
    print(f"Cargado {path} ({len(df)} partidos)")

    df = add_recent_form_features(df)

    print(f"Agregadas/recalculadas columnas (ventana={ROLLING_WINDOW} partidos):")
    print("  v4 (diferencial neto): home_recent_st_diff, away_recent_st_diff")
    print("  v5 (ataque/defensa separados): home_recent_attack, home_recent_defense, "
          "away_recent_attack, away_recent_defense")
    cols_to_show = ["Date", "HomeTeam", "AwayTeam", "home_recent_attack", "home_recent_defense",
                     "away_recent_attack", "away_recent_defense"]
    print(df[cols_to_show].tail(10))

    df.to_csv(path, index=False)
    print(f"\nGuardado (sobreescrito) -> {path}")


if __name__ == "__main__":
    run()