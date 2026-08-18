"""
Feature engineering: forma reciente por equipo -- tiros al arco (v4) y
ahora tambien corners (v6).

MOTIVACION (tiros al arco, v4): el modelo (v1/v2/v3) calcula el rating de
cada equipo unicamente a partir de goles marcados/recibidos, ponderados por
recencia. Los goles tienen mucha varianza en una sola muestra -- un equipo
puede ganar 1-0 dominando con 18 tiros al arco, o ganar 1-0 de rebote con 2.
Los tiros al arco son un proxy de "calidad subyacente" con menos ruido que
el resultado final.

MOTIVACION (corners, v6): los corners son otra señal de presion ofensiva
sostenida, parcialmente independiente de los tiros al arco -- un equipo
puede generar muchos corners sin rematar a puerta (juego por bandas,
centros) o viceversa (remates desde fuera del area). Se agrega como
diferencial neto (mismo criterio que v4, no separado en ataque/defensa --
v5 ya demostro que separar la señal de tiros en dos covariables no mejoro
el resultado, asi que no se repite ese patron aca sin evidencia a favor).

Este script agrega OCHO columnas nuevas a matches_clean.csv, todas basadas
en promedios moviles de los ultimos N=5 partidos ANTERIORES a cada fila
(shift(1) antes del rolling -- cero fuga de informacion). Prior neutro
(0.0) para el primer partido de un equipo en todo el dataset:

  v4 (diferencial neto de tiros al arco, usado en poisson_model_v4.py):
    - home_recent_st_diff / away_recent_st_diff

  v5 (tiros al arco, ataque y defensa SEPARADOS -- ver poisson_model_v5.py,
  resultado negativo documentado, se mantiene solo por referencia/registro):
    - home_recent_attack / home_recent_defense
    - away_recent_attack / away_recent_defense

  v6 (diferencial neto de corners, NUEVO -- ver poisson_model_v6.py):
    - home_recent_corner_diff / away_recent_corner_diff

Se corre UNA VEZ sobre el dataset completo (no por fold de walk-forward):
la forma reciente de un equipo en una fecha dada es un hecho historico fijo,
no depende de que version del modelo lo vaya a usar despues.

IDEMPOTENTE DE VERDAD: si matches_clean.csv ya tiene alguna de estas 8
columnas de una corrida anterior, se descartan ANTES de recalcular -- si
no, pandas .merge() encuentra nombres de columna duplicados y los renombra
a "..._x"/"..._y" en vez de sobreescribirlos, rompiendo todo el resto de la
funcion con un KeyError silencioso mas adelante. (Bug real que ya aparecio
una vez -- ver roadmap, seccion de incidentes -- corregido aca.)

IMPORTANTE: hay que volver a correr este script cada vez que se regenere
matches_clean.csv desde cero (por ejemplo, despues de agregar una temporada
nueva via clean_data.py) -- si no, las columnas de forma reciente quedan
desactualizadas o directamente no existen.

Fix 2026-08-18 (Fase 8, multi-liga): run() estaba hardcodeado a
PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv" -- a diferencia de
clean_data.py y football_data_loader.py, que ya nacieron parametrizados
por league_key. add_recent_form_features() en si (la logica de rolling
window) ya era agnostica de liga, no dependia de nada especifico de EPL --
el unico cambio necesario fue parametrizar run() por liga y loopear sobre
LEAGUES en __main__, mismo patron que los otros dos scripts.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ROLLING_WINDOW = 5

# Cada entrada: (nombre_stat_para_reportar, columna_local_cruda, columna_visitante_cruda)
STATS = [
    ("shots_on_target", "HST", "AST"),
    ("corners", "HC", "AC"),
]

FEATURE_COLS_ST = ["recent_st_diff", "recent_attack", "recent_defense"]  # v4 + v5, solo tiros
FEATURE_COLS_CORNERS = ["recent_corner_diff"]  # v6, solo diferencial neto (sin separar ataque/defensa)

GENERATED_COLS = (
    [f"{side}_{c}" for side in ("home", "away") for c in FEATURE_COLS_ST]
    + [f"{side}_{c}" for side in ("home", "away") for c in FEATURE_COLS_CORNERS]
)


def _rolling_team_features(df: pd.DataFrame, stat_for_col: str, stat_against_col: str,
                            window: int, out_prefix: str, include_split: bool) -> pd.DataFrame:
    """
    Construye la vista larga (una fila por equipo por partido) para UNA
    estadistica cruda (tiros al arco o corners) y devuelve un DataFrame
    con match_id + las columnas home_/away_ correspondientes, listas para
    mergear de vuelta al df original. include_split=True agrega tambien
    ataque/defensa separados (solo se usa para tiros al arco, v4/v5);
    para corners (v6) solo se calcula el diferencial neto.
    """
    home_perspective = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "team": df["HomeTeam"],
        "stat_for": df[stat_for_col],
        "stat_against": df[stat_against_col],
    })
    away_perspective = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "team": df["AwayTeam"],
        "stat_for": df[stat_against_col],
        "stat_against": df[stat_for_col],
    })
    team_long = pd.concat([home_perspective, away_perspective], ignore_index=True)
    team_long["stat_diff"] = team_long["stat_for"] - team_long["stat_against"]
    team_long = team_long.sort_values(["team", "Date", "match_id"])

    # Para corners (include_split=False) solo calculamos el diferencial.
    # Para tiros al arco (include_split=True) calculamos diferencial + ataque + defensa,
    # reutilizando los nombres historicos de v4/v5 (recent_st_diff/recent_attack/recent_defense).
    if include_split:
        raw_to_feature = {
            "stat_diff": "recent_st_diff",
            "stat_for": "recent_attack",
            "stat_against": "recent_defense",
        }
    else:
        raw_to_feature = {"stat_diff": f"{out_prefix}_diff"}

    for raw_col, out_col in raw_to_feature.items():
        team_long[out_col] = team_long.groupby("team")[raw_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        team_long[out_col] = team_long[out_col].fillna(0.0)

    feature_cols = list(raw_to_feature.values())

    home_side = team_long.merge(
        df[["match_id", "HomeTeam"]], left_on=["match_id", "team"], right_on=["match_id", "HomeTeam"], how="inner"
    )[["match_id"] + feature_cols].rename(columns={c: f"home_{c}" for c in feature_cols})

    away_side = team_long.merge(
        df[["match_id", "AwayTeam"]], left_on=["match_id", "team"], right_on=["match_id", "AwayTeam"], how="inner"
    )[["match_id"] + feature_cols].rename(columns={c: f"away_{c}" for c in feature_cols})

    return home_side, away_side


def add_recent_form_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Agrega las 8 columnas descritas arriba a df. Requiere que df tenga HST,
    AST, HC y AC (football-data.co.uk las trae para EPL desde hace muchas
    temporadas, pero se valida igual).
    """
    required_raw_cols = [c for _, home_col, away_col in STATS for c in (home_col, away_col)]
    missing = [c for c in required_raw_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} en el dataset -- football-data.co.uk deberia traerlas "
            f"para EPL, pero si estas usando otra liga/fuente puede que no esten disponibles. "
            f"Revisa el CSV crudo antes de seguir."
        )

    df = df.copy()
    # Idempotencia real: ver docstring del modulo.
    df = df.drop(columns=[c for c in GENERATED_COLS if c in df.columns])

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    df["match_id"] = df.index

    # --- tiros al arco: diferencial neto (v4) + ataque/defensa separados (v5) ---
    home_st, away_st = _rolling_team_features(
        df, stat_for_col="HST", stat_against_col="AST", window=window,
        out_prefix="recent_st", include_split=True,
    )
    # --- corners: solo diferencial neto (v6) ---
    home_corners, away_corners = _rolling_team_features(
        df, stat_for_col="HC", stat_against_col="AC", window=window,
        out_prefix="recent_corner", include_split=False,
    )

    df = (
        df.merge(home_st, on="match_id", how="left")
          .merge(away_st, on="match_id", how="left")
          .merge(home_corners, on="match_id", how="left")
          .merge(away_corners, on="match_id", how="left")
    )
    for c in GENERATED_COLS:
        df[c] = df[c].fillna(0.0)
    df = df.drop(columns=["match_id"])
    return df


def run(league_key: str) -> None:
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe {path} -- corre clean_data.py primero.")
        return

    df = pd.read_csv(path)
    print(f"[{league_key}] Cargado {path} ({len(df)} partidos)")

    df = add_recent_form_features(df)

    print(f"[{league_key}] Agregadas/recalculadas columnas (ventana={ROLLING_WINDOW} partidos):")
    print("  v4 (diferencial neto de tiros al arco): home_recent_st_diff, away_recent_st_diff")
    print("  v5 (tiros al arco, ataque/defensa separados): home_recent_attack, home_recent_defense, "
          "away_recent_attack, away_recent_defense")
    print("  v6 (diferencial neto de corners): home_recent_corner_diff, away_recent_corner_diff")
    cols_to_show = ["Date", "HomeTeam", "AwayTeam", "home_recent_st_diff", "away_recent_st_diff",
                     "home_recent_corner_diff", "away_recent_corner_diff"]
    print(df[cols_to_show].tail(5))

    df.to_csv(path, index=False)
    print(f"[{league_key}] Guardado (sobreescrito) -> {path}\n")


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)