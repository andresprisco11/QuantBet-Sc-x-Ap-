"""
Fase 10 (NBA) -- variables de descanso/calendario del modelo de NBA. A
diferencia de NFL, ninguno de estos datos viene en ningun archivo
descargado -- `nba_api`/`games_clean.csv` no trae dias de descanso ni
densidad de calendario -- asi que se CALCULAN directo de las fechas reales
de partido que ya tenemos, no hace falta ninguna fuente nueva.

Dos variables, agregadas en dos rondas (la segunda motivada por el
resultado real de la primera):

1. **home_rest / away_rest** -- dias de descanso desde el partido anterior
   del mismo equipo, MENOS 1 (back-to-back real = 0). Probado primero como
   regresor CONTINUO en `nba_margin_model_v2.py` -- resultado real: p=0.335,
   NO significativo. Se sigue calculando (es la base del back-to-back
   binario, que si funciono), pero el propio valor continuo no se usa solo
   en el modelo.

2. **home_3in4 / away_3in4** -- NUEVO (agregado despues de confirmar que el
   back-to-back binario si funciona): indicador de "3 partidos en 4 noches"
   (incluyendo el partido actual), una fatiga MAS severa que un simple
   back-to-back. Motivado por la literatura de NBA (3-en-4 es un umbral de
   fatiga citado en la industria, mas severo que el back-to-back solo) y
   confirmado con datos propios ANTES de construir el modelo v3: OLS
   controlando por elo_diff y b2b_diff da 3in4_diff coef=-0.3737, t=-3.086,
   p=0.002 -- significativo incluso encima del back-to-back.

**Definicion de "dias de descanso"** -- convencion estandar de la industria,
explicita para no confundir con la diferencia cruda de fechas: dias entre
el partido actual y el anterior del mismo equipo, MENOS 1. El primer
partido de un equipo en todo el historico no tiene partido anterior --
queda NaN, explicito, no se rellena con un valor inventado.

**Definicion de "3 en 4"** -- ventana de 4 dias de calendario terminando
(inclusive) en la fecha del partido actual (dia_actual-3 hasta dia_actual),
contando cuantos partidos jugo ese equipo en esa ventana INCLUYENDO el
partido actual. Si cuenta >=3, `home_3in4`/`away_3in4` = 1. Implementado
con `rolling('4D', closed='right')` de pandas (equivalente exacto a un
conteo por ventana de fecha, verificado contra una implementacion de
fuerza bruta partido por partido antes de aceptarlo -- 100% de coincidencia
en los 71,092 partidos-equipo del historico. La version rolling corre en
~0.03s vs ~1.0s de la version de fuerza bruta -- se elige la rolling por
velocidad, no cambia el resultado).

Requiere que games_clean.csv ya exista (`clean_nba_data.py`). Agrega
home_rest/away_rest/home_3in4/away_3in4, idempotente (mismo criterio
drop-antes-de-recalcular que el resto del proyecto).

Uso: python -m src.processing.add_nba_rest_features
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"
NEW_COLS = ["home_rest", "away_rest", "home_3in4", "away_3in4"]


def _compute_rest_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """Arma un DataFrame 'largo' (1 fila por equipo por partido) para poder
    calcular la diferencia de fecha con el partido ANTERIOR de ESE equipo
    -- sin importar si ese partido anterior fue de local o visitante (un
    equipo descansa igual entre un partido de visitante y el siguiente de
    local, no hay que separarlos)."""
    home = df[["game_date", "home_team"]].rename(columns={"home_team": "team"})
    away = df[["game_date", "away_team"]].rename(columns={"away_team": "team"})
    long_df = pd.concat([home, away], ignore_index=True).sort_values(["team", "game_date"])

    long_df["days_since_last"] = long_df.groupby("team")["game_date"].diff().dt.days
    long_df["rest_days"] = long_df["days_since_last"] - 1  # back-to-back real = 0

    # Ventana de 4 dias de calendario (incluye el partido actual) -- ver
    # docstring del modulo para la validacion contra fuerza bruta.
    windowed = long_df.set_index("game_date").sort_index()
    windowed["_one"] = 1
    games_in_4d = (
        windowed.groupby("team")["_one"]
        .rolling("4D", closed="right")
        .sum()
        .reset_index()
        .rename(columns={"_one": "games_in_4d"})
    )
    long_df = long_df.merge(games_in_4d, on=["team", "game_date"], how="left")
    long_df["is_3in4"] = (long_df["games_in_4d"] >= 3).astype(float)

    return long_df[["game_date", "team", "rest_days", "is_3in4"]]


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre 'python -m src.processing.clean_nba_data' primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])

    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    rest_long = _compute_rest_by_team(df)

    df = df.merge(
        rest_long.rename(columns={"team": "home_team", "rest_days": "home_rest", "is_3in4": "home_3in4"}),
        on=["game_date", "home_team"], how="left",
    )
    df = df.merge(
        rest_long.rename(columns={"team": "away_team", "rest_days": "away_rest", "is_3in4": "away_3in4"}),
        on=["game_date", "away_team"], how="left",
    )

    n_missing_home = int(df["home_rest"].isna().sum())
    n_missing_away = int(df["away_rest"].isna().sum())
    print(f"Partidos procesados: {len(df)}")
    print(f"Sin home_rest calculable (primer partido de esa franquicia en el historico): {n_missing_home}")
    print(f"Sin away_rest calculable: {n_missing_away}")

    # Chequeo de sanidad OBJETIVO -- back-to-backs (rest=0) deberian ser una
    # fraccion real y no trivial del calendario de NBA (bien documentado en la
    # industria, no un numero inventado para este proyecto).
    b2b_home = (df["home_rest"] == 0).mean()
    b2b_away = (df["away_rest"] == 0).mean()
    print(f"\nChequeo de sanidad -- % de partidos con el LOCAL en back-to-back (rest=0): {b2b_home:.1%}")
    print(f"% de partidos con el VISITANTE en back-to-back (rest=0): {b2b_away:.1%}")
    print(f"(si esto sale en 0% o >50%, revisar el calculo -- back-to-backs son reales y frecuentes en NBA, "
          f"pero no deberian ser la mayoria de los partidos)")

    tin4_home = df["home_3in4"].mean()
    tin4_away = df["away_3in4"].mean()
    print(f"\n% de partidos con el LOCAL en 3-en-4-noches: {tin4_home:.1%}")
    print(f"% de partidos con el VISITANTE en 3-en-4-noches: {tin4_away:.1%}")
    print(f"(3-en-4 deberia ser MAS frecuente que back-to-back solo, porque todo back-to-back que sigue a "
          f"un partido reciente cuenta tambien como 3-en-4 -- si sale MENOS frecuente que b2b, revisar)")

    print(f"\nDistribucion de home_rest (dias de descanso del local, valores mas comunes):")
    print(df["home_rest"].value_counts().sort_index().head(10).to_string())

    df.to_csv(DATA_PATH, index=False)
    print(f"\nGuardado (columnas home_rest/away_rest/home_3in4/away_3in4 agregadas) -> {DATA_PATH}")


if __name__ == "__main__":
    run()
