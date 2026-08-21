"""
Fase 10 (NBA) -- segunda variable real del modelo de NBA (despues de Elo),
mismo camino que ya funciono en NFL: agregar `rest_diff` (diferencia de
dias de descanso entre local y visitante) como segundo regresor de
`nba_margin_model.py`. A diferencia de NFL, este dato NO viene en ningun
archivo descargado -- `nba_api`/`games_clean.csv` no trae dias de descanso
-- asi que se CALCULA directo de las fechas reales de partido que ya
tenemos, no hace falta ninguna fuente nueva.

**Por que importa mas en NBA que en NFL**: NFL juega 1 partido por semana
(descanso casi siempre es 6-7 dias, poca variacion real). NBA juega 82
partidos en ~170-180 dias -- los "back-to-backs" (0 dias de descanso, jugar
dos noches seguidas) son frecuentes y estan documentados en la industria de
analitica de NBA como un factor real de fatiga/rendimiento, mucha mas
variacion real que en NFL.

**Definicion de "dias de descanso" -- convencion estandar de la industria,
explicita para no confundir con la diferencia cruda de fechas**: dias entre
el partido actual y el anterior del mismo equipo, MENOS 1. Un back-to-back
(jugar hoy habiendo jugado ayer) da rest=0, jugar cada 2 dias da rest=1,
etc. -- NO se usa la diferencia de dias cruda (eso daria back-to-back=1,
confuso). El primer partido de un equipo en todo el historico no tiene
partido anterior -- queda NaN, explicito, no se rellena con un valor
inventado (0 dias de descanso para el primer partido de la franquicia
seria un dato falso, no un dato faltante real).

Requiere que games_clean.csv ya exista (`clean_nba_data.py`). Agrega
home_rest/away_rest, idempotente (mismo criterio drop-antes-de-recalcular
que el resto del proyecto).

Uso: python -m src.processing.add_nba_rest_features
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"
NEW_COLS = ["home_rest", "away_rest"]


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
    return long_df[["game_date", "team", "rest_days"]]


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
        rest_long.rename(columns={"team": "home_team", "rest_days": "home_rest"}),
        on=["game_date", "home_team"], how="left",
    )
    df = df.merge(
        rest_long.rename(columns={"team": "away_team", "rest_days": "away_rest"}),
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

    print(f"\nDistribucion de home_rest (dias de descanso del local, valores mas comunes):")
    print(df["home_rest"].value_counts().sort_index().head(10).to_string())

    df.to_csv(DATA_PATH, index=False)
    print(f"\nGuardado (columnas home_rest/away_rest agregadas) -> {DATA_PATH}")


if __name__ == "__main__":
    run()
