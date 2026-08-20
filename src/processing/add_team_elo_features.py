"""
Motivado por el resultado real de Elo en tenis esta misma sesión
(2026-08-19/20): agregar un rating de fuerza ponderado por el RIVAL VENCIDO
(Elo walk-forward) mejoró el modelo de tenis de forma no trivial (Brier ATP
0.2189->0.2158, WTA 0.2178->0.2152) y destrabó la primera meseta de ROI
positivo real de todo el trabajo de tenis. El modelo de fútbol (v4, la
referencia en las 4 ligas europeas desde Fase 8) nunca probó esto -- sus
dos intentos de agregar señal nueva encima de v4 fallaron (v5: separar
ataque/defensa de los tiros al arco; v6: agregar corners), pero NINGUNO de
los dos agrega información sobre la fuerza del rival reciente -- ambos son
variantes de "cuánto generó este equipo", no de "contra quién lo generó".
`C(team)`/`C(opponent)` en el GLM de v4 sí capturan fuerza relativa, pero
como efecto fijo ESTÁTICO sobre toda la ventana de entrenamiento (ajustado
por recencia vía freq_weights) -- no como una trayectoria explícita que se
mueve partido a partido cada vez que un equipo le gana a un rival fuerte o
débil, que es justamente lo que Elo aporta y lo que forma/tiros/corners no
capturan. Se prueba esta hipótesis nueva en `poisson_model_v7.py`, sin
tocar nada de lo ya verificado.

Elo estándar de fútbol (mismo criterio "no tunear todavía" que ya se usó
en tenis, mismo disclaimer explícito): `INITIAL_ELO=1500`, `K_FACTOR=32`.
Se agrega `HOME_ADVANTAGE=100` (constante estándar de sistemas Elo de
fútbol, ej. World Football Elo Ratings / eloratings.net) SOLO en el
cálculo de la expectativa, no en el rating en sí -- el sistema se mantiene
zero-sum igual. Resultado real R: 1.0 si gana el equipo, 0.5 si empatan,
0.0 si pierde (a diferencia de tenis, que no tiene empate).

Walk-forward real y secuencial (NO vectorizable con rolling como
home_recent_st_diff) -- mismo patrón anti-fuga que
add_tennis_form_features.py: el rating ANTES de cada partido se guarda en
la fila, y el diccionario de estado se actualiza DESPUÉS.

Este script es completamente ADITIVO y separado de
add_team_form_features.py (no lo modifica, no depende de él) -- agrega
solo 2 columnas nuevas (home_team_elo/away_team_elo) a matches_clean.csv,
mismo criterio de idempotencia real (drop-antes-de-recalcular, ver el
bug ya documentado y corregido en add_team_form_features.py) para no
arriesgar nada del pipeline ya verificado.

Uso: python -m src.processing.add_team_elo_features
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

INITIAL_ELO = 1500.0
K_FACTOR = 32.0          # estandar, sin tunear -- mismo criterio y mismo valor que tenis.
HOME_ADVANTAGE = 100.0   # estandar de sistemas Elo de futbol, solo afecta la expectativa.

NEW_COLS = ["home_team_elo", "away_team_elo"]


def _elo_expected(rating_home: float, rating_away: float) -> float:
    """Probabilidad esperada de que gane el LOCAL (resultado=1), con ventaja de local
    aplicada solo a la expectativa -- el sistema de ratings en si sigue siendo zero-sum."""
    return 1.0 / (1.0 + 10.0 ** (-((rating_home + HOME_ADVANTAGE) - rating_away) / 400.0))


def _add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    elo = defaultdict(lambda: INITIAL_ELO)

    home_elo_before, away_elo_before = [], []
    for row in df.itertuples(index=False):
        home_team, away_team = row.HomeTeam, row.AwayTeam
        rating_home = elo[home_team]
        rating_away = elo[away_team]
        home_elo_before.append(rating_home)
        away_elo_before.append(rating_away)

        if row.FTR == "H":
            result_home = 1.0
        elif row.FTR == "D":
            result_home = 0.5
        else:
            result_home = 0.0

        expected_home = _elo_expected(rating_home, rating_away)
        elo[home_team] = rating_home + K_FACTOR * (result_home - expected_home)
        elo[away_team] = rating_away + K_FACTOR * ((1.0 - result_home) - (1.0 - expected_home))

    df["home_team_elo"] = home_elo_before
    df["away_team_elo"] = away_elo_before
    return df, dict(elo)


def run(league_key: str) -> None:
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe {path} -- corre clean_data.py primero.")
        return

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Idempotente de verdad -- mismo criterio que add_team_form_features.py: si ya
    # existen las columnas de una corrida anterior, se descartan ANTES de recalcular.
    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    df, final_elo = _add_elo_features(df)

    print(f"\n=== {league_key} ===")
    print(f"Partidos procesados: {len(df)}")
    elo_series = pd.Series(final_elo)
    print(f"Chequeo de sanidad -- Elo final: media={elo_series.mean():.1f} (esperado ~1500, zero-sum), "
          f"desvio={elo_series.std():.1f}")
    top5 = elo_series.sort_values(ascending=False).head(5)
    print(f"Top 5 equipos por Elo final (deberia coincidir con equipos realmente dominantes de la liga):")
    for team, rating in top5.items():
        print(f"  {team}: {rating:.0f}")

    df.to_csv(path, index=False)
    print(f"Guardado (columnas home_team_elo/away_team_elo agregadas) -> {path}")


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)