"""
Escalamiento a Tenis, paso 3 (tras tennis_data_loader.py y
clean_tennis_data.py, este último ya corrido y confirmado 2026-08-19:
ATP 28,424 partidos limpios, WTA 26,590, chequeo de sanidad del
reetiquetado neutral en 50.13%/49.75% -- sin sesgo detectable).

Feature engineering walk-forward de forma reciente por jugador. Misma
disciplina anti-fuga que add_team_form_features.py en fútbol: cada
feature de un partido se calcula usando SOLO partidos anteriores en el
tiempo de ese jugador -- nunca el resultado del partido que se está
prediciendo ni partidos futuros. Esto se garantiza en el código
recorriendo los partidos en orden cronológico y actualizando el
historial de cada jugador DESPUÉS de calcular sus features para el
partido actual, nunca antes.

A diferencia de fútbol, el objeto de features es un JUGADOR, no un
equipo, y el partido es Player1 vs Player2 en orden ALFABÉTICO (neutral,
sin relación con quién ganó -- ver clean_tennis_data.py). Por eso todas
las features se calculan por separado para Player1 y Player2, más una
versión "diff" (Player1 - Player2) pensada para uso directo en un modelo
de regresión logística (metodología Benter -- el análogo de tenis al
Poisson de goles de fútbol es más simple: resultado binario, sin empate,
exactamente el tipo de problema para el que Benter diseñó su regresión
logística original en carreras de caballos). Las features diff son
robustas al orden arbitrario Player1/Player2 por construcción, en vez de
depender de que el modelo aprenda solo esa simetría.

Features agregadas (sufijo _P1/_P2 en el nombre real de columna es
"Player1_"/"Player2_", más su diff):
1. WinRate20: win rate en los últimos 20 partidos del jugador (cualquier
   superficie) antes de la fecha del partido actual.
2. SurfaceWinRate: win rate en los últimos 15 partidos del jugador EN LA
   MISMA SUPERFICIE del partido actual (si tiene menos de 15 previos en
   esa superficie, usa los que tenga -- no fuerza el mínimo, y queda NaN
   si no tiene ninguno).
3. MatchesPlayed: conteo total de partidos previos del jugador en el
   dataset -- proxy de cuánta historia hay detrás de las otras features
   (más historia = features más confiables).
4. DaysRest: días desde el partido anterior del jugador (proxy de
   descanso/fatiga; NaN si es el primer partido del jugador en el
   dataset).
5. H2H_WinRate: win rate del jugador específicamente contra ESTE
   oponente, en enfrentamientos previos entre ambos (NaN si nunca se
   cruzaron antes -- head-to-head es un dataset chico y ruidoso por
   definición, se deja como feature aparte para que el modelo pueda
   aprender a no confiarle demasiado peso con pocos partidos, en vez de
   forzarlo a 50% por defecto).
6. rank_diff: Player2_Rank - Player1_Rank (positivo = Player1 mejor
   rankeado -- OJO con el signo, en tenis un ranking más BAJO es mejor).
   Se calcula directo de las columnas ya neutralizadas por
   clean_tennis_data.py, no requiere historial.
7. pts_diff: Player1_Pts - Player2_Pts (positivo = Player1 con más
   puntos ATP/WTA -- acá sí, más puntos es mejor, signo opuesto a rank).
8. Elo / Elo_Surface (agregado 2026-08-19, ver roadmap): rating Elo
   walk-forward por jugador, general y por superficie, ACTUALIZADO
   PARTIDO A PARTIDO -- motivado porque tras confirmar que el modelo v1
   (win rate + ranking oficial + contexto de partido) apenas mejoraba
   sobre el mercado, se identificó un hueco real: ni el ranking oficial
   ni el win rate reciente ponderan la FUERZA DEL RIVAL. Ganarle a un
   top-10 cuenta igual que ganarle a un jugador de clasificación en
   WinRate20. Elo sí lo pondera -- es el enfoque estándar en la
   literatura de predicción de tenis (ej. el Elo de tenis de
   FiveThirtyEight) precisamente porque corrige ese hueco. Se parte de
   1500 (convención estándar de Elo) para todo jugador nuevo, con
   K=32 (factor de ajuste estándar, no tuneado todavía -- ver nota en
   el código). El Elo por superficie es un rating INDEPENDIENTE del
   general (un jugador puede tener Elo general alto y Elo de polvo de
   ladrillo bajo si históricamente rinde peor ahí) -- mismo espíritu que
   SurfaceWinRate pero con la ponderación por fuerza del rival que
   WinRate/SurfaceWinRate no tienen.

Requiere haber corrido antes:
    python -m src.processing.clean_tennis_data --tours ATP,WTA

Salida (no sobreescribe matches_clean.csv, crea un archivo nuevo):
    data/processed/TENNIS_ATP/matches_features.csv
    data/processed/TENNIS_WTA/matches_features.csv
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

WIN_RATE_WINDOW = 20
SURFACE_WIN_RATE_WINDOW = 15

INITIAL_ELO = 1500.0  # convencion estandar de Elo -- todo jugador nuevo arranca acá
K_FACTOR = 32.0        # factor de ajuste estandar (mismo valor por defecto que ajedrez/otros deportes) --
                       # NO TUNEADO todavia para tenis especificamente, punto de partida razonable, no una
                       # regla confirmada -- misma disciplina del proyecto de no asumir que un valor por
                       # defecto es el optimo sin probarlo.


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Probabilidad esperada de que A le gane a B segun la formula estandar de Elo."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _win_rate(matches: list, n: int = None) -> float:
    if n is not None:
        matches = matches[-n:]
    if not matches:
        return np.nan
    return sum(m["won"] for m in matches) / len(matches)


def _normalize_player1_won(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza Player1_Won a bool real -- no confiar en que pandas siempre
    infiera bool nativo al releer el CSV (si llega como texto, bool("False")
    es True en Python -- hay que mapear explícitamente, no castear directo)."""
    if df["Player1_Won"].dtype != bool:
        print("  [AVISO] Player1_Won no se leyó como booleano nativo -- normalizando explícitamente desde texto.")
        df["Player1_Won"] = (
            df["Player1_Won"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        )
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    df = _normalize_player1_won(df)

    history = defaultdict(list)      # player -> [{date, surface, won}, ...], orden cronologico
    h2h_history = defaultdict(list)  # frozenset({p1,p2}) -> [{date, winner}, ...]
    elo_overall = defaultdict(lambda: INITIAL_ELO)              # player -> rating Elo general
    elo_by_surface = defaultdict(lambda: defaultdict(lambda: INITIAL_ELO))  # player -> {surface: rating}

    feature_rows = []

    for row in df.itertuples(index=False):
        p1, p2 = row.Player1, row.Player2
        date = row.Date
        surface = getattr(row, "Surface", None)

        p1_hist = history[p1]
        p2_hist = history[p2]

        # -- win rate, cualquier superficie (SOLO historial previo -- p1_hist/p2_hist
        #    todavia no incluyen el partido actual, se actualizan mas abajo) --
        p1_wr20 = _win_rate(p1_hist, WIN_RATE_WINDOW)
        p2_wr20 = _win_rate(p2_hist, WIN_RATE_WINDOW)

        # -- win rate en la misma superficie --
        p1_surf_hist = [m for m in p1_hist if m["surface"] == surface]
        p2_surf_hist = [m for m in p2_hist if m["surface"] == surface]
        p1_wr_surf = _win_rate(p1_surf_hist, SURFACE_WIN_RATE_WINDOW)
        p2_wr_surf = _win_rate(p2_surf_hist, SURFACE_WIN_RATE_WINDOW)

        # -- volumen de historia --
        p1_n = len(p1_hist)
        p2_n = len(p2_hist)

        # -- descanso --
        p1_days_rest = (date - p1_hist[-1]["date"]).days if p1_hist else np.nan
        p2_days_rest = (date - p2_hist[-1]["date"]).days if p2_hist else np.nan

        # -- head to head --
        key = frozenset((p1, p2))
        h2h_hist = h2h_history[key]
        if h2h_hist:
            p1_h2h_wr = sum(1 for m in h2h_hist if m["winner"] == p1) / len(h2h_hist)
            p2_h2h_wr = 1.0 - p1_h2h_wr
        else:
            p1_h2h_wr = np.nan
            p2_h2h_wr = np.nan

        # -- Elo, general y por superficie (SOLO el rating ANTES de este partido --
        #    se actualiza mas abajo, despues de calcular features, mismo patron
        #    que el resto) --
        p1_elo = elo_overall[p1]
        p2_elo = elo_overall[p2]
        has_surface = pd.notna(surface)
        p1_elo_surf = elo_by_surface[p1][surface] if has_surface else np.nan
        p2_elo_surf = elo_by_surface[p2][surface] if has_surface else np.nan

        feature_rows.append({
            "Player1_WinRate20": p1_wr20, "Player2_WinRate20": p2_wr20,
            "Player1_SurfaceWinRate": p1_wr_surf, "Player2_SurfaceWinRate": p2_wr_surf,
            "Player1_MatchesPlayed": p1_n, "Player2_MatchesPlayed": p2_n,
            "Player1_DaysRest": p1_days_rest, "Player2_DaysRest": p2_days_rest,
            "Player1_H2H_WinRate": p1_h2h_wr, "Player2_H2H_WinRate": p2_h2h_wr,
            "H2H_MatchesPlayed": len(h2h_hist),
            "Player1_Elo": p1_elo, "Player2_Elo": p2_elo,
            "Player1_Elo_Surface": p1_elo_surf, "Player2_Elo_Surface": p2_elo_surf,
        })

        # -- actualizar historial DESPUES de calcular features -- nunca antes,
        #    es la garantia central de que no hay fuga de informacion. --
        won_p1 = bool(row.Player1_Won)
        p1_hist.append({"date": date, "surface": surface, "won": won_p1})
        p2_hist.append({"date": date, "surface": surface, "won": not won_p1})
        h2h_hist.append({"date": date, "winner": p1 if won_p1 else p2})

        # -- actualizar Elo DESPUES de calcular features -- misma disciplina --
        expected_p1 = _elo_expected(p1_elo, p2_elo)
        actual_p1 = 1.0 if won_p1 else 0.0
        elo_overall[p1] = p1_elo + K_FACTOR * (actual_p1 - expected_p1)
        elo_overall[p2] = p2_elo + K_FACTOR * ((1.0 - actual_p1) - (1.0 - expected_p1))
        if has_surface:
            expected_p1_surf = _elo_expected(p1_elo_surf, p2_elo_surf)
            elo_by_surface[p1][surface] = p1_elo_surf + K_FACTOR * (actual_p1 - expected_p1_surf)
            elo_by_surface[p2][surface] = p2_elo_surf + K_FACTOR * ((1.0 - actual_p1) - (1.0 - expected_p1_surf))

    feat_df = pd.DataFrame(feature_rows)
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # -- features diff (P1 - P2), robustas al orden arbitrario Player1/Player2 --
    out["WinRate20_diff"] = out["Player1_WinRate20"] - out["Player2_WinRate20"]
    out["SurfaceWinRate_diff"] = out["Player1_SurfaceWinRate"] - out["Player2_SurfaceWinRate"]
    out["MatchesPlayed_diff"] = out["Player1_MatchesPlayed"] - out["Player2_MatchesPlayed"]
    out["DaysRest_diff"] = out["Player1_DaysRest"] - out["Player2_DaysRest"]
    out["H2H_WinRate_diff"] = out["Player1_H2H_WinRate"] - out["Player2_H2H_WinRate"]

    if "Player1_Rank" in out.columns and "Player2_Rank" in out.columns:
        out["rank_diff"] = out["Player2_Rank"] - out["Player1_Rank"]  # positivo = Player1 mejor rankeado (rank mas bajo)
    else:
        print("  [AVISO] no hay columnas Player1_Rank/Player2_Rank -- no se calcula rank_diff.")

    if "Player1_Pts" in out.columns and "Player2_Pts" in out.columns:
        out["pts_diff"] = out["Player1_Pts"] - out["Player2_Pts"]  # positivo = Player1 con mas puntos
    else:
        print("  [AVISO] no hay columnas Player1_Pts/Player2_Pts -- no se calcula pts_diff.")

    out["Elo_diff"] = out["Player1_Elo"] - out["Player2_Elo"]
    out["Elo_Surface_diff"] = out["Player1_Elo_Surface"] - out["Player2_Elo_Surface"]

    # se devuelve tambien el rating FINAL (post-ultimo partido) de cada jugador --
    # necesario para el chequeo de sanidad en run(); usar el valor de la columna
    # Player1_Elo/Player2_Elo por fila daria el rating ANTES de ese partido, no el
    # final real del jugador -- por eso se toma directo del diccionario acumulado.
    final_elo = dict(elo_overall)
    return out, final_elo


def run(tour: str) -> None:
    print(f"\n=== {tour.upper()} ===")
    in_path = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}" / "matches_clean.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"No existe {in_path}. Corre 'python -m src.processing.clean_tennis_data --tours {tour}' primero."
        )

    df = pd.read_csv(in_path, parse_dates=["Date"])
    print(f"Cargados {len(df)} partidos de {in_path}")

    out, final_elo = _add_features(df)

    n_players = len(set(out["Player1"]) | set(out["Player2"]))
    print(f"  Jugadores distintos detectados: {n_players}")
    if "rank_diff" in out.columns:
        print(f"  Cobertura de rank_diff (no-NaN): {out['rank_diff'].notna().mean():.2%}")
    if "pts_diff" in out.columns:
        print(f"  Cobertura de pts_diff (no-NaN): {out['pts_diff'].notna().mean():.2%}")
    print(f"  Cobertura de H2H (partidos con >=1 cruce previo entre ambos): {(out['H2H_MatchesPlayed'] > 0).mean():.2%}")
    print(f"  Cobertura de SurfaceWinRate en ambos jugadores (no-NaN): "
          f"{(out['Player1_SurfaceWinRate'].notna() & out['Player2_SurfaceWinRate'].notna()).mean():.2%}")

    # -- chequeo de sanidad de Elo: el rating FINAL real (tomado del diccionario
    #    acumulado, no de una columna del DataFrame -- una fila solo tiene el
    #    rating ANTES de ese partido) deberia separar bien a los mejores
    #    jugadores del dataset -- si todos terminan cerca de 1500, el sistema no
    #    esta discriminando nada (bug real a revisar, no cosmetico) --
    elo_series = pd.Series(final_elo)
    top5 = elo_series.sort_values(ascending=False).head(5)
    print(f"  Chequeo de sanidad de Elo: rating promedio final {elo_series.mean():.0f} "
          f"(esperado cerca de {INITIAL_ELO:.0f} en agregado -- Elo es de suma-cero relativa), "
          f"desvío estándar {elo_series.std():.0f} (si es chico, el sistema no está "
          f"discriminando jugadores -- revisar K_FACTOR).")
    print(f"  Top 5 Elo final más alto del dataset: {top5.round(0).to_dict()}")

    out_path = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}" / "matches_features.csv"
    out.to_csv(out_path, index=False)
    print(f"  Guardado -> {out_path} ({len(out)} partidos, {len(out.columns)} columnas)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature engineering walk-forward de tenis: forma reciente, superficie, descanso, H2H, ranking."
    )
    parser.add_argument("--tours", type=str, default="ATP,WTA", help="Tours a procesar, separados por coma (default: ATP,WTA).")
    args = parser.parse_args()

    for tour in args.tours.split(","):
        try:
            run(tour)
        except FileNotFoundError as e:
            print(f"[SKIP] {tour}: {e}")