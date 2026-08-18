"""
Nueva linea de investigacion (Fase 8/8b, "Proximos pasos" punto 2): por que
EPL tiene una regla de staking claramente rentable (+2.11%, mejorable a
+4.49%) y los otros 4 mercados no, si el mecanismo de fondo (winner's curse,
mercado eficiente) es el mismo en los 5?

Hipotesis nueva, aportada por el hallazgo de MLS (ver roadmap, Fase 8b):
MLS esta disenada para paridad competitiva (salary cap, draft, playoffs) y
casi no genera casos Tier 1 (>=80%) pese a calibrar perfecto -- si la
paridad competitiva (o su ausencia) tambien varia entre las 4 ligas
europeas, podria ser una variable estructural detras de por que unas
generan mas señal explotable que otras.

Este script NO asume la respuesta -- mide paridad competitiva de forma
directa y objetiva sobre datos que ya existen (matches_clean.csv de las 5
ligas, ya normalizado a la misma convencion de columnas desde Fase 8b) y
deja la tabla para cruzar a mano contra los ROI ya documentados en el
roadmap. Ya se sabe (ver roadmap) que La Liga -- historicamente de las mas
top-heavy de Europa -- contradice parcialmente esta hipotesis (es la peor
en Tier 1 pese a NO ser la mas pareja), asi que este script es para ver el
panorama completo, no para confirmar una corazonada.

Metricas por liga, promediadas sobre todas las temporadas disponibles:
- Desvio estandar de PPG (puntos por partido jugado) entre equipos de la
  misma temporada -- que tan pareja es la tabla esa temporada.
- Coeficiente de Gini de esa misma distribucion de PPG -- mide concentracion
  de forma mas robusta que el desvio estandar solo.
- Numero de campeones distintos / temporadas disponibles -- dominancia de
  largo plazo (bajo valor = pocos equipos ganan siempre).

Fuente: 'matches_clean.csv' de cada liga (clean_data.py). No entrena ni
simula nada -- es aritmetica de tabla de posiciones.

Salida: data/runs/competitive_balance_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ALL_LEAGUE_KEYS = list(LEAGUES.keys()) + ["MLS"]


def _season_standings(season_df: pd.DataFrame) -> pd.Series:
    """PPG (puntos por partido jugado) por equipo, para una sola temporada."""
    records = []
    for _, row in season_df.iterrows():
        if row["FTR"] == "H":
            home_pts, away_pts = 3, 0
        elif row["FTR"] == "A":
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1
        records.append((row["HomeTeam"], home_pts))
        records.append((row["AwayTeam"], away_pts))

    df = pd.DataFrame(records, columns=["team", "points"])
    played = df.groupby("team").size()
    total_pts = df.groupby("team")["points"].sum()
    ppg = (total_pts / played).rename("ppg")
    return ppg


def _gini(values: np.ndarray) -> float:
    """Coeficiente de Gini estandar, 0 = perfectamente parejo, 1 = totalmente concentrado."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return float("nan")
    cumulative = np.cumsum(values)
    return float((2 * np.sum((np.arange(1, n + 1)) * values) - (n + 1) * cumulative[-1]) / (n * cumulative[-1]))


def _champion(ppg: pd.Series) -> str:
    return ppg.idxmax()


def analyze_league(league_key: str) -> dict:
    print(f"\n=== {league_key} ===")
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not path.exists():
        print(f"[SKIP] No existe {path}.")
        return {}

    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    seasons = sorted(df["season"].unique())

    std_devs, ginis, champions = [], [], []
    for season in seasons:
        season_df = df[df["season"] == season]
        if season_df.empty:
            continue
        ppg = _season_standings(season_df)
        if len(ppg) < 2:
            continue
        std_devs.append(ppg.std())
        ginis.append(_gini(ppg.values))
        champions.append(_champion(ppg))

    n_seasons = len(std_devs)
    if n_seasons == 0:
        print(f"[SKIP] {league_key}: sin temporadas evaluables.")
        return {}

    n_distinct_champions = len(set(champions))
    champion_concentration = 1.0 - (n_distinct_champions / n_seasons)

    result = {
        "league_key": league_key,
        "n_seasons": n_seasons,
        "avg_std_ppg": float(np.mean(std_devs)),
        "avg_gini_ppg": float(np.mean(ginis)),
        "n_distinct_champions": n_distinct_champions,
        "champion_concentration": champion_concentration,
    }
    print(f"Temporadas evaluadas: {n_seasons}")
    print(f"Desvio estandar promedio de PPG entre equipos: {result['avg_std_ppg']:.4f}")
    print(f"Gini promedio de PPG: {result['avg_gini_ppg']:.4f}")
    print(f"Campeones distintos: {n_distinct_champions} de {n_seasons} temporadas "
          f"(concentracion de titulo: {champion_concentration:.2%})")
    return result


def run() -> None:
    results = []
    for league_key in ALL_LEAGUE_KEYS:
        result = analyze_league(league_key)
        if result:
            results.append(result)

    if not results:
        print("[SKIP] No se pudo evaluar ninguna liga.")
        return

    summary = pd.DataFrame(results).set_index("league_key")
    summary = summary.sort_values("avg_std_ppg", ascending=False)

    print("\n=== Comparacion de paridad competitiva entre las 5 ligas "
          "(ordenado de MENOS pareja a MAS pareja) ===")
    print(summary.round(4).to_string())

    print("\nReferencia rapida -- ROI ya documentado en el roadmap (mejor regla de staking, "
          "barrido de 24 combinaciones), para cruzar a mano contra la tabla de arriba:")
    print("  EPL:         +4.49%  (rentable)")
    print("  LALIGA:      +0.12%  (empate)")
    print("  SERIEA:      -3.60%  (no rentable)")
    print("  BUNDESLIGA:  -3.05%  (no rentable)")
    print("  MLS:         -3.50%  (no rentable)")

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "competitive_balance_summary.csv"
    summary.to_csv(out_path)
    print(f"\nGuardado -> {out_path}")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio de una hipotesis, "
          "no un modelo nuevo) -- mismo criterio que selection_bias_check.py.")


if __name__ == "__main__":
    run()