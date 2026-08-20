"""
Fase 10 -- limpieza de NFL, mismo patron "un script de limpieza por deporte"
que clean_data.py (futbol) y clean_tennis_data.py (tenis), pero con 4
diferencias reales confirmadas en nfl_data_loader.py que este script maneja
explicitamente en vez de asumir que el patron anterior aplica igual:

1. CUOTAS AMERICANAS, no decimales -- la conversion a probabilidad implicita
   usa la formula de odds americanas, NO 1/cuota_decimal que usa el resto
   del proyecto (futbol/tenis/MLS). Formula estandar:
     odds > 0:  prob_implicita = 100 / (odds + 100)
     odds < 0:  prob_implicita = -odds / (-odds + 100)
   Se aisla en su propia funcion (`_american_to_prob`) precisamente para no
   mezclar las dos formulas por error -- bug que ya se previno a proposito,
   ver docstring de nfl_data_loader.py.

2. SIN CLV -- `load_schedules()` trae un UNICO snapshot de moneyline/
   spread/total por partido, no apertura+cierre como football-data.co.uk.
   No hay proxy de CLV que calcular aca, a diferencia de clean_data.py. La
   provenance de la casa de apuestas tampoco esta confirmada (pregunta
   abierta documentada en nfl_data_loader.py) -- se guarda la probabilidad
   de mercado igual (es un benchmark real de todos modos) pero SIN
   pretender que es "cierre de un libro sharp conocido" como Pinnacle en
   futbol.

3. CORTE DE CONFIABILIDAD DISTINTO PARA MONEYLINE VS. SPREAD -- confirmado
   por probe() real (nfl_data_loader.py, corrida del usuario 2026-08-20):
   moneyline solo confiable desde 2010 (0% en 1999-2005, parcial e
   irregular en 2006-2009), pero spread_line/total_line tienen cobertura
   COMPLETA desde 1999. Este script NO fuerza un unico corte -- guarda TODO
   el historico disponible (1999-2026) y agrega una columna booleana
   explicita `reliable_moneyline` (season >= RELIABLE_ODDS_START_SEASON)
   para que cada modelo futuro (moneyline vs. distribucion de spread)
   filtre segun lo que realmente necesita, en vez de tirar datos de spread
   que si son buenos solo porque el moneyline de esos anos no lo es.

4. NFL SI PERMITE EMPATE (raro, del orden de 1 cada varias temporadas) --
   a diferencia de tenis (nunca empata). Se maneja explicitamente como
   resultado 'T' en vez de asumir un resultado binario H/A.

Consolida data/raw/NFL/nfl_*.csv -> data/processed/NFL/matches_clean.csv.

Uso: python -m src.processing.clean_nfl_data
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw" / "NFL"
OUT_DIR = PROCESSED_DATA_DIR / "NFL"

# Confirmado por probe() real (nfl_data_loader.py, 2026-08-20): antes de esta
# temporada la cobertura de MONEYLINE es 0% o muy irregular. spread_line/
# total_line NO tienen este problema -- son confiables desde 1999. No es una
# opinion, es un conteo real ya confirmado dos veces (sandbox + corrida real
# del usuario).
RELIABLE_ODDS_START_SEASON = 2010


def _american_to_prob(odds: float) -> float:
    """Formula estandar de odds americanas -- NO usar 1/odds (esa es la
    formula de odds decimales que usa el resto del proyecto)."""
    if pd.isna(odds):
        return float("nan")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def _american_to_b(odds: float) -> float:
    """Ganancia NETA por unidad de stake ('b' en la formula estandar de
    Kelly: f* = (b*p - q)/b) a partir de odds americanas -- distinto de
    (cuota_decimal - 1) que usa el resto del proyecto con cuotas decimales.
      odds > 0: b = odds / 100
      odds < 0: b = 100 / abs(odds)
    Usado por economic_backtest_nfl.py -- agregado aca (no en ese script)
    para mantener una sola fuente de verdad de las formulas de odds
    americanas, junto a _american_to_prob. Funcion nueva, sin efecto en el
    CSV ya generado por clean_nfl_data.py -- NO requiere re-correr la
    limpieza."""
    if pd.isna(odds):
        return float("nan")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def _remove_vig_two_way(prob_a: float, prob_b: float) -> tuple:
    """Normaliza dos probabilidades implicitas (con margen) para que sumen 1
    -- mismo criterio 'no-vig' ya usado en clean_data.py para las cuotas 1X2
    de futbol, aplicado aca al mercado de 2 vias del moneyline de NFL (la
    COTIZACION es de 2 vias aunque el RESULTADO real pueda empatar -- el
    empate no se cotiza aparte en el moneyline de NFL)."""
    if pd.isna(prob_a) or pd.isna(prob_b):
        return float("nan"), float("nan")
    total = prob_a + prob_b
    if total <= 0:
        return float("nan"), float("nan")
    return prob_a / total, prob_b / total


def _load_all_seasons() -> pd.DataFrame:
    files = sorted(RAW_DATA_DIR.glob("nfl_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No hay CSVs en {RAW_DATA_DIR} -- corre "
            "'python -m src.ingestion.nfl_data_loader --download-all' primero."
        )
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def _outcome(row) -> str:
    if row["home_score"] > row["away_score"]:
        return "H"
    if row["home_score"] < row["away_score"]:
        return "A"
    return "T"


def run() -> None:
    df = _load_all_seasons()
    df["gameday"] = pd.to_datetime(df["gameday"])

    # Solo partidos ya jugados (con resultado real) -- descarta partidos
    # futuros programados que puedan venir en el CSV de la temporada en
    # curso (2026, todavia sin terminar segun el probe).
    before = len(df)
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] {dropped} partidos sin jugar todavia (sin resultado) descartados.")

    df["FTR"] = df.apply(_outcome, axis=1)
    n_ties = int((df["FTR"] == "T").sum())
    print(f"Empates reales en el historico: {n_ties} de {len(df)} ({n_ties / len(df):.2%}) "
          f"-- NFL SI permite empate, a diferencia de tenis.")

    df["point_margin"] = df["home_score"] - df["away_score"]  # positivo = gano el local

    # Probabilidad implicita de moneyline (formula AMERICANA, no decimal) + no-vig.
    df["home_ml_prob_raw"] = df["home_moneyline"].apply(_american_to_prob)
    df["away_ml_prob_raw"] = df["away_moneyline"].apply(_american_to_prob)
    novig = df.apply(lambda r: _remove_vig_two_way(r["home_ml_prob_raw"], r["away_ml_prob_raw"]), axis=1)
    df["market_prob_home"], df["market_prob_away"] = zip(*novig)

    df["reliable_moneyline"] = df["season"] >= RELIABLE_ODDS_START_SEASON

    n_with_ml = int(df["market_prob_home"].notna().sum())
    n_reliable = int(df["reliable_moneyline"].sum())
    print(f"Partidos con moneyline valido (no-vig calculable): {n_with_ml} de {len(df)}")
    print(f"Partidos en el rango confiable de moneyline (season>={RELIABLE_ODDS_START_SEASON}): {n_reliable}")

    # Chequeo de sanidad -- NFL tiene ventaja de local documentada historicamente
    # (fuentes publicas como Pro Football Reference la ubican en general arriba
    # de 50%, con tendencia a bajar en temporadas recientes). No se fija un
    # numero exacto de referencia para no inventar precision que no tenemos --
    # se imprime el valor real para que el usuario lo audite a ojo (algo muy
    # fuera de 50-60% si seria señal de un bug de mapeo home/away).
    home_win_rate = (df["FTR"] == "H").mean()
    print(f"Chequeo de sanidad -- % de victorias del local (historico completo, "
          f"1999-{df['season'].max()}): {home_win_rate:.2%} "
          f"(si esto sale muy fuera de ~50-60%, revisar mapeo home/away antes de seguir).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "matches_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path} ({len(df)} partidos, temporadas {df['season'].min()}-{df['season'].max()})")


if __name__ == "__main__":
    run()