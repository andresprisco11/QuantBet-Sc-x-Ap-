"""
Módulo de limpieza y feature base: toma los CSV crudos de data/raw/<liga>/,
los consolida, extrae las cuotas de las casas prioritarias y calcula
probabilidades implícitas normalizadas (sin margen de la casa) para
apertura y cierre — el insumo base para medir CLV (Closing Line Value).

Salida: un único CSV limpio por liga en data/processed/<liga>/matches_clean.csv

Fix 2026-08-18 (Fase 8b, MLS): MLS trae los mismos datos (fecha, equipos,
goles, resultado, cuotas) pero bajo OTROS NOMBRES DE COLUMNA (confirmado
con datos reales del crudo ya descargado por mls_loader.py: 'Home'/'Away'/
'HG'/'AG'/'Res'/'Season' en vez de 'HomeTeam'/'AwayTeam'/'FTHG'/'FTAG'/
'FTR'/'season'), y SIN cuotas de apertura (ni Pinnacle PSH/PSD/PSA ni
Bet365 B365H/B365D/B365A) — solo tiene cierre. Dos cambios, ambos
deliberadamente genéricos en vez de "if MLS" repartidos por todo el
archivo:
1. `_normalize_mls_raw()`: renombra las columnas de MLS a la convención
   estándar UNA sola vez, al entrar a `clean_league()`. Así el resto de
   esta función y absolutamente todo el pipeline aguas abajo (features,
   modelo, evaluación — que ya asumen FTR/FTHG/FTAG/HomeTeam/AwayTeam en
   todos lados) sigue funcionando sin ningún cambio adicional.
2. `add_bookmaker_features()`: antes, una casa de apuestas se OMITÍA por
   completo si le faltaba CUALQUIERA de las 6 columnas (apertura + cierre).
   Eso habría descartado silenciosamente toda la información de cierre de
   MLS (que sí existe y sí sirve) solo porque no tiene apertura. Ahora se
   calculan probabilidades de CIERRE si están disponibles, y las de
   APERTURA + CLV solo si también están disponibles — si falta la apertura,
   se avisa explícitamente que no hay CLV para esa liga/casa, no se inventa
   un proxy. Backward-compatible: para las 4 ligas europeas (que sí tienen
   ambas) el resultado es idéntico a antes.
"""

import difflib
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PRIORITY_BOOKMAKERS, PROCESSED_DATA_DIR, RAW_DATA_DIR

# Columnas base que siempre deben existir en un CSV de football-data.co.uk
CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Mapeo de nombres de columna del crudo de MLS (ver src/ingestion/mls_loader.py)
# a la convención estándar que usa el resto del pipeline. Confirmado con
# datos reales el 2026-08-18 -- si el sitio cambia el esquema, el chequeo
# en _normalize_mls_raw() lo va a detectar en vez de fallar en silencio.
MLS_RAW_COLUMN_MAP = {
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
    "Season": "season",
}


# --- Canonicalizacion de nombres de equipo -------------------------------
# BUG REAL encontrado el 2026-08-22 al descargar la temporada 2026-27:
# football-data.co.uk NO usa un nombre estable para un mismo club entre
# temporadas, y en LaLiga incluso DENTRO del mismo archivo. Confirmado con
# datos reales del crudo (data/raw/LALIGA/2627.csv, 7 partidos):
#   - 'Vallecano' (15/08/2026, vs Sevilla) y 'Rayo Vallecano' (20/08/2026,
#     vs Alaves) son el MISMO club, en el MISMO archivo, con 5 dias de
#     diferencia.
#   - 'Ath Madrid' (6 temporadas, 2020-2026) pasa a 'Atl. Madrid' en 2026-27.
#
# Por que esto NO es cosmetico: el modelo Poisson usa 'C(team)' como
# categorica (ver poisson_model_v4.py). Dos strings distintos = DOS
# coeficientes de equipo distintos. Sin esta correccion, el Atletico entrena
# un coeficiente nuevo con 1 partido mientras sus 6 temporadas de historia
# quedan huerfanas bajo el nombre viejo -- y el modelo cotiza un equipo que
# no existe.
#
# Direccion del mapeo: SIEMPRE hacia el nombre con historia (el viejo), no
# hacia el nuevo -- el objetivo es preservar las temporadas ya acumuladas.
#
# Cada entrada aca esta VERIFICADA contra los crudos reales, no supuesta.
# Para lo que todavia no este verificado esta _detect_name_drift() abajo,
# que avisa en cada corrida en vez de dejar que se corrompa en silencio.
TEAM_NAME_CANONICAL = {
    "LALIGA": {
        "Rayo Vallecano": "Vallecano",   # confirmado: ambos en 2627.csv, mismo club
        "Atl. Madrid": "Ath Madrid",     # confirmado: 'Ath Madrid' en 2021-2526, 'Atl. Madrid' desde 2627
    },
    "EPL": {},
    "SERIEA": {},
    "BUNDESLIGA": {},
    "MLS": {},
}

# Umbral de similitud para sospechar que dos nombres son el mismo club.
# 0.80 medido contra las 4 ligas reales (2026-08-22): atrapa los 2 casos
# reales ('Ath Madrid'/'Atl. Madrid' con 0.86, y 'Rayo Vallecano'/
# 'Vallecano' por contencion) y produce 1 falso positivo
# ('Atl. Madrid'/'Real Madrid', 0.82) porque en una temporada recien
# empezada todavia no se enfrentaron.
#
# Se deja en 0.80 a proposito y NO se sube para eliminar ese falso
# positivo: este detector solo AVISA, nunca corrige solo. Un falso
# positivo cuesta una linea de salida que se descarta a mano; un falso
# negativo cuesta un club partido en dos coeficientes y un modelo
# corrompido en silencio. La asimetria manda.
NAME_DRIFT_SIMILARITY = 0.80


def _normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _canonicalize_team_names(df: pd.DataFrame, league_key: str) -> pd.DataFrame:
    """Unifica los nombres de club que la fuente escribe de mas de una forma
    (ver TEAM_NAME_CANONICAL). Reporta cuantas filas toco, para que el
    cambio quede visible en la salida y no sea una transformacion silenciosa."""
    mapping = TEAM_NAME_CANONICAL.get(league_key, {})
    if not mapping:
        return df
    out = df.copy()
    for col in ["HomeTeam", "AwayTeam"]:
        n_touched = out[col].isin(mapping).sum()
        if n_touched:
            for viejo, nuevo in mapping.items():
                afectadas = (out[col] == viejo).sum()
                if afectadas:
                    print(f"  [CANONICAL] {league_key}.{col}: '{viejo}' -> '{nuevo}' "
                          f"({afectadas} filas)")
            out[col] = out[col].replace(mapping)
    return out


def _detect_name_drift(df: pd.DataFrame, league_key: str) -> None:
    """Detector automatico de nombres que probablemente sean el MISMO club
    escrito distinto -- lo que importa a futuro, porque esto va a volver a
    pasar cada vez que la fuente cambie una convencion.

    Heuristica, con la segunda condicion haciendo el trabajo pesado:
      1. Los dos nombres son parecidos: uno contiene al otro, o su
         similitud de secuencia supera NAME_DRIFT_SIMILARITY.
      2. **Nunca jugaron entre si.** Un club no puede enfrentarse a si
         mismo, asi que dos nombres parecidos que SI se enfrentaron son
         clubes distintos con seguridad. Esto es lo que evita marcar
         'Man City'/'Man United' o 'Ath Bilbao'/'Ath Madrid'.

    Solo avisa -- no corrige nada por su cuenta. Lo que reporte hay que
    verificarlo a mano y, si es real, agregarlo a TEAM_NAME_CANONICAL."""
    teams = sorted(set(df["HomeTeam"].dropna()).union(set(df["AwayTeam"].dropna())))
    enfrentamientos = set()
    for h, a in zip(df["HomeTeam"], df["AwayTeam"]):
        enfrentamientos.add(frozenset((h, a)))

    sospechas = []
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            if frozenset((a, b)) in enfrentamientos:
                continue  # se enfrentaron -> clubes distintos, seguro
            na, nb = _normalize_name(a), _normalize_name(b)
            contiene = na in nb or nb in na
            ratio = difflib.SequenceMatcher(None, na, nb).ratio()
            if contiene or ratio >= NAME_DRIFT_SIMILARITY:
                sospechas.append((a, b, ratio, contiene))

    if sospechas:
        print(f"  [AVISO DE DERIVA DE NOMBRES] {league_key}: {len(sospechas)} par(es) de nombres "
              f"parecidos que NUNCA se enfrentaron -- posible mismo club escrito de dos formas. "
              f"Verificar y, si corresponde, agregar a TEAM_NAME_CANONICAL:")
        for a, b, ratio, contiene in sospechas:
            motivo = "uno contiene al otro" if contiene else f"similitud {ratio:.2f}"
            print(f"     '{a}'  <->  '{b}'   ({motivo})")


def _normalize_mls_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra el crudo de MLS a la convención estándar (ver docstring del
    módulo). No se toca el nombre de las columnas de cuotas (PSCH/PSCD/PSCA,
    B365CH/CD/CA, etc.) -- esas ya coinciden con la convención de las otras
    4 ligas; MLS simplemente no tiene las de apertura."""
    missing = [c for c in MLS_RAW_COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"MLS: se esperaban las columnas {list(MLS_RAW_COLUMN_MAP.keys())} en el crudo "
            f"(salida de mls_loader.py) pero faltan {missing}. Columnas presentes: "
            f"{list(df.columns)}. El esquema de MLS pudo haber cambiado en el sitio -- no se "
            f"renombra a ciegas, revisar antes de seguir."
        )
    return df.rename(columns=MLS_RAW_COLUMN_MAP)


def load_raw_league(league_key: str) -> pd.DataFrame:
    """Concatena todos los CSV crudos de una liga en un solo DataFrame."""
    league_dir = RAW_DATA_DIR / league_key
    csv_files = sorted(league_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No hay CSVs crudos en {league_dir}. Corre la ingesta primero.")

    frames = [pd.read_csv(f) for f in csv_files]
    return pd.concat(frames, ignore_index=True)


def implied_prob_no_vig(odds_home: pd.Series, odds_draw: pd.Series, odds_away: pd.Series) -> pd.DataFrame:
    """
    Convierte cuotas decimales en probabilidades implícitas SIN el margen
    de la casa (overround). Formula estándar:
        prob_bruta = 1 / cuota
        prob_justa = prob_bruta / suma(prob_brutas)
    Esto es matemática determinística (no una predicción) — el punto de
    partida estándar en la literatura de Benter/Mack para comparar contra
    el modelo propio.
    """
    raw_home = 1 / odds_home
    raw_draw = 1 / odds_draw
    raw_away = 1 / odds_away
    overround = raw_home + raw_draw + raw_away

    return pd.DataFrame({
        "prob_home": raw_home / overround,
        "prob_draw": raw_draw / overround,
        "prob_away": raw_away / overround,
        "overround": overround,  # >1.0 = margen de la casa; útil para medir "sharpness"
    })


def add_bookmaker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada casa prioritaria configurada en settings, calcula probabilidades
    implícitas de cierre siempre que estén disponibles, y de apertura + CLV
    solo si TAMBIÉN está disponible la apertura (algunas ligas, como MLS,
    únicamente tienen cierre -- ver docstring del módulo). Si a una casa le
    falta incluso el cierre, se omite por completo, sin romper el pipeline.
    """
    out = df.copy()

    for book_name, cols in PRIORITY_BOOKMAKERS.items():
        # cols["open"] es tipo "PSH" -> away de apertura es "PSA"
        open_away_col = cols["open"][:-1] + "A"
        open_cols = [cols["open"], cols["draw"], open_away_col]
        close_cols = [cols["close_home"], cols["close_draw"], cols["close_away"]]

        has_open = all(c in df.columns for c in open_cols)
        has_close = all(c in df.columns for c in close_cols)

        if not has_close:
            print(f"[SKIP] {book_name}: columnas de cierre no disponibles en este dataset, se omite.")
            continue

        close_probs = implied_prob_no_vig(df[cols["close_home"]], df[cols["close_draw"]], df[cols["close_away"]])
        out[f"{book_name}_close_prob_home"] = close_probs["prob_home"]
        out[f"{book_name}_close_prob_draw"] = close_probs["prob_draw"]
        out[f"{book_name}_close_prob_away"] = close_probs["prob_away"]
        out[f"{book_name}_close_overround"] = close_probs["overround"]

        if not has_open:
            print(f"[AVISO] {book_name}: columnas de APERTURA no disponibles en este dataset -- "
                  f"se calculan probabilidades de CIERRE igual, pero NO hay CLV (requiere apertura "
                  f"y cierre juntos). No se inventa un proxy: columnas de apertura/CLV para esta "
                  f"casa quedan sin generar.")
            continue

        open_probs = implied_prob_no_vig(df[cols["open"]], df[cols["draw"]], df[open_away_col])
        out[f"{book_name}_open_prob_home"] = open_probs["prob_home"]
        out[f"{book_name}_open_prob_draw"] = open_probs["prob_draw"]
        out[f"{book_name}_open_prob_away"] = open_probs["prob_away"]
        out[f"{book_name}_open_overround"] = open_probs["overround"]

        # CLV proxy: movimiento de probabilidad justa entre apertura y cierre.
        # Positivo en "home" = el mercado le dio más chance al local al cerrar.
        out[f"{book_name}_clv_home"] = close_probs["prob_home"] - open_probs["prob_home"]
        out[f"{book_name}_clv_draw"] = close_probs["prob_draw"] - open_probs["prob_draw"]
        out[f"{book_name}_clv_away"] = close_probs["prob_away"] - open_probs["prob_away"]

    return out


def clean_league(league_key: str) -> Path:
    """Pipeline completo: cargar crudo -> normalizar (si hace falta) -> validar
    columnas core -> features de cuotas -> guardar."""
    df = load_raw_league(league_key)

    if league_key == "MLS":
        df = _normalize_mls_raw(df)

    missing_core = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_core:
        raise ValueError(f"Faltan columnas core en {league_key}: {missing_core}")

    # Unificar nombres de club ANTES de cualquier otra cosa -- todo lo que
    # viene aguas abajo (features de forma, C(team) del Poisson, Elo) asume
    # que un club = un string. Ver TEAM_NAME_CANONICAL.
    df = _canonicalize_team_names(df, league_key)
    _detect_name_drift(df, league_key)

    n_before = len(df)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    n_bad_dates = int(df["Date"].isna().sum())
    if n_bad_dates > 0:
        pct_bad = n_bad_dates / n_before
        print(f"[AVISO] {league_key}: {n_bad_dates}/{n_before} fechas no se pudieron parsear "
              f"({pct_bad:.1%}) con el formato asumido (dayfirst=True) -- esas filas se van a "
              f"descartar en el siguiente paso. Si este porcentaje es alto (no un puñado de filas "
              f"sueltas), el formato real de fecha de esta liga puede ser distinto -- revisar antes "
              f"de confiar en el resultado.")

    df = df.dropna(subset=["Date", "FTR"])  # partidos sin resultado no sirven para entrenar

    df = add_bookmaker_features(df)

    out_dir = PROCESSED_DATA_DIR / league_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matches_clean.csv"
    df.to_csv(out_path, index=False)

    return out_path


if __name__ == "__main__":
    for league_key in LEAGUES:
        path = clean_league(league_key)
        print(f"[OK] {league_key} limpio -> {path}")

    # MLS deliberadamente NO esta en LEAGUES (ver config/settings.py, Fase 8b)
    # -- se corre aparte, explicito, no como parte del loop silencioso de arriba.
    mls_path = clean_league("MLS")
    print(f"[OK] MLS limpio -> {mls_path}")