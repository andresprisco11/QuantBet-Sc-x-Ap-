"""
Configuración central del proyecto.
Cualquier ruta, liga o constante que se reutilice entre módulos vive aquí —
NUNCA hardcodeada dentro de la lógica de ingesta/modelado.
"""

from pathlib import Path

# --- Rutas base ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# --- Ligas activas ---
# Códigos de football-data.co.uk: E0=Premier League, E1=Championship,
# SP1=La Liga, I1=Serie A, D1=Bundesliga, F1=Ligue 1
#
# Fase 8 (2026-08-18): agregadas La Liga, Serie A y Bundesliga. Confirmado
# con datos reales (headers de CSV descargados de football-data.co.uk) que
# las tres comparten exactamente el mismo esquema de columnas que EPL
# (HS/AS/HST/AST/HC/AC + cuotas Pinnacle apertura/cierre completas), asi que
# clean_data.py y football_data_loader.py no necesitaron ningun cambio de
# logica -- ambos ya iteraban sobre este diccionario. Unica diferencia real:
# ninguna de las tres tiene columna 'Referee' (EPL si) -- no afecta nada,
# no se usa en ningun feature actual.
#
# MLS quedo deliberadamente afuera: football-data.co.uk la sirve en un
# formato completamente distinto (archivo unico 'new/USA.csv', sin tiros al
# arco/corners, sin cuota de apertura Pinnacle, columnas con otros nombres) --
# no es una extension gratuita de este pipeline. Ver roadmap, Fase 8b.
LEAGUES = {
    "EPL": {
        "code": "E0",
        "country": "England",
        "tier": 1,
    },
    "LALIGA": {
        "code": "SP1",
        "country": "Spain",
        "tier": 1,
    },
    "SERIEA": {
        "code": "I1",
        "country": "Italy",
        "tier": 1,
    },
    "BUNDESLIGA": {
        "code": "D1",
        "country": "Germany",
        "tier": 1,
    },
}

# --- Temporadas a descargar (formato football-data.co.uk: "2324" = 2023-24) ---
SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526", "2627"]

# --- Casas de apuestas prioritarias para cálculo de CLV ---
# Pinnacle (PS) es el estándar de referencia por su bajo margen (sharp book)
PRIORITY_BOOKMAKERS = {
    "pinnacle": {"open": "PSH", "draw": "PSD", "close_home": "PSCH", "close_draw": "PSCD", "close_away": "PSCA"},
    "bet365": {"open": "B365H", "draw": "B365D", "close_home": "B365CH", "close_draw": "B365CD", "close_away": "B365CA"},
}