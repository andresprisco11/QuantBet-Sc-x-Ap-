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

# --- Ligas activas (banco de pruebas: solo EPL en Fase 1) ---
# Códigos de football-data.co.uk: E0=Premier League, E1=Championship,
# SP1=La Liga, I1=Serie A, D1=Bundesliga, F1=Ligue 1
LEAGUES = {
    "EPL": {
        "code": "E0",
        "country": "England",
        "tier": 1,
    },
}

# --- Temporadas a descargar (formato football-data.co.uk: "2324" = 2023-24) ---
SEASONS = ["2021", "2122", "2223", "2324", "2425"]

# --- Casas de apuestas prioritarias para cálculo de CLV ---
# Pinnacle (PS) es el estándar de referencia por su bajo margen (sharp book)
PRIORITY_BOOKMAKERS = {
    "pinnacle": {"open": "PSH", "draw": "PSD", "close_home": "PSCH", "close_draw": "PSCD", "close_away": "PSCA"},
    "bet365": {"open": "B365H", "draw": "B365D", "close_home": "B365CH", "close_draw": "B365CD", "close_away": "B365CA"},
}
