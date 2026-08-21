"""
Registro de gastos reales del proyecto (infraestructura de datos, NO
bankroll de apuestas). Cada entrada es un gasto REAL ya confirmado por el
usuario ("ya pague") -- no una cotizacion ni una intencion de compra.

Uso: referencia rapida de a que se debe cada suscripcion recurrente y en
que estado quedo la verificacion tecnica de lo que promete cada una --
mismo criterio de honestidad que el resto del proyecto, incluye las dudas
sin resolver, no solo lo confirmado.
"""

EXPENSES = [
    {
        "date": "2026-08-20",
        "vendor": "TheStatsAPI",
        "amount_usd": 50.0,
        "frequency": "monthly",
        "reason": "xG + cuotas (incl. Pinnacle) para futbol -- para llegar a la calidad de "
                  "modelo necesaria antes de apostar plata real, decision de uso delegada al "
                  "CTO por el usuario ('para llegar a lo que queremos antes de apostar plata').",
        "verification": "CONFIRMADO con probe real contra la API (no marketing): xG real "
                         "100%/cerca en temporadas 22/23-25/26 para EPL/LaLiga/SerieA/Bundesliga "
                         "(antes de eso, 0% confirmado, no 'no cargado'). Cuotas (incluye "
                         "Pinnacle) van bastante mas atras -- 18/19 en EPL, 20/21 en las otras 3. "
                         "Trial de 7 dias sin costo hoy, cobra $50 automatico el 2026-08-27 si no "
                         "se cancela antes -- alarma programada para 2026-08-25 para revisar.",
    },
    {
        "date": "2026-08-21",
        "vendor": "The Odds API",
        "amount_usd": 30.0,
        "frequency": "monthly",
        "reason": "Cuotas de NBA (historico desde ~2020 + actual) para destrabar el 4to deporte "
                  "del mandato original -- tambien candidata a resolver el hueco de cuotas en "
                  "vivo para NFL/otros deportes, riesgo operativo anotado en el roadmap desde "
                  "antes de esta compra. Sin prueba gratis, pagado directo.",
        "verification": "CONFIRMADO con probe real (2026-08-21) contra /v4/sports/basketball_nba/"
                         "odds, regions=us,us2,eu,uk,au: 47 bookmakers reales devueltos (DraftKings, "
                         "FanDuel, BetMGM, Fanatics, ESPN Bet, Bovada, y otros retail/regionales), "
                         "**Pinnacle NO esta entre ellos, en ninguna region probada**. Confirma la "
                         "evaluacion anterior del roadmap (correcta), NO el WebFetch de la pagina de "
                         "marketing hecho el dia de la compra (equivocado -- error del CTO, "
                         "reconocido). CONCLUSION: este gasto SI sirve para el objetivo original de "
                         "destrabar NBA (cuotas reales, historicas y en vivo, de las casas retail "
                         "donde el usuario apostaria de verdad) pero NO resuelve la metodologia de "
                         "CLV-vs-libro-sharp del proyecto -- ese problema sigue abierto, para NBA y "
                         "para NFL, sin solucion todavia. No se recomienda cancelar (el valor de "
                         "cuotas reales para NBA es real), pero el pitch de 'resuelve tambien el "
                         "problema de cuotas en vivo con CLV' que motivo parte de la compra fue "
                         "incorrecto.",
    },
]


def total_monthly_usd() -> float:
    return sum(e["amount_usd"] for e in EXPENSES if e["frequency"] == "monthly")


if __name__ == "__main__":
    for e in EXPENSES:
        print(f"{e['date']} -- {e['vendor']}: ${e['amount_usd']:.2f}/{e['frequency']}")
        print(f"  Motivo: {e['reason']}")
        print(f"  Verificacion: {e['verification']}\n")
    print(f"Total mensual recurrente: ${total_monthly_usd():.2f}")
