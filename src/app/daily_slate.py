"""
Fase 22 -- Selección del día: un instrumento de medición, no un pronóstico.

### Lo que hay que entender antes de usar esto

El proyecto midió su propio edge y dio **cero**: CLV +0.06% con t=-0.02 sobre
105 apuestas independientes. Un modulo que elige apuestas sobre un edge
medido en cero **no puede tener valor esperado positivo**. Este archivo no
existe para ganar plata: existe para MEDIR si algun dia hay algo que ganar.

La diferencia es toda: un pronosticador dice "juga esto". Un instrumento dice
"tome este precio a esta hora, veamos si el mercado se movio hacia mi". Lo
segundo produce el dato que hace falta; lo primero produce solo opiniones.

### Sobre que se selecciona

Sobre la unica magnitud que este proyecto pudo medir y que existe de verdad:
el **premio de la mejor casa sobre la mediana**. No es edge -- es lo que
ganas por buscar precio en vez de tomar el primero que aparece. Se mide sin
modelo, no depende de predecir nada, y ya sabemos que su valor es real
aunque no se traduzca (todavia) en CLV positivo.

Filtros: solo casas accesibles desde Nueva York, minimo de casas cotizando
(el premio depende del numero de sorteos, no solo del desacuerdo -- fase 17),
y se descarta lo que arranca en menos de una hora.

### Por que las combinadas van marcadas como PEOR opcion

`parlay_engine` midio rho ~ 0 sobre 6 temporadas: los partidos son
independientes. Con independencia, la tasa de crecimiento con Kelly es

    G ~ e² / (2·(cuota − 1))

El edge de una combinada crece LINEAL con las piernas y la cuota crece
EXPONENCIAL. La exponencial gana siempre. Por eso el modulo calcula, para
cada combinada que arma, cuanto rinde comparada con jugar esas mismas
piernas sueltas -- y ese numero siempre sale menor que 1.

Se generan igual porque el usuario las pidio y porque medirlas tambien es
medir. Pero cada boleto lleva su propio veredicto encima.

Uso:
    python -m src.app.daily_slate --generar
    python -m src.app.daily_slate --generar --piernas 3 --min-premio 0.04
    python -m src.app.daily_slate --reporte
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(RAIZ))
from src.tracking.run_logger import RUNS_DIR

APP = RAIZ / "app"
LOG = RUNS_DIR / "slate_log.csv"

# Casas usables desde Nueva York. Sin este filtro la seleccion se llena de
# oportunidades europeas que no se pueden tomar -- pasó en la fase 13.
ACCESIBLES = {
    "fanduel", "draftkings", "betmgm", "williamhill_us", "betrivers",
    "fanatics", "espnbet", "ballybet", "resortsworld", "betonlineag",
    "lowvig", "betus", "bovada", "mybookieag", "betanysports", "gtbets",
    "everygame", "BetOnline", "Flutter", "WilliamHill",
}
MIN_CASAS = 12          # el premio depende del numero de sorteos (fase 17)
MIN_HORAS = 1.0
KELLY_FRAC = 0.10

COLUMNAS = ["slate_id", "generado_utc", "tipo", "piernas", "descripcion",
            "cuota", "prob_mercado", "premio_medio", "stake_frac",
            "eficiencia_vs_sueltas", "detalle", "resuelto", "gano", "clv"]


def candidatas(data: dict, min_premio: float) -> pd.DataFrame:
    ahora = datetime.now(timezone.utc)
    filas = []
    for p in data["partidos"]:
        try:
            faltan = (pd.Timestamp(p["ts"]).tz_convert("UTC") - ahora).total_seconds() / 3600
        except Exception:
            continue
        if faltan < MIN_HORAS:
            continue
        for o in p.get("outcomes", []):
            if (o.get("premio") is None or o.get("mejor_cuota") is None
                    or o.get("mkt") is None):
                continue
            if o["premio"] < min_premio:
                continue
            if (o.get("n_casas") or 0) < MIN_CASAS:
                continue
            if o.get("mejor_casa") not in ACCESIBLES:
                continue
            filas.append({
                "partido": f"{p['home']['name']} vs {p['away']['name']}",
                "liga": p["league"], "arranca": p["date"], "horas": faltan,
                "via": o["name"], "cuota": float(o["mejor_cuota"]),
                "casa": o["mejor_casa"], "p": float(o["mkt"]),
                "premio": float(o["premio"]),
            })
    d = pd.DataFrame(filas)
    if d.empty:
        return d
    # una sola via por partido: dos vias del mismo partido son excluyentes,
    # combinarlas daria una combinada imposible de ganar
    d = d.sort_values("premio", ascending=False).drop_duplicates("partido")
    return d.reset_index(drop=True)


def kelly(p: float, cuota: float) -> float:
    e = p * cuota - 1.0
    if e <= 0 or cuota <= 1:
        return 0.0
    return max(0.0, min(KELLY_FRAC * e / (cuota - 1.0), 0.05))


def crecimiento(p: float, cuota: float) -> float:
    """G ~ e²/(2(cuota-1)), la aproximacion de Kelly para edge chico."""
    e = p * cuota - 1.0
    return (e * e) / (2.0 * (cuota - 1.0)) if cuota > 1 else 0.0


def armar_combinada(sel: pd.DataFrame) -> dict:
    cuota = float(np.prod(sel["cuota"]))
    p = float(np.prod(sel["p"]))          # rho ~ 0 medido: se multiplican
    g_comb = crecimiento(p, cuota)
    g_sueltas = sum(crecimiento(r.p, r.cuota) for r in sel.itertuples())
    return {
        "cuota": round(cuota, 2), "p": round(p, 4),
        "premio": round(float(sel["premio"].mean()), 4),
        "stake": round(kelly(p, cuota), 5),
        # < 1 significa que jugarlas sueltas rinde mas
        "eficiencia": round(g_comb / g_sueltas, 3) if g_sueltas > 0 else 0.0,
        "detalle": " + ".join(f"{r.via} @{r.cuota} ({r.casa})" for r in sel.itertuples()),
        "partidos": " | ".join(sel["partido"]),
    }


def generar(min_premio: float, max_piernas: int, n_simples: int) -> None:
    js = APP / "data.js"
    txt = js.read_text(encoding="utf-8")
    data = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    d = candidatas(data, min_premio)
    if d.empty:
        print(f"Sin candidatas: ninguna con premio >= {min_premio:.0%} en casa "
              f"accesible con >= {MIN_CASAS} casas cotizando.")
        return

    ahora = datetime.now(timezone.utc)
    sid = ahora.strftime("%Y%m%d%H%M")
    filas = []

    print(f"{len(d)} candidatas (premio >= {min_premio:.0%}, casa accesible, "
          f"arranque > {MIN_HORAS:.0f}h)\n")
    print("=" * 92)
    print("SIMPLES  --  la estructura optima bajo independencia")
    print("=" * 92)
    print(f"{'apuesta':<44}{'cuota':>7}{'mkt':>7}{'premio':>8}{'stake':>8}{'casa':>14}")
    print("-" * 92)
    for r in d.head(n_simples).itertuples():
        st = kelly(r.p, r.cuota)
        print(f"{(r.via + ' — ' + r.partido)[:43]:<44}{r.cuota:>7.2f}{r.p:>7.0%}"
              f"{r.premio:>8.1%}{st:>8.2%}{r.casa[:13]:>14}")
        filas.append({
            "slate_id": sid, "generado_utc": ahora.isoformat(), "tipo": "simple",
            "piernas": 1, "descripcion": f"{r.via} — {r.partido}",
            "cuota": r.cuota, "prob_mercado": round(r.p, 4),
            "premio_medio": round(r.premio, 4), "stake_frac": round(st, 5),
            "eficiencia_vs_sueltas": 1.0,
            "detalle": f"{r.via} @{r.cuota} ({r.casa})",
            "resuelto": 0, "gano": "", "clv": "",
        })

    for k in range(2, max_piernas + 1):
        if len(d) < k:
            break
        sel = d.head(k * 3).sample(n=k, random_state=int(sid[-4:]) + k) \
            if len(d) >= k * 3 else d.head(k)
        c = armar_combinada(sel)
        print(f"\n{'=' * 92}")
        print(f"COMBINADA DE {k}  --  eficiencia {c['eficiencia']:.2f} vs jugarlas sueltas")
        print("=" * 92)
        print(f"   {c['partidos']}")
        print(f"   {c['detalle']}")
        print(f"   cuota {c['cuota']}  ·  prob mercado {c['p']:.1%}  ·  "
              f"stake {c['stake']:.2%}")
        if c["eficiencia"] < 1:
            print(f"   >>> Jugar estas {k} sueltas rinde {1/c['eficiencia']:.1f}x mas "
                  f"que combinarlas.")
        filas.append({
            "slate_id": sid, "generado_utc": ahora.isoformat(), "tipo": "combinada",
            "piernas": k, "descripcion": c["partidos"], "cuota": c["cuota"],
            "prob_mercado": c["p"], "premio_medio": c["premio"],
            "stake_frac": c["stake"], "eficiencia_vs_sueltas": c["eficiencia"],
            "detalle": c["detalle"], "resuelto": 0, "gano": "", "clv": "",
        })

    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLUMNAS)
    log = pd.concat([log, pd.DataFrame(filas)], ignore_index=True)[COLUMNAS]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG, index=False)

    ev = sum(f["stake_frac"] * (f["prob_mercado"] * f["cuota"] - 1) for f in filas)
    print(f"\n{'=' * 92}")
    print(f"{len(filas)} apuestas registradas en {LOG.name} (id {sid}).")
    print(f"\nValor esperado del boleto completo segun el propio mercado: {ev:+.3%} "
          f"de la banca.")
    print("Es negativo por construccion: la probabilidad usada es la del mercado,")
    print("y el mercado cobra margen. Para que fuera positivo haria falta creer que")
    print("la probabilidad real es MEJOR que la del mercado -- y eso es exactamente")
    print("lo que medimos y dio cero (CLV +0.06%, t=-0.02, n=105).")
    print("\nEsto NO es un pronostico. Es un registro de precios tomados a una hora,")
    print("para poder medir despues si el mercado se movio hacia nosotros.")


def reporte() -> None:
    if not LOG.exists():
        print("Sin selecciones registradas.")
        return
    d = pd.read_csv(LOG)
    print(f"{len(d)} apuestas en {d['slate_id'].nunique()} selecciones\n")
    g = d.groupby("tipo").agg(n=("cuota", "size"), cuota=("cuota", "mean"),
                              premio=("premio_medio", "mean"),
                              efic=("eficiencia_vs_sueltas", "mean"))
    print(g.to_string())
    res = d[d["resuelto"] == 1]
    if len(res) < 20:
        print(f"\n[MUESTRA CHICA] {len(res)} resueltas. Sin veredicto posible.")
        return
    print(f"\nresueltas: {len(res)} | aciertos: {res['gano'].astype(float).mean():.1%}")
    if res["clv"].notna().any():
        c = pd.to_numeric(res["clv"], errors="coerce").dropna()
        t = c.mean() / (c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else np.nan
        print(f"CLV medio: {c.mean():+.2%}  (n={len(c)}, t={t:+.2f})")
        print("\nEsa es la unica linea que importa. Si algun dia da positivo con")
        print("|t| > 2 sostenido, ahi si hay algo. Hasta entonces, es medicion.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generar", action="store_true")
    ap.add_argument("--reporte", action="store_true")
    ap.add_argument("--min-premio", type=float, default=0.03)
    ap.add_argument("--piernas", type=int, default=3)
    ap.add_argument("--simples", type=int, default=6)
    args = ap.parse_args()
    if args.reporte:
        reporte()
    elif args.generar:
        generar(args.min_premio, max(2, args.piernas), args.simples)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
