"""
Fase 19b -- ¿Quien mueve primero? La pregunta que nunca hicimos.

### El supuesto que nunca testeamos

Todo el proyecto asumio que **Pinnacle es el que sabe**. El detector medía
edge contra Pinnacle. El CLV medía contra el cierre de Pinnacle. La tesis 3
entera descansaba en que su linea es el valor justo.

Y despues Pinnacle se movio -5.06% sistematicamente en contra de lo que
detectabamos, con t=-3.49, y no supimos explicar por que.

Nunca preguntamos lo obvio: **¿es Pinnacle el que lidera, o solo el que
mejor precio pone?** No son lo mismo. Un libro puede tener el margen mas
fino del mundo y aun asi estar copiando el movimiento de otro.

### Que mide esto

Para cada par de casas (A, B) y cada resultado de cada partido:

    ¿El cambio de precio de A en el instante t predice el cambio de B en
    t+1, mas de lo que el cambio de B en t predice el de A en t+1?

La ASIMETRIA es lo que importa. Si A predice a B pero B no predice a A, A
lidera. Si se predicen mutuamente por igual, comparten proveedor de precios
y no hay informacion en el vinculo -- solo un cable comun.

Se reporta por casa un `liderazgo` = (cuanto predice a los demas) menos
(cuanto es predicha por los demas). Positivo = va adelante.

### Por que esto NO se puede satisfacer por construccion

El movimiento futuro del precio de otra casa no lo controlamos ni entra en
el calculo del propio movimiento. Es la misma clase de metrica que
`movimiento de Pinnacle` en la fase 14 -- por eso aquella pudo dar un
veredicto creible cuando el CLV no podia.

### Tres formas en que esto puede enganar, dichas antes

1. **Feeds compartidos.** Muchas casas chicas compran precios al mismo
   proveedor. Se mueven juntas sin que ninguna sepa nada. Por eso se mide
   asimetria y no correlacion simple, y por eso se reporta tambien la
   correlacion en lag 0: si es altisima, son la misma cosa.
2. **Frecuencia de actualizacion.** Una casa que refresca cada 5 minutos va
   a "predecir" a una que refresca cada hora, sin tener mas informacion.
   Se reporta `pct_cambios` -- si una casa casi no se mueve, su liderazgo
   aparente no vale.
3. **Muestra chica.** Con pocas fotos el ruido domina. Se exige un minimo.

Uso:
    python -m src.discovery.lead_lag
    python -m src.discovery.lead_lag --dia 20260902 --min-fotos 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.tracking.run_logger import RUNS_DIR

DIR = RUNS_DIR / "snapshots"
MIN_FOTOS = 6
MIN_OBS = 30          # observaciones (evento,resultado,par de fotos) por casa
MIN_CAMBIOS = 0.05    # una casa que se mueve en <5% de los pasos no informa


def cargar(dia: str | None) -> pd.DataFrame:
    archivos = sorted(DIR.glob(f"odds_{dia}.csv" if dia else "odds_*.csv"))
    if not archivos:
        print("Sin snapshots. Corre odds_snapshots --capturar varias veces.")
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    d["snap_utc"] = pd.to_datetime(d["snap_utc"], utc=True, errors="coerce")
    d["p"] = 1.0 / pd.to_numeric(d["outcome_price_decimal"], errors="coerce")
    return d.dropna(subset=["p", "snap_utc"])


def construir_cambios(d: pd.DataFrame) -> pd.DataFrame:
    """Serie de cambios por (evento, resultado, casa) entre fotos consecutivas.

    Se usa la probabilidad implicita CRUDA, sin desviguear. Es a proposito:
    el desvig introduce el supuesto de Shin, y aca queremos observar el
    movimiento tal cual lo publica la casa. Ademas el margen de cada casa es
    aproximadamente constante entre dos fotos seguidas, asi que se cancela
    en la diferencia."""
    d = d.sort_values("snap_utc")
    d["clave"] = d["event_id"] + "|" + d["outcome_name"]
    piv = d.pivot_table(index=["clave", "snap_utc"], columns="casa",
                        values="p", aggfunc="first")
    cambios = piv.groupby(level=0).diff()
    return cambios.dropna(how="all")


def analizar(cambios: pd.DataFrame) -> pd.DataFrame:
    casas = [c for c in cambios.columns
             if cambios[c].notna().sum() >= MIN_OBS]
    if len(casas) < 3:
        print("Muy pocas casas con datos suficientes.")
        return pd.DataFrame()

    # adelantar una foto dentro de cada clave: el cambio de B en t+1
    sig = cambios.groupby(level=0).shift(-1)

    filas = []
    for a in casas:
        pred, sido_pred, lag0 = [], [], []
        for b in casas:
            if a == b:
                continue
            # a(t) contra b(t+1) -> a predice a b
            m = pd.concat([cambios[a], sig[b]], axis=1).dropna()
            if len(m) >= MIN_OBS and m.iloc[:, 0].std() > 0 and m.iloc[:, 1].std() > 0:
                pred.append(np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1])
            # b(t) contra a(t+1) -> a es predicha por b
            m2 = pd.concat([cambios[b], sig[a]], axis=1).dropna()
            if len(m2) >= MIN_OBS and m2.iloc[:, 0].std() > 0 and m2.iloc[:, 1].std() > 0:
                sido_pred.append(np.corrcoef(m2.iloc[:, 0], m2.iloc[:, 1])[0, 1])
            # lag 0: si es altisimo comparten proveedor
            m0 = pd.concat([cambios[a], cambios[b]], axis=1).dropna()
            if len(m0) >= MIN_OBS and m0.iloc[:, 0].std() > 0 and m0.iloc[:, 1].std() > 0:
                lag0.append(np.corrcoef(m0.iloc[:, 0], m0.iloc[:, 1])[0, 1])

        if not pred or not sido_pred:
            continue
        serie = cambios[a].dropna()
        filas.append({
            "casa": a,
            "n_obs": int(cambios[a].notna().sum()),
            "pct_cambios": float((serie.abs() > 1e-6).mean()),
            "predice": float(np.mean(pred)),
            "es_predicha": float(np.mean(sido_pred)),
            "liderazgo": float(np.mean(pred) - np.mean(sido_pred)),
            "sincronia_lag0": float(np.mean(lag0)) if lag0 else np.nan,
        })
    return pd.DataFrame(filas).sort_values("liderazgo", ascending=False)


def reportar(t: pd.DataFrame) -> None:
    print(f"\n{'='*94}")
    print("¿QUIEN MUEVE PRIMERO? -- asimetria de prediccion entre casas")
    print(f"{'='*94}")
    print(f"{'casa':<24}{'obs':>7}{'%mueve':>9}{'predice':>10}"
          f"{'es pred.':>10}{'LIDERAZGO':>12}{'sincr.lag0':>12}")
    print("-" * 94)
    for _, r in t.iterrows():
        marca = ""
        if r["pct_cambios"] < MIN_CAMBIOS:
            marca = "  <- casi no se mueve"
        elif r["sincronia_lag0"] > 0.7:
            marca = "  <- feed compartido?"
        print(f"{str(r['casa'])[:23]:<24}{int(r['n_obs']):>7}{r['pct_cambios']:>9.1%}"
              f"{r['predice']:>10.3f}{r['es_predicha']:>10.3f}"
              f"{r['liderazgo']:>+12.3f}{r['sincronia_lag0']:>12.3f}{marca}")
    print("-" * 94)

    if "pinnacle" in set(t["casa"]):
        p = t[t["casa"] == "pinnacle"].iloc[0]
        puesto = list(t["casa"]).index("pinnacle") + 1
        print(f"\nPINNACLE: puesto {puesto} de {len(t)} en liderazgo "
              f"({p['liderazgo']:+.3f})")
        if puesto > len(t) / 2:
            print("   >>> Pinnacle NO lidera. Todo el proyecto asumio que si.")
            print("       Si esto aguanta con mas fotos, explica el movimiento")
            print("       adverso de la fase 14: estabamos midiendo contra un")
            print("       libro que SIGUE, no contra uno que va adelante.")
        else:
            print("   >>> Pinnacle lidera, como se asumia. El supuesto queda validado.")

    print("\n[LECTURA] liderazgo = cuanto predice a los demas menos cuanto es predicha.")
    print("          sincronia lag0 alta (>0.7) = misma fuente de precios, no informacion.")
    print("          %mueve bajo = la casa esta congelada; su liderazgo no vale.")
    print("\n[AVISO] Que una casa lidere NO significa que copiarla de plata: para cuando")
    print("        veas su movimiento, ya movio. Lo que dice es POR DONDE entra la")
    print("        informacion -- que es lo que hay que saber antes de decidir a quien")
    print("        vigilar y contra quien medir el valor justo.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dia", default=None, help="YYYYMMDD; por defecto todos")
    ap.add_argument("--min-fotos", type=int, default=MIN_FOTOS)
    args = ap.parse_args()

    d = cargar(args.dia)
    if d.empty:
        return
    n_fotos = d["snap_utc"].nunique()
    print(f"{len(d)} filas | {n_fotos} fotos | {d['casa'].nunique()} casas | "
          f"{d['event_id'].nunique()} eventos")
    if n_fotos < args.min_fotos:
        print(f"\n[INSUFICIENTE] Con {n_fotos} fotos no se puede medir lead-lag.")
        print(f"               Hacen falta >= {args.min_fotos}. Segui capturando.")
        return

    cambios = construir_cambios(d)
    t = analizar(cambios)
    if not t.empty:
        reportar(t)
        salida = RUNS_DIR / "lead_lag.csv"
        t.to_csv(salida, index=False)
        print(f"\n-> {salida}")


if __name__ == "__main__":
    main()
