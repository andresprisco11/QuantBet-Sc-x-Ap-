"""
Utilidad de consulta sobre el registro de corridas (data/runs/experiment_log.jsonl).

Ejemplos de uso:
    python -m src.tracking.query_runs                     # lista todas las corridas, resumen
    python -m src.tracking.query_runs --model poisson      # filtra por model_name
    python -m src.tracking.query_runs --run-id 20260817_...  # detalle completo de una corrida

Para analisis mas a fondo (comparar corridas, graficar evolucion de metricas
en el tiempo), carga el archivo directamente en un notebook:
    import pandas as pd
    df = pd.read_json("data/runs/experiment_log.jsonl", lines=True)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.tracking.run_logger import LOG_PATH


def load_runs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [c for c in ["run_id", "logged_at", "script", "model_name", "model_version", "git.commit", "git.dirty"] if c in df.columns]
    metric_cols = sorted(c for c in df.columns if c.startswith("metrics."))
    return df[base_cols + metric_cols]


def main():
    parser = argparse.ArgumentParser(description="Consulta el registro de corridas del proyecto.")
    parser.add_argument("--model", default=None, help="Filtra por model_name (ej. poisson)")
    parser.add_argument("--run-id", default=None, help="Muestra el detalle JSON completo de una corrida especifica")
    args = parser.parse_args()

    df = load_runs()
    if df.empty:
        print("Todavia no hay corridas registradas en", LOG_PATH)
        return

    if args.run_id:
        row = df[df["run_id"] == args.run_id]
        if row.empty:
            print(f"No se encontro run_id={args.run_id}")
            return
        print(json.dumps(row.iloc[0].to_dict(), indent=2, ensure_ascii=False, default=str))
        return

    if args.model:
        df = df[df["model_name"] == args.model]

    summary = summarize(df).sort_values("logged_at")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()