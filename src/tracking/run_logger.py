"""
Sistema de trazabilidad de corridas ("guarda TODO").

Cada corrida importante (entrenamiento, backtest, tuneo de hiperparametros)
queda registrada con todo lo necesario para reconstruirla exactamente
despues: codigo, datos, configuracion y resultado, todos amarrados por un
mismo run_id.

Motivacion: sin esto, "que sabia el sistema el 14 de septiembre de 2026"
es una pregunta que solo se puede responder con memoria humana de lo que
paso, no con evidencia reconstruible. Eso hace el proyecto auditable de
palabra pero no de verdad.

Uso: cada script de modelado llama a log_run() una vez al final de su
ejecucion -- no hace falta llamarlo a mano, los scripts ya lo hacen.

Diseño deliberadamente simple (JSON Lines, append-only, sin dependencias
nuevas): un archivo de texto plano versionable en git, legible por humanos,
cargable en pandas con pd.read_json(path, lines=True). No es un sistema de
tracking de nivel MLflow -- es lo minimo que hace el proyecto auditable
desde hoy. Si el proyecto escala a necesitar mas (dashboards, comparacion
visual de corridas, etc.), esto se puede migrar despues sin perder el
historial ya acumulado.
"""
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BASE_DIR

RUNS_DIR = BASE_DIR / "data" / "runs"
LOG_PATH = RUNS_DIR / "experiment_log.jsonl"


def _git_info() -> dict:
    """
    Commit activo del repositorio y si hay cambios sin commitear.
    Esto ultimo importa: si dirty=True, el run_id NO queda 100% amarrado
    a un estado de codigo reproducible -- alguien podria modificar un
    archivo despues sin que quede registro de que ese cambio existio.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=BASE_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(status)
        return {
            "commit": commit,
            "dirty": dirty,
            "dirty_files": status.splitlines() if dirty else [],
        }
    except Exception as e:
        return {"commit": None, "dirty": None, "error": f"no se pudo leer git: {e}"}


def _file_snapshot(path: Path) -> dict:
    """Hash SHA-256 + metadata de un archivo de datos, para detectar cambios silenciosos."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    content = path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    try:
        rel_path = str(path.relative_to(BASE_DIR))
    except ValueError:
        rel_path = str(path)
    return {
        "path": rel_path,
        "exists": True,
        "sha256": sha256,
        "size_bytes": len(content),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def log_run(
    script: str,
    model_name: str,
    model_version: str,
    data_paths: list,
    features: str,
    hyperparameters: dict,
    metrics: dict,
    predictions_path=None,
    notes: str = "",
) -> str:
    """
    Registra una corrida completa. Devuelve el run_id generado.

    script: nombre del archivo que genero la corrida (ej. "backtest_v2.py")
    model_name / model_version: ej. "poisson", "v2"
    data_paths: lista de rutas (str o Path) a los archivos de datos usados -- se hashean
    features: string de la formula/features usadas (ej. "goals ~ is_home + C(team) + C(opponent)")
    hyperparameters: dict con los parametros de ESTA corrida especifica (ej. {"half_life_days": 200})
    metrics: dict con TODAS las metricas relevantes de esta corrida (Brier, n_partidos, etc.)
    predictions_path: ruta al CSV con predicciones detalladas por partido (opcional pero recomendado --
        ahi quedan las cuotas de cierre, resultado real y probabilidades partido por partido)
    notes: texto libre para contexto humano (ej. "resultado negativo del tuneo de half-life")
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    try:
        pred_rel = str(Path(predictions_path).relative_to(BASE_DIR)) if predictions_path else None
    except ValueError:
        pred_rel = str(predictions_path) if predictions_path else None

    record = {
        "run_id": run_id,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "script": script,
        "model_name": model_name,
        "model_version": model_version,
        "git": _git_info(),
        "data_snapshot": [_file_snapshot(p) for p in data_paths],
        "features": features,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "predictions_path": pred_rel,
        "notes": notes,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    git_warning = ""
    if record["git"].get("dirty"):
        git_warning = " [AVISO: hay cambios sin commitear -- este run_id no queda 100% amarrado a un commit reproducible]"
    print(f"[TRACKING] Corrida registrada -> run_id={run_id}{git_warning}")

    return run_id


if __name__ == "__main__":
    # Prueba rapida: confirma que el logger escribe correctamente.
    run_id = log_run(
        script="run_logger.py (prueba manual)",
        model_name="demo",
        model_version="v0",
        data_paths=[],
        features="ninguna -- solo prueba",
        hyperparameters={},
        metrics={"prueba": True},
        notes="Corrida de prueba para verificar que el logger funciona antes de retrofitear los scripts reales.",
    )
    print(f"Prueba completada. Revisa {LOG_PATH}")