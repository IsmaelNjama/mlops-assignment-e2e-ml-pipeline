"""
evaluate_agent DAG
==================
Airflow pipeline: prepare_run -> run_agent -> run_eval -> summarize_and_log

Params (all configurable from the Airflow UI):
  split       - SWE-bench split (default: "test")
  subset      - SWE-bench subset (default: "verified")
  workers     - parallel eval workers (default: 1)
  model       - LLM model ID (default: "nebius/moonshotai/Kimi-K2.6")
  task_slice  - slice of tasks to run, e.g. "0:3" (default: "0:3")
  run_id      - unique identifier for this run (default: auto-generated timestamp)
  cost_limit  - per-run cost cap in USD, 0 = unlimited (default: 0)

Artifact layout:
  runs/<run-id>/
    config.json
    run-agent/
      preds.json
      <task-id>/   (trajectory folders written by mini-swe-agent)
    run-eval/
      eval.log
      logs/        (swebench harness logs, per-task report.json files)
    metrics.json
    manifest.json
"""

import glob
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Helper functions (importable from pipeline.experiment, also inlined here
# so the DAG is self-contained and easy to read)
# ---------------------------------------------------------------------------


def build_run_config(params: dict[str, Any]) -> dict[str, Any]:
    """Normalise Airflow params into a run config dict."""
    # Auto-generate a run_id if the user left the default placeholder
    run_id = params.get("run_id", "")
    if not run_id or run_id == "auto":
        run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    return {
        "split": params.get("split", "test"),
        "subset": params.get("subset", "verified"),
        "workers": int(params.get("workers", 1)),
        "model": params.get("model", "nebius/moonshotai/Kimi-K2.6"),
        "task_slice": params.get("task_slice", "0:3"),
        "run_id": run_id,
        "cost_limit": int(params.get("cost_limit", 0)),
    }


def prepare_run_dir(config: dict[str, Any], run_dir: Path) -> Path:
    """Create the run directory tree and write config.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("run-agent", "run-eval"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return run_dir


def collect_metrics(eval_dir: Path) -> dict[str, Any]:
    """
    Parse per-task report.json files produced by swebench harness.

    swebench writes one report.json per task under:
      <eval_dir>/logs/run_evaluation/<split>/<model_slug>/<task_id>/report.json

    Falls back to the top-level summary JSON if per-task files are absent.
    """
    per_task_reports = list(eval_dir.rglob("report.json"))

    if per_task_reports:
        resolved = 0
        total = 0
        for rp in per_task_reports:
            try:
                data = json.loads(rp.read_text())
                for _task_id, info in data.items():
                    total += 1
                    if info.get("resolved") is True:
                        resolved += 1
            except Exception:
                pass
        return {
            "tasks_total": total,
            "tasks_resolved": resolved,
            "resolve_rate": round(resolved / total, 4) if total else 0.0,
        }

    # Fallback: look for a summary JSON written next to preds.json
    summary_candidates = list(eval_dir.rglob("*.json"))
    for candidate in summary_candidates:
        try:
            data = json.loads(candidate.read_text())
            if "resolved_instances" in data:
                total = data.get("submitted_instances", 0)
                resolved = data.get("resolved_instances", 0)
                return {
                    "tasks_total": total,
                    "tasks_resolved": resolved,
                    "resolve_rate": round(resolved / total, 4) if total else 0.0,
                }
        except Exception:
            pass

    return {"tasks_total": 0, "tasks_resolved": 0, "resolve_rate": 0.0}


def write_manifest(run_dir: Path, metrics: dict[str, Any]) -> Path:
    """Write manifest.json summarising where everything lives."""
    manifest = {
        "run_dir": str(run_dir),
        "config": str(run_dir / "config.json"),
        "agent_outputs": str(run_dir / "run-agent"),
        "predictions": str(run_dir / "run-agent" / "preds.json"),
        "eval_outputs": str(run_dir / "run-eval"),
        "eval_log": str(run_dir / "run-eval" / "eval.log"),
        "metrics": str(run_dir / "metrics.json"),
        "summary": metrics,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _prepare_run(**context):
    params = context["params"]
    config = build_run_config(params)
    run_dir = prepare_run_dir(config, RUNS_DIR / config["run_id"])

    context["ti"].xcom_push(key="run_dir", value=str(run_dir))
    context["ti"].xcom_push(key="config", value=json.dumps(config))
    print(f"[prepare_run] run_dir={run_dir}")
    return str(run_dir)


def _run_agent(**context):
    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="prepare_run", key="run_dir"))
    config = json.loads(ti.xcom_pull(task_ids="prepare_run", key="config"))

    agent_dir = run_dir / "run-agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "SUBSET": config["subset"],
        "SPLIT": config["split"],
        "MODEL": config["model"],
        "TASK_SLICE": config["task_slice"],
        "WORKERS": str(config["workers"]),
        "COST_LIMIT": str(config["cost_limit"]),
        "OUTPUT_DIR": str(agent_dir),
        "MSWEA_COST_TRACKING": "ignore_errors",
    })

    print(f"[run_agent] running mini-swe-bench-batch.sh -> {agent_dir}")
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "mini-swe-bench-batch.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    ti.xcom_push(key="run_dir", value=str(run_dir))
    return str(run_dir)


def _run_eval(**context):
    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="run_agent", key="run_dir"))
    config = json.loads(ti.xcom_pull(task_ids="prepare_run", key="config"))

    eval_dir = run_dir / "run-eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # mini-swe-agent writes preds.json directly into OUTPUT_DIR
    preds_path = run_dir / "run-agent" / "preds.json"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"preds.json not found at {preds_path}. "
            "Check that run_agent completed successfully."
        )

    env = os.environ.copy()
    env.update({
        "PREDICTIONS_PATH": str(preds_path),
        "MAX_WORKERS": str(config["workers"]),
        "RUN_ID": config["run_id"],
        "OUTPUT_DIR": str(eval_dir),
    })

    print(f"[run_eval] running swe-bench-eval.sh -> {eval_dir}")
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "swe-bench-eval.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    ti.xcom_push(key="run_dir", value=str(run_dir))
    return str(run_dir)


def _summarize_and_log(**context):
    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="run_eval", key="run_dir"))
    config = json.loads(ti.xcom_pull(task_ids="prepare_run", key="config"))

    eval_dir = run_dir / "run-eval"
    metrics = collect_metrics(eval_dir)

    # Write metrics.json
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[summarize_and_log] metrics={metrics}")

    # Write manifest.json
    write_manifest(run_dir, metrics)

    # Log to MLflow
    try:
        import mlflow  # noqa: PLC0415

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("evaluate_agent")

        with mlflow.start_run(run_name=config["run_id"]):
            # Log all config values as params
            mlflow.log_params({
                "split": config["split"],
                "subset": config["subset"],
                "workers": config["workers"],
                "model": config["model"],
                "task_slice": config["task_slice"],
                "run_id": config["run_id"],
                "cost_limit": config["cost_limit"],
            })
            # Log metrics
            mlflow.log_metrics({
                "tasks_total": float(metrics["tasks_total"]),
                "tasks_resolved": float(metrics["tasks_resolved"]),
                "resolve_rate": float(metrics["resolve_rate"]),
            })
            # Log artifact path (local; swap for mlflow.log_artifacts for full upload)
            mlflow.set_tag("artifact_path", str(run_dir))
            mlflow.set_tag("run_dir", str(run_dir))

        print(f"[summarize_and_log] MLflow run logged to {mlflow_uri}")
    except ImportError:
        print(
            "[summarize_and_log] WARNING: mlflow not installed, skipping MLflow logging. "
            "Add 'mlflow' to pyproject.toml dependencies and run `uv sync`."
        )
    except Exception as exc:
        print(f"[summarize_and_log] WARNING: MLflow logging failed: {exc}")

    return str(run_dir)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["swe-bench", "mini-swe-agent"],
    params={
        "split": "test",
        "subset": "verified",
        "workers": 1,
        "model": "nebius/moonshotai/Kimi-K2.6",
        "task_slice": "0:3",
        "run_id": "auto",      # "auto" triggers timestamp-based ID generation
        "cost_limit": 0,
    },
) as dag:
    prepare_run = PythonOperator(
        task_id="prepare_run",
        python_callable=_prepare_run,
    )
    run_agent = PythonOperator(
        task_id="run_agent",
        python_callable=_run_agent,
    )
    run_eval = PythonOperator(
        task_id="run_eval",
        python_callable=_run_eval,
    )
    summarize_and_log = PythonOperator(
        task_id="summarize_and_log",
        python_callable=_summarize_and_log,
    )

    prepare_run >> run_agent >> run_eval >> summarize_and_log
