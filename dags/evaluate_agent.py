from pipeline.experiment import build_run_config, collect_metrics, prepare_run_dir, write_manifest
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUNS_DIR = PROJECT_ROOT / "runs"


def _prepare_run(**context):
    params = context["params"]
    config = build_run_config(params)
    run_dir = prepare_run_dir(config, RUNS_DIR / config["run_id"])

    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    context["ti"].xcom_push(key="run_dir", value=str(run_dir))
    context["ti"].xcom_push(key="config", value=json.dumps(config))
    return str(run_dir)


def _run_agent(**context):
    run_dir = Path(context["ti"].xcom_pull(
        task_ids="prepare_run", key="run_dir"))
    config = json.loads(context["ti"].xcom_pull(
        task_ids="prepare_run", key="config"))

    agent_dir = run_dir / "run-agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "SUBSET": config["subset"],
            "SPLIT": config["split"],
            "MODEL": config["model"],
            "TASK_SLICE": config["task_slice"],
            "WORKERS": str(config["workers"]),
            "COST_LIMIT": str(config["cost_limit"]),
            "OUTPUT_DIR": str(agent_dir),
        }
    )

    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "mini-swe-bench-batch.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    return str(run_dir)


def _run_eval(**context):
    run_dir = Path(context["ti"].xcom_pull(
        task_ids="run_agent", key="return_value"))
    config = json.loads(context["ti"].xcom_pull(
        task_ids="prepare_run", key="config"))

    eval_dir = run_dir / "run-eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    preds_path = run_dir / "run-agent" / "preds.json"
    if not preds_path.exists():
        preds_path = run_dir / "run-agent" / "trajectories" / "preds.json"

    env = os.environ.copy()
    env.update(
        {
            "PREDICTIONS_PATH": str(preds_path),
            "MAX_WORKERS": str(config["workers"]),
            "RUN_ID": config["run_id"],
            "OUTPUT_DIR": str(eval_dir),
        }
    )

    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "swe-bench-eval.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    return str(run_dir)


def _summarize_and_log(**context):
    run_dir = Path(context["ti"].xcom_pull(
        task_ids="run_eval", key="return_value"))
    report_path = run_dir / "run-eval" / "logs" / \
        "run_evaluation" / config["run_id"] / "report.json"
    metrics = collect_metrics(report_path)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    write_manifest(run_dir)

    return str(run_dir)


with DAG(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "split": "test",
        "subset": "verified",
        "workers": 1,
        "model": "nebius/moonshotai/Kimi-K2.6",
        "task_slice": "0:3",
        "run_id": "demo-run",
        "cost_limit": 0,
    },
) as dag:
    prepare_run = PythonOperator(
        task_id="prepare_run", python_callable=_prepare_run)
    run_agent = PythonOperator(task_id="run_agent", python_callable=_run_agent)
    run_eval = PythonOperator(task_id="run_eval", python_callable=_run_eval)
    summarize_and_log = PythonOperator(
        task_id="summarize_and_log", python_callable=_summarize_and_log)

    prepare_run >> run_agent >> run_eval >> summarize_and_log
