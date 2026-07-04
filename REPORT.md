# REPORT — End-to-end ML Pipeline (MLOps Assignment)

## Architecture

The pipeline turns two ad-hoc shell scripts (`mini-swe-bench-batch.sh` and `swe-bench-eval.sh`) into a four-task Airflow DAG with MLflow tracking.

```
prepare_run  →  run_agent  →  run_eval  →  summarize_and_log
```

| Task | What it does |
|---|---|
| `prepare_run` | Reads Airflow params, generates a `run_id` (timestamp if set to `"auto"`), creates `runs/<run-id>/` directory tree, writes `config.json` |
| `run_agent` | Calls `scripts/mini-swe-bench-batch.sh` via subprocess with env vars from config; outputs `preds.json` and per-task trajectory folders to `runs/<run-id>/run-agent/` |
| `run_eval` | Calls `scripts/swe-bench-eval.sh` with `preds.json`; outputs per-task `report.json` files and an `eval.log` to `runs/<run-id>/run-eval/` |
| `summarize_and_log` | Parses `report.json` files to compute `tasks_total`, `tasks_resolved`, `resolve_rate`; writes `metrics.json` and `manifest.json`; logs params + metrics to MLflow |

**Deployment mode**: standalone Airflow (`uv tool run apache-airflow standalone`) with MLflow server run separately.

---

## How to Trigger a Run

### 1. Install dependencies

```bash
uv sync
```

### 2. Start MLflow tracking server (in a separate terminal)

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### 3. Start Airflow standalone (in a separate terminal)

```bash
bash run-airflow-standalone.sh
```

Open http://localhost:8080 (password is `admin` / `admin`).

### 4. Trigger the DAG

In the Airflow UI:

1. Open the `evaluate_agent` DAG.
2. Click **Trigger DAG w/ config**.
3. Optionally customise params (see table below) or leave defaults for a quick 3-task run.
4. Click **Trigger**.

Available params:

| Param | Default | Description |
|---|---|---|
| `split` | `test` | SWE-bench split |
| `subset` | `verified` | SWE-bench subset |
| `workers` | `1` | Parallel evaluation workers |
| `model` | `nebius/moonshotai/Kimi-K2.6` | LLM model ID |
| `task_slice` | `0:3` | Which tasks to run (Python slice syntax) |
| `run_id` | `auto` | Run identifier; `auto` generates a timestamp-based ID |
| `cost_limit` | `0` | Per-run cost cap in USD; `0` = unlimited |

---

## Artifact Layout

Every run produces:

```
runs/
  <run-id>/
    config.json          # all params that produced this run
    run-agent/
      preds.json         # model predictions (input to swebench eval)
      <task-id>/         # per-task trajectory folders from mini-swe-agent
    run-eval/
      eval.log           # combined stdout/stderr from swebench harness
      logs/
        run_evaluation/
          <split>/
            <model-slug>/
              <task-id>/
                report.json       # pass/fail verdict per task
                run_instance.log
                test_output.txt
                patch.diff
    metrics.json         # tasks_total, tasks_resolved, resolve_rate
    manifest.json        # index of all important files in this run
```

`manifest.json` example:
```json
{
  "run_dir": "/path/to/runs/run-20240104-120000",
  "config": "…/config.json",
  "agent_outputs": "…/run-agent",
  "predictions": "…/run-agent/preds.json",
  "eval_outputs": "…/run-eval",
  "eval_log": "…/run-eval/eval.log",
  "metrics": "…/metrics.json",
  "summary": {"tasks_total": 3, "tasks_resolved": 1, "resolve_rate": 0.3333}
}
```

---

## MLflow Tracking

MLflow is used in the `summarize_and_log` task. By default it connects to `http://localhost:5000`. Override with the `MLFLOW_TRACKING_URI` environment variable.

Each run logs:

- **Params**: `split`, `subset`, `workers`, `model`, `task_slice`, `run_id`, `cost_limit`
- **Metrics**: `tasks_total`, `tasks_resolved`, `resolve_rate`
- **Tags**: `artifact_path` (absolute path to `runs/<run-id>/`)

To compare runs, open http://localhost:5000, select the `evaluate_agent` experiment, and use the built-in comparison view.

---

## Rerunning by `run_id`

To rerun from a specific run folder, trigger the DAG with `run_id` set to the existing folder name. The `prepare_run` task will reuse the directory (it calls `mkdir -p`), and you can inspect `config.json` inside that folder for the exact parameters used.

To reproduce a run from scratch on a fresh machine:

1. Copy `runs/<run-id>/config.json` to the new machine.
2. Start Airflow + MLflow.
3. Trigger the DAG with all params taken from `config.json`.

---

## Notes on Remote Storage (S3)

Remote upload is not implemented in this iteration. To add it, extend `summarize_and_log` with:

```python
import subprocess
subprocess.run(
    ["aws", "s3", "sync", str(run_dir), f"s3://your-bucket/runs/{config['run_id']}/"],
    check=True,
)
mlflow.set_tag("s3_artifact_uri", f"s3://your-bucket/runs/{config['run_id']}/")
```

Then log the S3 URI to MLflow instead of (or in addition to) the local path.
