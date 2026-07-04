import json
import shutil
from pathlib import Path
from typing import Any


def build_run_config(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": params.get("split", "test"),
        "subset": params.get("subset", "verified"),
        "workers": int(params.get("workers", 1)),
        "model": params.get("model", "nebius/moonshotai/Kimi-K2.6"),
        "task_slice": params.get("task_slice", "0:3"),
        "run_id": params.get("run_id", "manual-run"),
        "cost_limit": int(params.get("cost_limit", 0)),
    }


def prepare_run_dir(config: dict[str, Any], run_dir: Path | None = None) -> Path:
    run_dir = run_dir or Path("runs") / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ["run-agent", "run-eval"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return run_dir


def collect_metrics(report_path: Path | str) -> dict[str, Any]:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text()
                        ) if report_path.exists() else {}

    task_ids = list(report.keys())
    resolved = sum(1 for task in report.values()
                   if task.get("resolved") is True)

    return {
        "tasks_total": len(task_ids),
        "tasks_resolved": resolved,
        "resolve_rate": round(resolved / len(task_ids), 4) if task_ids else 0.0,
    }


def write_manifest(run_dir: Path) -> Path:
    manifest = {
        "run_dir": str(run_dir),
        "config": str(run_dir / "config.json"),
        "agent_outputs": str(run_dir / "run-agent"),
        "eval_outputs": str(run_dir / "run-eval"),
        "metrics": str(run_dir / "metrics.json"),
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
