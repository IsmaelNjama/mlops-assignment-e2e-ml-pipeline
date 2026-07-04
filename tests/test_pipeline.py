import json
import tempfile
import unittest
from pathlib import Path

from pipeline.experiment import build_run_config, collect_metrics, prepare_run_dir


class PipelineHelpersTest(unittest.TestCase):
    def test_build_run_config_uses_params_and_defaults(self):
        params = {
            "split": "test",
            "subset": "verified",
            "workers": "3",
            "model": "demo/model",
            "task_slice": "0:2",
            "run_id": "demo-run",
            "cost_limit": "1",
        }

        config = build_run_config(params)

        self.assertEqual(config["split"], "test")
        self.assertEqual(config["subset"], "verified")
        self.assertEqual(config["workers"], 3)
        self.assertEqual(config["model"], "demo/model")
        self.assertEqual(config["run_id"], "demo-run")
        self.assertEqual(config["cost_limit"], 1)

    def test_prepare_run_dir_creates_expected_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "demo-run"
            config = build_run_config({
                "split": "test",
                "subset": "verified",
                "workers": "2",
                "model": "demo/model",
                "task_slice": "0:1",
                "run_id": "demo-run",
                "cost_limit": "0",
            })

            prepare_run_dir(config, run_dir)

            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "run-agent").is_dir())
            self.assertTrue((run_dir / "run-eval").is_dir())

    def test_collect_metrics_reads_report_and_summarizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps({
                "task-1": {
                    "resolved": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["a"], "failure": []},
                        "PASS_TO_PASS": {"success": ["b"], "failure": []},
                        "FAIL_TO_FAIL": {"success": [], "failure": []},
                        "PASS_TO_FAIL": {"success": [], "failure": []},
                    },
                },
                "task-2": {
                    "resolved": False,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": [], "failure": ["c"]},
                        "PASS_TO_PASS": {"success": ["d"], "failure": []},
                        "FAIL_TO_FAIL": {"success": [], "failure": []},
                        "PASS_TO_FAIL": {"success": [], "failure": []},
                    },
                },
            }))

            metrics = collect_metrics(report_path)

            self.assertEqual(metrics["tasks_total"], 2)
            self.assertEqual(metrics["tasks_resolved"], 1)
            self.assertEqual(metrics["resolve_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
