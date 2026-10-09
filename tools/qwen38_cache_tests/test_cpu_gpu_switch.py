"""CPU checks for restore-before-wake ordering without touching Docker/GPU."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "flash-next-plan/sglang-gpu-switch-workflow/switch_test.py"
)
spec = importlib.util.spec_from_file_location("gpu_switch", SOURCE)
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)


class SwitchTests(unittest.TestCase):
    def workflow(self, root):
        config = {
            "worktree": str(root),
            "workset": str(root / "workset"),
            "gpu": {},
            "development": {
                "stop_command": ["stop"],
                "start_command": ["start"],
                "health_url": "http://localhost/health_generate",
                "health_timeout_seconds": 1,
            },
            "test": {
                "mode": "batch",
                "batch_command": ["batch"],
                "batch_timeout_seconds": 0,
                "cleanup_command": ["cleanup"],
            },
            "completion": {"command": ["wake"], "required": True, "delay_seconds": 0},
        }
        path = root / "config.json"
        path.write_text(json.dumps(config))
        return switch.Workflow(path)

    def check_flow(
        self,
        batch_code=0,
        restore_failure=False,
        preflight_failure=False,
        wake_failure=False,
    ):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self.workflow(Path(directory))
            events = []

            def command(argv, **kwargs):
                events.append(argv[0])
                if argv[0] == "batch":
                    self.assertIsNone(kwargs["timeout"])
                if argv[0] == "wake":
                    saved = json.loads(
                        (workflow.workset / "artifacts/latest-test.json").read_text()
                    )
                    self.assertIn("development_ready_at", saved)
                    if wake_failure:
                        raise subprocess.TimeoutExpired(argv, 30)
                return subprocess.CompletedProcess(
                    argv, batch_code if argv[0] == "batch" else 0, ""
                )

            def ready(*args):
                events.append("health")
                if restore_failure:
                    raise TimeoutError("not ready")

            def preflight():
                if preflight_failure:
                    raise RuntimeError("preflight failed")
                return {"commit": "cpu-test"}

            with (
                patch.object(workflow, "preflight", side_effect=preflight),
                patch.object(switch, "wait_gpu_free", return_value={}),
                patch.object(switch, "run_command", side_effect=command),
                patch.object(switch, "http_ready", side_effect=ready),
            ):
                code = workflow.run()
            return events, code, workflow.result

    def test_success_restores_and_publishes_before_wake(self):
        events, code, result = self.check_flow()
        self.assertEqual(
            events, ["stop", "batch", "cleanup", "start", "health", "wake"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["completion"]["exit_code"], 0)

    def test_test_failure_also_restores_then_wakes(self):
        events, code, _ = self.check_flow(batch_code=124)
        self.assertEqual(
            events, ["stop", "batch", "cleanup", "start", "health", "wake"]
        )
        self.assertEqual(code, 1)

    def test_unready_service_never_wakes(self):
        events, code, _ = self.check_flow(restore_failure=True)
        self.assertNotIn("wake", events)
        self.assertEqual(code, 1)

    def test_preflight_failure_never_claims_ready_or_wakes(self):
        events, code, _ = self.check_flow(preflight_failure=True)
        self.assertNotIn("wake", events)
        self.assertEqual(code, 1)

    def test_wake_exception_is_saved_after_restore(self):
        events, code, result = self.check_flow(wake_failure=True)
        self.assertEqual(events[-2:], ["health", "wake"])
        self.assertEqual(code, 1)
        self.assertIn("CompletionCommandError", result["errors"][-1])

    def test_missing_required_wake_rejected_before_hardware_check(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self.workflow(Path(directory))
            workflow.config["completion"]["command"] = []
            with self.assertRaisesRegex(ValueError, "completion.command"):
                workflow.preflight()

    def test_missing_wake_executable_rejected_before_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self.workflow(Path(directory))
            workflow.config["completion"]["command"] = ["/nonexistent/wake-codex"]
            with self.assertRaisesRegex(FileNotFoundError, "Completion executable"):
                workflow.preflight()

    def test_recovery_rejects_active_workflow(self):
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            workflow = self.workflow(Path(directory))
            workflow.runtime_dir.mkdir(parents=True)
            with workflow.lock_path.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "running"):
                    workflow.recover()

    def test_recovery_clears_stale_queue_only_after_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self.workflow(Path(directory))
            workflow.queue_dir.mkdir(parents=True)
            pending = workflow.queue_dir / "stale.running"
            pending.write_text("{}")
            with patch.object(workflow, "_recover", return_value=1):
                self.assertEqual(workflow.recover(), 1)
                self.assertTrue(pending.exists())
            with patch.object(workflow, "_recover", return_value=0):
                self.assertEqual(workflow.recover(), 0)
                self.assertFalse(pending.exists())
                self.assertTrue(pending.with_suffix(".recovered").exists())

    def test_dispatch_freezes_job_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = self.workflow(root)
            with (
                patch.object(workflow, "preflight", return_value={"commit": "test"}),
                patch.object(
                    workflow,
                    "daemon_status",
                    return_value={"running": True, "pid": 123},
                ),
            ):
                self.assertEqual(workflow.dispatch(), 0)
            queued = json.loads(next(workflow.queue_dir.glob("*.json")).read_text())
            saved = Path(queued["config"])
            self.assertNotEqual(saved, workflow.config_path)
            workflow.config_path.write_text("{}")
            self.assertEqual(
                json.loads(saved.read_text())["test"]["batch_command"], ["batch"]
            )

    def test_daemon_stop_rejects_pending_job(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self.workflow(Path(directory))
            workflow.queue_dir.mkdir(parents=True)
            (workflow.queue_dir / "pending.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "pending"):
                workflow.daemon_stop()

    def test_recovery_removes_only_saved_owned_container(self):
        for name, valid in [
            ("qwen38-test-" + "a" * 16, True),
            ("normal-sglang", False),
        ]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workflow = self.workflow(root)
                workflow.config["test"]["cleanup_command"] = []
                workflow.config["test"]["result_json"] = "{result_dir}/docker-job.json"
                result_dir = root / "results"
                result_dir.mkdir()
                (result_dir / "docker-job.json").write_text(
                    json.dumps({"container": name})
                )
                workflow.runtime_dir.mkdir(parents=True)
                switch.atomic_json(workflow.state_path, {"result_dir": str(result_dir)})
                with (
                    patch.object(workflow, "start_development"),
                    patch.object(
                        switch,
                        "run_command",
                        return_value=subprocess.CompletedProcess([], 0, ""),
                    ) as command,
                ):
                    code = workflow.recover()
                self.assertEqual(code, 0 if valid else 1)
                if valid:
                    command.assert_called_once_with(
                        ["docker", "rm", "-f", name], timeout=30, check=False
                    )
                else:
                    command.assert_not_called()
                    self.assertEqual(
                        switch.read_json(workflow.state_path)["result_dir"],
                        str(result_dir),
                    )
