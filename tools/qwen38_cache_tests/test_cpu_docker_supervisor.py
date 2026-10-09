import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ._load_source import REPO_ROOT, load_source

supervisor = load_source(
    "qwen38_supervisor_under_test", "tools/qwen38_cache_tests/docker_supervisor.py"
)


class TestDockerSupervisor(unittest.TestCase):
    def _run(self, slow, cleanup_failure=False, interrupt=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "docker"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys,time\n"
                "from pathlib import Path\n"
                "p=Path(os.environ['FAKE_DOCKER_LOG'])\n"
                "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if sys.argv[1]=='run':\n"
                " print('fixture output',flush=True)\n"
                " if os.environ['FAKE_DOCKER_SLOW']=='1': time.sleep(30)\n"
                "if sys.argv[1]=='rm' and os.environ['FAKE_CLEANUP_FAILURE']=='1':\n"
                " print('daemon cleanup failure',file=sys.stderr); sys.exit(1)\n"
            )
            fake.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(root) + os.pathsep + os.environ["PATH"],
                "FAKE_DOCKER_LOG": str(root / "calls.jsonl"),
                "FAKE_DOCKER_SLOW": str(int(slow)),
                "FAKE_CLEANUP_FAILURE": str(int(cleanup_failure)),
            }
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/qwen38_cache_tests/docker_supervisor.py"),
                "--timeout",
                "1" if slow and not interrupt else "10",
                "--results",
                str(root / "results"),
                "--",
                "docker",
                "run",
                "--rm",
                "fixture-image",
            ]
            if interrupt:
                import signal
                import time

                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    deadline = time.monotonic() + 5
                    while not (root / "calls.jsonl").exists():
                        if time.monotonic() > deadline:
                            self.fail("fixture did not start")
                        time.sleep(0.02)
                    process.send_signal(signal.SIGTERM)
                    stdout, stderr = process.communicate(timeout=20)
                    completed = subprocess.CompletedProcess(
                        command, process.returncode, stdout, stderr
                    )
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            else:
                completed = subprocess.run(
                    command, env=env, capture_output=True, text=True, timeout=20
                )
            report = json.loads((root / "results/docker-job.json").read_text())
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(calls[0][:2], ["run", "--name"])
            self.assertEqual(calls[1], ["rm", "-f", report["container"]])
            self.assertIn("fixture output", Path(report["log"]).read_text())
            return completed.returncode, report

    def test_success_saves_log_and_cleans_only_own_container(self):
        code, report = self._run(False)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")

    def test_timeout_cleans_container_and_reports_failure(self):
        code, report = self._run(True)
        self.assertEqual(code, 124)
        self.assertEqual(report["status"], "timeout")

    def test_cleanup_failure_cannot_report_pass(self):
        code, report = self._run(False, cleanup_failure=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "cleanup_failed")

    def test_sigterm_cleans_container_and_saves_report(self):
        code, report = self._run(True, interrupt=True)
        self.assertEqual(code, 130)
        self.assertEqual(report["status"], "interrupted")


class TestIdleWatchdog(unittest.TestCase):
    @staticmethod
    def idle():
        import time

        return {
            "cpu_total_ns": 10,
            "io_total_bytes": 20,
            "gpu_percent": 0.0,
            "counter_read": str(time.monotonic_ns()),
        }

    def test_continuous_idle_requires_confirmation(self):
        watcher = supervisor.IdleWatchdog(60)
        for now in range(0, 60, 5):
            self.assertFalse(watcher.update(now, now, self.idle()))
        self.assertTrue(watcher.update(60, 60, self.idle()))
        self.assertFalse(watcher.confirmed)
        for now in range(65, 90, 5):
            self.assertTrue(watcher.update(now, now, self.idle()))
            self.assertFalse(watcher.confirmed)
        self.assertTrue(watcher.update(90, 90, self.idle()))
        self.assertTrue(watcher.confirmed)

    def test_activity_or_missing_metrics_restart_entire_window(self):
        for key, value in [
            ("cpu_total_ns", 11),
            ("io_total_bytes", 21),
            ("gpu_percent", 1),
            ("cpu_total_ns", None),
            ("io_total_bytes", None),
            ("gpu_percent", None),
            ("gpu_percent", float("nan")),
            ("cpu_total_ns", -1),
            ("cpu_total_ns", 0),
        ]:
            with self.subTest(key=key, value=value):
                watcher = supervisor.IdleWatchdog(60)
                for now in range(0, 55, 5):
                    self.assertFalse(watcher.update(now, now, self.idle()))
                changed = {**self.idle(), key: value}
                self.assertFalse(
                    watcher.update(55, 55, {**changed, "counter_read": str(now)})
                )
                self.assertFalse(watcher.update(60, 60, self.idle()))
                self.assertFalse(watcher.confirmed)

    def test_intermittent_cpu_bursts_cannot_alias_as_idle(self):
        watcher = supervisor.IdleWatchdog(60)
        for now in range(0, 600, 5):
            sample = {**self.idle(), "cpu_total_ns": 10 + now // 30}
            self.assertFalse(watcher.update(now, now, sample))
            self.assertFalse(watcher.confirmed)

    def test_formatted_zero_percent_without_raw_counters_is_unknown(self):
        watcher = supervisor.IdleWatchdog(60)
        sample = {"cpu_percent": 0, "gpu_percent": 0, "block_io": "0B / 0B"}
        for now in range(0, 300, 5):
            self.assertFalse(watcher.update(now, now, sample))

    def test_sampling_gap_does_not_count_as_continuous_observation(self):
        watcher = supervisor.IdleWatchdog(60)
        for now in (0, 5, 10, 15, 100, 105, 110):
            self.assertFalse(watcher.update(now, now, self.idle()))
        self.assertFalse(watcher.confirmed)

    def test_logs_during_confirmation_restart_observation(self):
        watcher = supervisor.IdleWatchdog(60)
        for now in range(0, 70, 5):
            watcher.update(now, now, self.idle())
        self.assertFalse(watcher.update(70, 0, self.idle()))
        for now in range(75, 130, 5):
            self.assertFalse(watcher.update(now, now - 70, self.idle()))
        self.assertFalse(watcher.confirmed)

    def test_io_burst_during_confirmation_restarts_observation(self):
        watcher = supervisor.IdleWatchdog(60)
        for now in range(0, 70, 5):
            watcher.update(now, now, self.idle())
        changed = {**self.idle(), "io_total_bytes": 21}
        for now in range(70, 130, 5):
            self.assertFalse(
                watcher.update(now, now, {**changed, "counter_read": str(now)})
            )
        self.assertFalse(watcher.confirmed)

    def test_zero_values_are_valid_only_when_accounting_explicit(self):
        watcher = supervisor.IdleWatchdog(60)
        for now in range(0, 100, 5):
            watcher.update(
                now,
                now,
                {
                    "cpu_total_ns": 0,
                    "io_total_bytes": 0,
                    "gpu_percent": 0,
                    "counter_read": str(now),
                },
            )
        self.assertTrue(watcher.confirmed)

    def test_repeated_stats_snapshot_cannot_confirm_idle(self):
        watcher = supervisor.IdleWatchdog(60)
        sample = self.idle()
        for now in range(0, 300, 5):
            self.assertFalse(watcher.update(now, now, sample))
            self.assertFalse(watcher.confirmed)


class TestStallLifecycle(unittest.TestCase):
    def run_job(self, action=None, activity=False, missing=False):
        from unittest.mock import patch

        clock = [0.0]
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryFile() as stdout,
        ):

            class Process:
                returncode = None

                def poll(self):
                    return self.returncode

                def wait(self, timeout):
                    if self.returncode is None:
                        clock[0] += 5
                        if clock[0] >= 150:
                            self.returncode = 0
                        else:
                            raise subprocess.TimeoutExpired("fixture", timeout)
                    return self.returncode

                def terminate(self):
                    self.returncode = -15

                def kill(self):
                    self.returncode = -9

            process = Process()
            process.stdout = stdout
            args = [
                "supervisor",
                "--timeout",
                "0",
                "--gpu-device",
                "0",
                "--results",
                directory,
                "--",
                "docker",
                "run",
                "fixture",
            ]
            if action:
                args[1:1] = ["--stall-action", action]

            def sample(*args):
                return {
                    "cpu_total_ns": int(clock[0] // 20) if activity else 0,
                    "io_total_bytes": None if missing else 0,
                    "gpu_percent": 0,
                    "counter_read": str(clock[0]),
                }

            with (
                patch.object(sys, "argv", args),
                patch.object(
                    supervisor.time, "monotonic", side_effect=lambda: clock[0]
                ),
                patch.object(
                    supervisor, "docker_endpoint", return_value="unix:///fake"
                ),
                patch.object(supervisor, "sample_resources", side_effect=sample),
                patch.object(supervisor.subprocess, "Popen", return_value=process),
                patch.object(
                    supervisor.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as commands,
                patch.object(supervisor, "save_diagnostics") as diagnostics,
            ):
                code = supervisor.main()
            report = json.loads((Path(directory) / "docker-job.json").read_text())
            commands.assert_called_once_with(
                ["docker", "rm", "-f", report["container"]],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return code, report, diagnostics.call_count

    def test_default_kill_waits_for_full_confirmation_and_cleans(self):
        code, report, diagnostics = self.run_job()
        self.assertEqual(code, 124)
        self.assertEqual(report["stall_action"], "kill")
        self.assertEqual(report["status"], "suspected_stall")
        self.assertGreaterEqual(report["seconds"], 90)
        self.assertEqual(diagnostics, 1)

    def test_explicit_warn_keeps_job_until_success(self):
        code, report, diagnostics = self.run_job(action="warn")
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(diagnostics, 1)

    def test_default_kill_preserves_quiet_bursty_cpu_job(self):
        code, report, diagnostics = self.run_job(activity=True)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(diagnostics, 0)

    def test_default_kill_preserves_job_with_missing_accounting(self):
        code, report, diagnostics = self.run_job(missing=True)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(diagnostics, 0)


class TestRawDockerStats(unittest.TestCase):
    def test_raw_counter_parsing_and_missing_io(self):
        import datetime as dt
        from unittest.mock import MagicMock, patch

        read = dt.datetime.now(dt.timezone.utc).isoformat()

        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = 200
        for entries, expected in [
            (
                [
                    {"op": "Read", "value": 2},
                    {"op": "Write", "value": 3},
                    {"op": "Total", "value": 5},
                ],
                5,
            ),
            ([], None),
            (None, None),
            ([{"op": "Read", "value": -1}], None),
        ]:
            with self.subTest(entries=entries):
                response.read.return_value = json.dumps(
                    {
                        "read": read,
                        "cpu_stats": {"cpu_usage": {"total_usage": 123}},
                        "blkio_stats": {"io_service_bytes_recursive": entries},
                    }
                ).encode()
                with patch.object(
                    supervisor, "UnixHTTPConnection", return_value=connection
                ):
                    if expected is None:
                        with self.assertRaises(ValueError):
                            supervisor.cumulative_resources("owned", "unix:///fake")
                    else:
                        self.assertEqual(
                            supervisor.cumulative_resources("owned", "unix:///fake"),
                            {
                                "cpu_total_ns": 123,
                                "io_total_bytes": expected,
                                "counter_read": read,
                            },
                        )
                connection.close.assert_called()

        response.read.return_value = json.dumps({"read": "2000-01-01T00:00:00Z"}).encode()
        with patch.object(supervisor, "UnixHTTPConnection", return_value=connection):
            with self.assertRaisesRegex(ValueError, "Stale"):
                supervisor.cumulative_resources("owned", "unix:///fake")
