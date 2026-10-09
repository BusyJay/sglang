#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REQUIRED_WORKSET_FILES = (
    "RESUME.md",
    "COMPACT.md",
    "STATE.md",
    "NEXT.md",
    "TESTS.md",
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        expanded = value
        for key, replacement in variables.items():
            expanded = expanded.replace("{" + key + "}", replacement)
        return os.path.expandvars(os.path.expanduser(expanded))
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def run_command(
    command: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    log_file=None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError(f"Invalid command: {command!r}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=cwd,
        env=merged_env,
        stdin=subprocess.DEVNULL,
        stdout=log_file or subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=check,
    )


def git_output(worktree: Path, *args: str) -> str:
    result = run_command(["git", *args], cwd=str(worktree), check=True)
    return (result.stdout or "").strip()


def gpu_memory(gpu_id: str) -> dict[str, int]:
    result = run_command(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
    )
    line = (result.stdout or "").strip().splitlines()[0]
    used, free, total = [int(part.strip()) for part in line.split(",")]
    return {"used_mib": used, "free_mib": free, "total_mib": total}


def wait_gpu_free(config: dict[str, Any], report) -> dict[str, int]:
    gpu = config["gpu"]
    deadline = time.monotonic() + int(gpu["release_timeout_seconds"])
    threshold = int(gpu["minimum_free_mib"])
    poll = float(gpu.get("poll_interval_seconds", 2))
    last = None
    while time.monotonic() < deadline:
        last = gpu_memory(str(gpu["id"]))
        report(f"GPU memory: {last}")
        if last["free_mib"] >= threshold:
            return last
        time.sleep(poll)
    raise TimeoutError(
        f"GPU {gpu['id']} did not reach {threshold} MiB free; last={last}"
    )


def http_ready(
    url: str, timeout_seconds: int, process: subprocess.Popen | None = None
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Process exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 300:
                    return
        except urllib.error.HTTPError as exc:
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(
        f"Health endpoint did not become ready: {url}; last={last_error}"
    )


def proc_start_ticks(pid: int) -> int | str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        try:
            os.kill(pid, 0)
            completed = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            identity = completed.stdout.strip()
            return identity or None
        except (ProcessLookupError, PermissionError, OSError):
            return None


def terminate_saved_process(runtime_path: Path, timeout_seconds: int, report) -> None:
    if not runtime_path.exists():
        return
    runtime = read_json(runtime_path)
    pid = int(runtime.get("test_pid", 0))
    expected_ticks = runtime.get("test_start_ticks")
    if pid <= 0:
        return
    actual_ticks = proc_start_ticks(pid)
    if actual_ticks is None or actual_ticks != expected_ticks:
        report(
            f"Saved test PID {pid} is no longer the same process; not terminating it"
        )
        return
    report(f"Stopping test process group {pid}")
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and proc_start_ticks(pid) is not None:
        time.sleep(1)
    if proc_start_ticks(pid) is not None:
        report(f"Test process group {pid} did not stop; sending SIGKILL")
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


class Workflow:
    def __init__(self, config_path: Path):
        raw = read_json(config_path)
        self.config_path = config_path.resolve()
        initial = {
            "worktree": str(Path(raw["worktree"]).expanduser().resolve()),
            "workset": str(Path(raw["workset"]).expanduser().resolve()),
            # Preserve this placeholder until a timestamped result directory
            # exists inside run().
            "result_dir": "{result_dir}",
        }
        self.config = expand(raw, initial)
        self.worktree = Path(self.config["worktree"]).resolve()
        self.workset = Path(self.config["workset"]).resolve()
        self.runtime_dir = self.workset / "artifacts" / "gpu-switch"
        self.runtime_path = self.runtime_dir / "runtime.json"
        self.state_path = self.runtime_dir / "state.json"
        self.lock_path = self.runtime_dir / "workflow.lock"
        self.dispatch_path = self.runtime_dir / "dispatch.json"
        self.queue_dir = self.runtime_dir / "queue"
        self.daemon_pid_path = self.runtime_dir / "supervisor.pid"
        self.daemon_log_path = self.runtime_dir / "supervisor.log"
        self.test_process: subprocess.Popen | None = None
        self.dev_stopped = False
        self.development_ready = False
        self.result: dict[str, Any] = {
            "started_at": now_iso(),
            "finished_at": None,
            "passed": False,
            "phase": "created",
            "tests": [],
            "errors": [],
        }

    def report(self, message: str) -> None:
        print(f"[{now_iso()}] {message}", flush=True)

    def dispatch(self) -> int:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with (self.runtime_dir / "dispatch.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._dispatch()

    def _dispatch(self) -> int:
        """Queue a job for the independent supervisor and return immediately."""
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        existing_jobs = sorted(
            [
                *self.queue_dir.glob("*.json"),
                *self.queue_dir.glob("*.running"),
            ]
        )
        if existing_jobs:
            raise RuntimeError(
                "A GPU-switch job is already pending or running: "
                + ", ".join(path.name for path in existing_jobs)
            )
        snapshot = self.preflight()
        daemon = self.daemon_status()
        if not daemon["running"]:
            raise RuntimeError(
                "GPU-switch supervisor is not running. Start it once from a "
                "regular terminal with: switch-test daemon-start --config ..."
            )
        stamp = dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{os.getpid()}"
        job_path = self.queue_dir / f"{job_id}.json"
        config_snapshot = self.runtime_dir / "configs" / f"{job_id}.json"
        atomic_json(config_snapshot, self.config)
        record = {
            "job_id": job_id,
            "config": str(config_snapshot),
            "dispatched_at": now_iso(),
            "source_commit": snapshot["commit"],
        }
        atomic_json(job_path, record)
        atomic_json(self.dispatch_path, record)
        atomic_json(
            self.state_path,
            {"phase": "dispatched", "dispatch": record},
        )
        print(
            json.dumps(
                {
                    "accepted": True,
                    "phase": "dispatched",
                    "job_id": job_id,
                    "supervisor_pid": daemon["pid"],
                    "log_path": str(self.daemon_log_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    def daemon_status(self) -> dict[str, Any]:
        if not self.daemon_pid_path.exists():
            return {"running": False, "pid": None}
        try:
            record = read_json(self.daemon_pid_path)
            pid = int(record["pid"])
            start_ticks = record["start_ticks"]
        except (KeyError, ValueError, json.JSONDecodeError):
            return {"running": False, "pid": None}
        current_identity = proc_start_ticks(pid)
        # Linux provides a stable /proc start tick. On platforms where only
        # liveness can be established, a missing stored identity falls back to
        # the PID-alive result instead of treating two None values as proof.
        running = current_identity is not None and (
            start_ticks is None or current_identity == start_ticks
        )
        return {"running": running, "pid": pid, "start_ticks": start_ticks}

    def daemon_start(self) -> int:
        """Double-fork a small supervisor that polls the on-disk job queue."""
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        status = self.daemon_status()
        if status["running"]:
            print(json.dumps(status), flush=True)
            return 0

        first = os.fork()
        if first > 0:
            os.waitpid(first, 0)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = self.daemon_status()
                if status["running"]:
                    print(json.dumps(status), flush=True)
                    return 0
                time.sleep(0.1)
            raise RuntimeError("Supervisor did not start")

        os.setsid()
        second = os.fork()
        if second > 0:
            os._exit(0)

        os.chdir("/")
        os.umask(0o077)
        devnull = os.open(os.devnull, os.O_RDONLY)
        logfd = os.open(
            self.daemon_log_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        os.dup2(devnull, 0)
        os.dup2(logfd, 1)
        os.dup2(logfd, 2)
        os.close(devnull)
        os.close(logfd)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        pid = os.getpid()
        atomic_json(
            self.daemon_pid_path,
            {"pid": pid, "start_ticks": proc_start_ticks(pid), "started_at": now_iso()},
        )
        self.daemon_loop()
        os._exit(0)

    def daemon_loop(self) -> None:
        poll = float(self.config.get("dispatch", {}).get("poll_interval_seconds", 1))
        delay = float(self.config.get("dispatch", {}).get("delay_seconds", 8))
        while True:
            jobs = sorted(self.queue_dir.glob("*.json"))
            if not jobs:
                time.sleep(poll)
                continue
            job_path = jobs[0]
            running_path = job_path.with_suffix(".running")
            try:
                with (self.runtime_dir / "dispatch.lock").open("w") as dispatch_lock:
                    fcntl.flock(dispatch_lock, fcntl.LOCK_EX)
                    os.replace(job_path, running_path)
                    job = read_json(running_path)
                time.sleep(delay)
                workflow = Workflow(Path(job["config"]))
                workflow.run()
                running_path.rename(running_path.with_suffix(".done"))
            except BaseException as exc:
                print(f"[{now_iso()}] supervisor job failed: {exc}", flush=True)
                with contextlib.suppress(OSError):
                    running_path.rename(running_path.with_suffix(".failed"))

    def daemon_stop(self) -> int:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "A GPU test is running; let it restore the service before stopping the supervisor"
                )
            if list(self.queue_dir.glob("*.json")) or list(
                self.queue_dir.glob("*.running")
            ):
                raise RuntimeError(
                    "A GPU job is pending; finish it or recover before stopping the supervisor"
                )
            return self._daemon_stop()

    def _daemon_stop(self) -> int:
        status = self.daemon_status()
        if not status["running"]:
            self.daemon_pid_path.unlink(missing_ok=True)
            return 0
        os.kill(int(status["pid"]), signal.SIGTERM)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if proc_start_ticks(int(status["pid"])) is None:
                self.daemon_pid_path.unlink(missing_ok=True)
                return 0
            time.sleep(0.2)
        raise TimeoutError("Supervisor did not stop")

    def set_phase(self, phase: str) -> None:
        self.result["phase"] = phase
        atomic_json(self.state_path, self.result)
        self.report(f"Phase: {phase}")

    def preflight(self) -> dict[str, Any]:
        completion = self.config.get("completion", {})
        if completion.get("required", False) and not completion.get("command"):
            raise ValueError("completion.command is required to wake the Codex session")
        if completion.get("required", False):
            executable = expand(
                completion["command"][0],
                {
                    "worktree": str(self.worktree),
                    "workset": str(self.workset),
                },
            )
            if not shutil_which(executable):
                raise FileNotFoundError(
                    f"Completion executable is not available: {executable}"
                )
        if not self.worktree.is_dir():
            raise FileNotFoundError(f"Worktree not found: {self.worktree}")
        if not (self.worktree / ".git").exists() and not git_output(
            self.worktree, "rev-parse", "--is-inside-work-tree"
        ):
            raise RuntimeError(f"Not a Git worktree: {self.worktree}")
        missing = [
            name
            for name in REQUIRED_WORKSET_FILES
            if not (self.workset / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Workset is incomplete: {missing}")
        if not shutil_which("nvidia-smi"):
            raise FileNotFoundError("nvidia-smi is not available")
        variables = {
            "worktree": str(self.worktree),
            "workset": str(self.workset),
            "result_dir": "",
        }
        for raw_command in self.config.get("preflight_commands", []):
            command = expand(raw_command, variables)
            completed = run_command(
                command, cwd=str(self.worktree), timeout=30, check=False
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Preflight failed: {command}\n{completed.stdout or ''}"
                )
        snapshot = {
            "captured_at": now_iso(),
            "commit": git_output(self.worktree, "rev-parse", "HEAD"),
            "branch": git_output(self.worktree, "branch", "--show-current"),
            "status": git_output(self.worktree, "status", "--short"),
            "diff_stat": git_output(self.worktree, "diff", "--stat"),
            "gpu": gpu_memory(str(self.config["gpu"]["id"])),
        }
        return snapshot

    def stop_development(self) -> None:
        development = self.config["development"]
        command = development["stop_command"]
        self.report(f"Stopping development backend: {command}")
        # Once a stop attempt begins, cleanup should always try the configured
        # start command. A stop script may return nonzero after partially or
        # fully stopping the backend.
        self.development_ready = False
        self.dev_stopped = True
        try:
            result = run_command(
                command,
                timeout=int(development.get("stop_timeout_seconds", 30)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            force_command = development.get("force_stop_command") or []
            if not force_command:
                raise TimeoutError(
                    "Development stop timed out and no force_stop_command is configured"
                )
            self.report(f"Development stop timed out; forcing stop: {force_command}")
            result = run_command(
                force_command,
                timeout=int(development.get("force_stop_timeout_seconds", 15)),
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Development stop failed: {result.stdout}")

    def start_development(self) -> None:
        command = self.config["development"]["start_command"]
        self.report(f"Starting development backend: {command}")
        result = run_command(
            command,
            timeout=int(self.config["development"].get("start_timeout_seconds", 30)),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Development start failed: {result.stdout}")
        http_ready(
            self.config["development"]["health_url"],
            int(self.config["development"]["health_timeout_seconds"]),
        )
        self.dev_stopped = False
        self.development_ready = True
        self.result["development_ready_at"] = now_iso()
        self.report("Development backend is ready")

    def start_test_server(self, result_dir: Path) -> None:
        variables = {
            "worktree": str(self.worktree),
            "workset": str(self.workset),
            "result_dir": str(result_dir),
        }
        test = expand(self.config["test"], variables)
        environment = os.environ.copy()
        environment.update(test.get("environment", {}))
        log_path = result_dir / "test-server.log"
        log_handle = log_path.open("w", encoding="utf-8")
        self.report(f"Starting test server: {test['start_command']}")
        self.test_process = subprocess.Popen(
            test["start_command"],
            cwd=test.get("cwd") or str(self.worktree),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        runtime = {
            "test_pid": self.test_process.pid,
            "test_start_ticks": proc_start_ticks(self.test_process.pid),
            "test_log": str(log_path),
            "result_dir": str(result_dir),
        }
        atomic_json(self.runtime_path, runtime)
        http_ready(
            test["health_url"],
            int(test["health_timeout_seconds"]),
            process=self.test_process,
        )
        self.report("Test server is ready")

    def stop_test_server(self) -> None:
        test_config = self.config["test"]
        timeout = int(test_config.get("shutdown_timeout_seconds", 60))
        if self.test_process is not None and self.test_process.poll() is None:
            self.report(f"Stopping test process group {self.test_process.pid}")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.test_process.pid, signal.SIGTERM)
            try:
                self.test_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.test_process.pid, signal.SIGKILL)
                self.test_process.wait(timeout=10)
        else:
            terminate_saved_process(self.runtime_path, timeout, self.report)
        self.test_process = None
        self.runtime_path.unlink(missing_ok=True)
        cleanup_command = test_config.get("cleanup_command") or []
        if cleanup_command:
            variables = {
                "worktree": str(self.worktree),
                "workset": str(self.workset),
                "result_dir": str(self.result.get("result_dir") or ""),
            }
            cleanup = run_command(
                expand(cleanup_command, variables),
                timeout=int(test_config.get("cleanup_timeout_seconds", 30)),
                check=False,
            )
            if cleanup.returncode != 0:
                self.report(f"Test cleanup command failed: {cleanup.stdout}")

    def run_tests(self, result_dir: Path) -> bool:
        all_passed = True
        variables = {
            "worktree": str(self.worktree),
            "workset": str(self.workset),
            "result_dir": str(result_dir),
        }
        for index, raw_test in enumerate(self.config.get("tests", []), start=1):
            test = expand(raw_test, variables)
            name = test.get("name", f"test-{index}")
            log_path = result_dir / f"test-{index:02d}-{safe_name(name)}.log"
            started = now_iso()
            self.report(f"Running test: {name}")
            with log_path.open("w", encoding="utf-8") as log_handle:
                try:
                    completed = run_command(
                        test["command"],
                        cwd=test.get("cwd") or str(self.worktree),
                        env=test.get("environment"),
                        timeout=int(test.get("timeout_seconds", 900)),
                        log_file=log_handle,
                        check=False,
                    )
                    exit_code = completed.returncode
                    error = None
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    error = "timeout"
            passed = exit_code == 0
            record = {
                "name": name,
                "started_at": started,
                "finished_at": now_iso(),
                "exit_code": exit_code,
                "passed": passed,
                "error": error,
                "log_path": str(log_path),
            }
            self.result["tests"].append(record)
            atomic_json(self.state_path, self.result)
            self.report(f"Test {name}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
                if self.config.get("stop_after_first_failure", True):
                    break
        return all_passed

    def run_batch_test(self, result_dir: Path) -> bool:
        """Run one foreground batch that owns server startup, tests and cleanup."""
        variables = {
            "worktree": str(self.worktree),
            "workset": str(self.workset),
            "result_dir": str(result_dir),
        }
        test = expand(self.config["test"], variables)
        command = test["batch_command"]
        self.result["batch_detail_path"] = test.get("result_json")
        atomic_json(self.state_path, self.result)
        log_path = result_dir / "batch.log"
        started = now_iso()
        self.report(f"Running test batch: {command}")
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = run_command(
                    command,
                    cwd=test.get("cwd") or str(self.worktree),
                    env=test.get("environment"),
                    timeout=int(test.get("batch_timeout_seconds", 0)) or None,
                    log_file=log_handle,
                    check=False,
                )
            exit_code = completed.returncode
            error = None
        except subprocess.TimeoutExpired:
            exit_code = 124
            error = "timeout"

        passed = exit_code == 0
        record = {
            "name": test.get("name", "docker-batch"),
            "started_at": started,
            "finished_at": now_iso(),
            "exit_code": exit_code,
            "passed": passed,
            "error": error,
            "log_path": str(log_path),
        }
        result_json = test.get("result_json")
        if result_json:
            detail_path = Path(expand(result_json, variables))
            record["detail_path"] = str(detail_path)
            if detail_path.is_file():
                try:
                    record["detail"] = read_json(detail_path)
                except (OSError, json.JSONDecodeError) as exc:
                    record["detail_error"] = f"{type(exc).__name__}: {exc}"
        self.result["tests"].append(record)
        atomic_json(self.state_path, self.result)
        return passed

    def run(self) -> int:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another GPU-switch workflow is already running")

            stamp = (
                dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
            )
            result_dir = self.runtime_dir / stamp
            result_dir.mkdir(parents=True, exist_ok=False)
            self.result["result_dir"] = str(result_dir)

            try:
                self.set_phase("preflight")
                snapshot = self.preflight()
                self.result["source_commit"] = snapshot["commit"]
                atomic_json(result_dir / "preflight.json", snapshot)

                self.set_phase("stopping-development")
                self.stop_development()

                self.set_phase("waiting-for-gpu")
                self.result["gpu_after_stop"] = wait_gpu_free(self.config, self.report)

                if self.config["test"].get("mode", "server") == "batch":
                    self.set_phase("running-test-batch")
                    tests_passed = self.run_batch_test(result_dir)
                else:
                    self.set_phase("starting-test-server")
                    self.start_test_server(result_dir)

                    self.set_phase("running-tests")
                    tests_passed = self.run_tests(result_dir)
                self.result["passed"] = tests_passed

            except BaseException as exc:
                self.result["errors"].append(f"{type(exc).__name__}: {exc}")
                self.report(self.result["errors"][-1])
                self.result["passed"] = False
            finally:
                if self.config["test"].get("mode", "server") == "server":
                    with contextlib.suppress(Exception):
                        self.set_phase("stopping-test-server")
                        self.stop_test_server()
                else:
                    cleanup_command = self.config["test"].get("cleanup_command") or []
                    if cleanup_command:
                        with contextlib.suppress(Exception):
                            self.set_phase("cleaning-test-batch")
                            variables = {
                                "worktree": str(self.worktree),
                                "workset": str(self.workset),
                                "result_dir": str(result_dir),
                            }
                            run_command(
                                expand(cleanup_command, variables),
                                timeout=int(
                                    self.config["test"].get(
                                        "cleanup_timeout_seconds", 30
                                    )
                                ),
                                check=False,
                            )
                if self.dev_stopped:
                    try:
                        self.set_phase("restoring-development")
                        self.start_development()
                    except Exception as exc:
                        self.result["errors"].append(
                            f"DevelopmentRestoreError: {type(exc).__name__}: {exc}"
                        )
                        self.report(self.result["errors"][-1])
                        self.result["passed"] = False
                self.result["finished_at"] = now_iso()
                self.result["phase"] = "complete"
                atomic_json(result_dir / "result.json", self.result)
                atomic_json(
                    self.workset / "artifacts" / "latest-test.json", self.result
                )
                atomic_json(self.state_path, self.result)
                self.dispatch_path.unlink(missing_ok=True)

                completion = self.config.get("completion", {})
                completion_command = completion.get("command") or []
                if completion_command and self.development_ready:
                    time.sleep(float(completion.get("delay_seconds", 0)))
                    variables = {
                        "worktree": str(self.worktree),
                        "workset": str(self.workset),
                        "result_dir": str(result_dir),
                    }
                    try:
                        completed = run_command(
                            expand(completion_command, variables),
                            cwd=str(self.worktree),
                            timeout=int(completion.get("timeout_seconds", 30)),
                            check=False,
                        )
                        self.result["completion"] = {
                            "exit_code": completed.returncode,
                            "output": completed.stdout,
                            "finished_at": now_iso(),
                        }
                        if completed.returncode != 0:
                            raise RuntimeError(
                                completed.stdout or f"exit {completed.returncode}"
                            )
                    except Exception as exc:
                        self.result["passed"] = False
                        self.result["errors"].append(
                            f"CompletionCommandError: {type(exc).__name__}: {exc}"
                        )
                    atomic_json(result_dir / "result.json", self.result)
                    atomic_json(
                        self.workset / "artifacts" / "latest-test.json", self.result
                    )
                    atomic_json(self.state_path, self.result)

            return 0 if self.result["passed"] and not self.result["errors"] else 1

    def recover(self) -> int:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with (
            (self.runtime_dir / "dispatch.lock").open("w") as dispatch_lock,
            self.lock_path.open("w") as lock,
        ):
            fcntl.flock(dispatch_lock, fcntl.LOCK_EX)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Cannot recover while a GPU test is running")
            if (
                list(self.queue_dir.glob("*.running"))
                and self.daemon_status()["running"]
            ):
                raise RuntimeError(
                    "Supervisor has claimed a GPU job; wait for restoration before recovery"
                )
            code = self._recover()
            if code == 0:
                for pending in [
                    *self.queue_dir.glob("*.running"),
                    *self.queue_dir.glob("*.json"),
                ]:
                    pending.rename(pending.with_suffix(".recovered"))
                self.dispatch_path.unlink(missing_ok=True)
            return code

    def _recover(self) -> int:
        errors = []
        try:
            terminate_saved_process(
                self.runtime_path,
                int(self.config["test"].get("shutdown_timeout_seconds", 60)),
                self.report,
            )
            self.runtime_path.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"test cleanup: {exc}")
        # The inner watchdog persists its unique container identity before launch,
        # allowing recovery after SIGKILL or a host supervisor crash.
        result_json = self.config.get("test", {}).get("result_json")
        if self.state_path.is_file():
            try:
                saved_state = read_json(self.state_path)
                result_dir = saved_state.get("result_dir")
                result_json = saved_state.get("batch_detail_path") or result_json
                if result_dir and result_json:
                    detail_path = Path(
                        expand(
                            saved_state.get("batch_detail_path") or result_json,
                            {
                                "worktree": str(self.worktree),
                                "workset": str(self.workset),
                                "result_dir": result_dir,
                            },
                        )
                    )
                    if detail_path.is_file():
                        detail = read_json(detail_path)
                        name = detail.get("container", "")
                        if not re.fullmatch(r"qwen38-test-[0-9a-f]{16}", name):
                            raise ValueError("Invalid saved test container identity")
                        completed = run_command(
                            ["docker", "rm", "-f", name], timeout=30, check=False
                        )
                        if completed.returncode and "No such container" not in (
                            completed.stdout or ""
                        ):
                            raise RuntimeError(completed.stdout)
            except Exception as exc:
                errors.append(f"saved Docker cleanup: {exc}")
        cleanup_command = self.config.get("test", {}).get("cleanup_command") or []
        if cleanup_command:
            try:
                variables = {
                    "worktree": str(self.worktree),
                    "workset": str(self.workset),
                    "result_dir": "",
                }
                completed = run_command(
                    expand(cleanup_command, variables),
                    timeout=int(
                        self.config.get("test", {}).get("cleanup_timeout_seconds", 30)
                    ),
                    check=False,
                )
                if completed.returncode != 0:
                    errors.append(f"configured cleanup: {completed.stdout}")
            except Exception as exc:
                errors.append(f"configured cleanup: {exc}")
        try:
            self.start_development()
        except Exception as exc:
            errors.append(f"development restore: {exc}")
        state = {
            "result_dir": read_json(self.state_path).get("result_dir")
            if self.state_path.is_file()
            else None,
            "batch_detail_path": read_json(self.state_path).get("batch_detail_path")
            if self.state_path.is_file()
            else None,
            "phase": "recovered" if not errors else "recovery-failed",
            "finished_at": now_iso(),
            "errors": errors,
        }
        atomic_json(self.state_path, state)
        return 0 if not errors else 1


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-")


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "dispatch",
            "daemon-start",
            "daemon-stop",
            "daemon-status",
            "worker",
            "run",
            "check",
            "recover",
            "status",
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--delay-seconds", type=float, default=0)
    args = parser.parse_args()

    workflow = Workflow(args.config)
    if args.action == "dispatch":
        return workflow.dispatch()
    if args.action == "daemon-start":
        return workflow.daemon_start()
    if args.action == "daemon-stop":
        return workflow.daemon_stop()
    if args.action == "daemon-status":
        print(json.dumps(workflow.daemon_status()), flush=True)
        return 0
    if args.action == "worker":
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)
        return workflow.run()
    if args.action == "run":
        return workflow.run()
    if args.action == "check":
        snapshot = workflow.preflight()
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return 0
    if args.action == "recover":
        return workflow.recover()
    if workflow.state_path.exists():
        print(workflow.state_path.read_text(encoding="utf-8"), end="")
    else:
        print('{"phase":"not-started"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
