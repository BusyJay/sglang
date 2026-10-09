#!/usr/bin/env python3
"""Bound a single Docker job and remove only its named container on exit."""

from __future__ import annotations

import argparse
import codecs
import datetime as dt
import http.client
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path


class IdleWatchdog:
    """Detect observed inactivity; instantaneous percentages never authorize kill.

    Raw cumulative counters cover bursts between samples. Gaps, unknown metrics,
    new logs, counter resets and any CPU/IO progress restart the whole window.
    """

    def __init__(self, seconds=60, confirmation_seconds=30, max_gap=20):
        self.seconds = seconds
        self.confirmation_seconds = confirmation_seconds
        self.max_gap = max_gap
        self.since = None
        self.previous = None
        self.samples = 0
        self.confirmed = False

    def update(self, now, log_age, sample):
        self.confirmed = False
        counters = (sample.get("cpu_total_ns"), sample.get("io_total_bytes"))
        gpu = sample.get("gpu_percent")
        known = all(type(value) is int and value >= 0 for value in counters)
        known = (
            known and isinstance(gpu, (int, float)) and math.isfinite(gpu) and gpu == 0
        )
        stamp = sample.get("counter_read")
        known = known and isinstance(stamp, str) and bool(stamp)
        previous = self.previous
        self.previous = (now, log_age, counters, stamp) if known else None
        quiet = (
            known
            and previous is not None
            and 0 < now - previous[0] <= self.max_gap
            and counters == previous[2]
            and stamp != previous[3]
            and log_age >= previous[1] + now - previous[0] - 0.05
        )
        if not quiet:
            self.since = None
            self.samples = 0
            return False
        if self.since is None:
            self.since = previous[0]
        self.samples += 1
        duration = now - self.since
        suspected = (
            duration >= self.seconds and log_age >= self.seconds and self.samples >= 3
        )
        self.confirmed = (
            suspected and duration >= self.seconds + self.confirmation_seconds
        )
        return suspected


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=4)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def docker_endpoint():
    """Use the same daemon as the Docker CLI; unsupported endpoints stay unknown."""
    # DOCKER_CONTEXT overrides DOCKER_HOST in the CLI.
    if os.environ.get("DOCKER_HOST") and not os.environ.get("DOCKER_CONTEXT"):
        return os.environ["DOCKER_HOST"]
    result = subprocess.run(
        ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
        capture_output=True,
        text=True,
        timeout=4,
        check=True,
    )
    return result.stdout.strip()


def cumulative_resources(name, endpoint):
    if not endpoint or not endpoint.startswith("unix://"):
        raise ValueError(
            "Raw stats require a local Docker Unix socket; inactivity is unknown"
        )
    connection = UnixHTTPConnection(endpoint[len("unix://") :])
    try:
        connection.request(
            "GET", f"/containers/{name}/stats?stream=false&one-shot=true"
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"Docker stats HTTP {response.status}")
        stats = json.loads(response.read())
        read = stats.get("read")
        observed = (
            dt.datetime.fromisoformat(read.replace("Z", "+00:00"))
            if isinstance(read, str)
            else None
        )
        if (
            observed is None
            or observed.tzinfo is None
            or not -5
            <= (dt.datetime.now(dt.timezone.utc) - observed).total_seconds()
            <= 15
        ):
            raise ValueError("Stale or missing Docker stats timestamp")
        cpu = stats["cpu_stats"]["cpu_usage"]["total_usage"]
        entries = stats["blkio_stats"].get("io_service_bytes_recursive")
        # Empty or unavailable block accounting is unknown, never synthesized zero.
        if not isinstance(entries, list) or not entries:
            raise ValueError("Docker does not expose cumulative block IO")
        values = [
            entry["value"]
            for entry in entries
            if str(entry["op"]).lower() in ("read", "write")
        ]
        if (
            not values
            or type(cpu) is not int
            or cpu < 0
            or any(type(v) is not int or v < 0 for v in values)
        ):
            raise ValueError("Invalid raw CPU/IO counters")
        return {
            "cpu_total_ns": cpu,
            "io_total_bytes": sum(values),
            "counter_read": read,
        }
    finally:
        connection.close()


def save_diagnostics(name, output):
    diagnostics = {}
    for label, command in (
        ("inspect", ["docker", "inspect", "--format", "{{json .State}}", name]),
        ("processes", ["docker", "top", name, "-eo", "pid,ppid,stat,etime,time,comm"]),
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=4, check=False
            )
            diagnostics[label] = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as error:
            diagnostics[label] = {"error": str(error)}
    # Keep diagnostic files owner-readable; inspect includes only container state.
    path = output / "stall-diagnostics.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(diagnostics, handle, indent=2)


def sample_resources(name, gpu_device, endpoint=None):
    sample = {"cpu_percent": None, "gpu_percent": None, "block_io": None}
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
            capture_output=True,
            text=True,
            timeout=4,
        )
        result.check_returncode()
        value = json.loads(result.stdout.strip())
        sample["cpu_percent"] = float(value["CPUPerc"].rstrip("%"))
        sample["block_io"] = value.get("BlockIO")
    except Exception as error:
        sample["cpu_error"] = str(error)
    try:
        sample.update(cumulative_resources(name, endpoint))
    except Exception as error:
        sample["cumulative_error"] = str(error)
    if gpu_device is not None and shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    gpu_device,
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if result.returncode:
                raise ValueError(result.stderr.strip())
            sample["gpu_percent"] = max(
                float(line.strip()) for line in result.stdout.splitlines()
            )
        except Exception as error:
            sample["gpu_error"] = str(error)
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--gpu-device")
    parser.add_argument("--stall-action", choices=["warn", "kill"], default="kill")
    parser.add_argument("--confirm-seconds", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (
        not all(
            math.isfinite(value)
            for value in (args.idle_seconds, args.poll_seconds, args.confirm_seconds)
        )
        or args.timeout < 0
        or args.idle_seconds <= 0
        or args.poll_seconds <= 0
        or args.confirm_seconds < 30
    ):
        parser.error(
            "timeout must be nonnegative; idle/poll must be positive and finite; confirmation >= 30 seconds"
        )
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if command[:2] != ["docker", "run"]:
        parser.error("expected docker run command")
    name = "qwen38-test-" + uuid.uuid4().hex[:16]
    command = command[:2] + ["--name", name] + command[2:]
    output = Path(args.results)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_log = [started]
    watchdog = IdleWatchdog(
        args.idle_seconds, args.confirm_seconds, max_gap=max(20, args.poll_seconds * 3)
    )
    endpoint = None
    endpoint_error = None
    if args.gpu_device is not None:
        try:
            endpoint = docker_endpoint()
        except Exception as error:
            endpoint_error = str(error)
    warned = False
    observed_server = None
    history = []
    report = {
        "container": name,
        "command": command,
        "timeout_seconds": args.timeout,
        "idle_seconds": args.idle_seconds,
        "stall_action": args.stall_action,
        "confirm_seconds": args.confirm_seconds,
        "docker_endpoint_error": endpoint_error,
        "gpu_device": args.gpu_device,
        "status": "running",
        "exit_code": None,
    }

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    (output / "docker-job.json").write_text(json.dumps(report, indent=2) + "\n")
    process = None
    reader = None
    log_path = output / (name + ".log")
    print(f"Docker job {name}; log: {log_path}", flush=True)
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )

            def relay():
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    last_log[0] = time.monotonic()
                    value = decoder.decode(chunk)
                    log.write(value)
                    log.flush()
                    print(value, end="", flush=True)
                tail = decoder.decode(b"", final=True)
                log.write(tail)
                log.flush()

            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            try:
                next_sample = started
                while process.poll() is None:
                    now = time.monotonic()
                    # Server output is redirected by the canary; inspect the
                    # mounted server.log as well as container stdout.
                    server_log = output / "server.log"
                    if server_log.is_file():
                        stamp = (
                            server_log.stat().st_size,
                            server_log.stat().st_mtime_ns,
                        )
                        if stamp != observed_server:
                            last_log[0] = now
                            observed_server = stamp
                    if args.timeout and now - started >= args.timeout:
                        report.update(status="timeout", exit_code=124)
                        break
                    if now >= next_sample and args.gpu_device is not None:
                        sample = sample_resources(name, args.gpu_device, endpoint)
                        now = time.monotonic()
                        # Sampling may block. Recheck redirected logs before making
                        # a termination decision, including activity during sampling.
                        if server_log.is_file():
                            stamp = (
                                server_log.stat().st_size,
                                server_log.stat().st_mtime_ns,
                            )
                            if stamp != observed_server:
                                last_log[0] = now
                                observed_server = stamp
                        sample.update(
                            seconds=now - started, log_silence_seconds=now - last_log[0]
                        )
                        history.append(sample)
                        history = history[-120:]
                        suspected = watchdog.update(now, now - last_log[0], sample)
                        sample["idle_observation_seconds"] = (
                            now - watchdog.since if watchdog.since is not None else 0
                        )
                        sample["idle_confirmed"] = watchdog.confirmed
                        (output / "activity.json").write_text(
                            json.dumps(sample, indent=2) + "\n"
                        )
                        if suspected and not warned:
                            warned = True
                            report.setdefault("stall_observations", []).append(
                                {"seconds": now - started, "sample": sample.copy()}
                            )
                            report["stall_observations"] = report["stall_observations"][
                                -20:
                            ]
                            print(
                                "Suspected inactivity; saving diagnostics, continuing confirmation/observation",
                                flush=True,
                            )
                            try:
                                save_diagnostics(name, output)
                            except Exception as error:
                                report["diagnostic_error"] = str(error)
                        elif not suspected:
                            warned = False
                        if (
                            watchdog.confirmed
                            and args.stall_action == "kill"
                            and process.poll() is None
                        ):
                            report.update(
                                status="suspected_stall",
                                exit_code=124,
                                reason="Confirmed inactivity: unchanged raw CPU/IO counters, zero sampled GPU, quiet logs, plus confirmation window",
                            )
                            print(
                                "Confirmed inactivity; removing " + name,
                                flush=True,
                            )
                            break
                        next_sample = time.monotonic() + args.poll_seconds
                    try:
                        code = process.wait(timeout=min(args.poll_seconds, 1))
                        report.update(
                            status="passed" if code == 0 else "failed", exit_code=code
                        )
                    except subprocess.TimeoutExpired:
                        pass
                if report["exit_code"] is None and process.poll() is not None:
                    code = process.returncode
                    report.update(
                        status="passed" if code == 0 else "failed", exit_code=code
                    )
            finally:
                # Scope cleanup to our unique container even on interrupt or a
                # hung CUDA context. Never stop any pre-existing GPU service.
                try:
                    removed = subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if removed.returncode and "No such container" not in removed.stderr:
                        report["cleanup_error"] = removed.stderr.strip()
                except Exception as error:
                    report["cleanup_error"] = str(error)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                reader.join(timeout=5)
    except KeyboardInterrupt:
        report.update(status="interrupted", exit_code=130)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if report.get("cleanup_error") and report["exit_code"] in (None, 0):
            report.update(status="cleanup_failed", exit_code=1)
        report["activity_history"] = history
        report["seconds"] = time.monotonic() - started
        report["log"] = str(log_path)
        (output / "docker-job.json").write_text(json.dumps(report, indent=2) + "\n")
    return report["exit_code"] if report["exit_code"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
