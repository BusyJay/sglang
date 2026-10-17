"""Owned-process full-model runner; failures and partial streams are artifacts."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .core import (ROOT, StreamMeasurement, check_effective, read, telemetry,
                   token_file, write, subprocess_environment, source_loader_config)


def request(url, data=None, timeout=30):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def assert_port_available(port):
    # Like the HTTP server, allow reuse after an owned process exits while
    # its connections remain in TIME_WAIT. A live listener still rejects bind.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def generate(url, ids, turn, timeout, artifact):
    measure = StreamMeasurement()
    started = time.perf_counter()
    body = json.dumps({"input_ids": ids, "sampling_params": turn["sampling"], "stream": True}).encode()
    req = urllib.request.Request(url + "/generate", body, {"Content-Type": "application/json"})
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            # A socket deadline bounds silent streams and the total request lifetime.
            while True:
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("total request deadline exceeded")
                response.fp.raw._sock.settimeout(remaining)
                line = response.readline()
                if not line:
                    break
                received = time.perf_counter() - started
                if received > timeout:
                    raise TimeoutError("total request deadline exceeded")
                if line.startswith(b"data:"):
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    measure.feed(json.loads(payload), received)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = measure.finish(time.perf_counter() - started, turn, len(ids))
    result["input_ids"] = ids
    if error:
        result["status"] = "failed"
        result["errors"].append(error)
    write(artifact, result)
    return result


def stop_owned(process):
    if process is None:
        return
    # Kill the owned group even if its launcher has exited leaving children.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def log_snapshots(path):
    values = []
    if Path(path).exists():
        for line_no, line in enumerate(
            Path(path).read_text(errors="replace").splitlines(), 1
        ):
            if "Qwen38ExpertCacheStats[" in line:
                try:
                    from sglang.srt.layers.moe.qwen38_cache_stats_format import (
                        expand_stats,
                    )

                    payload = line[line.index("{", line.index("Qwen38ExpertCacheStats[")) :]
                    values.append(
                        {"line": line_no, "raw": line,
                         "stats": expand_stats(json.loads(payload))}
                    )
                except (ValueError, json.JSONDecodeError):
                    continue
    return values


def run_unit(unit, suite, machine, inputs, output, mode, fingerprint=None):
    candidate = next(c for c in suite["candidates"] if c["id"] == unit["candidate"])
    case = next(c for c in suite["cases"] if c["id"] == unit["case"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loader = {**candidate["loader"], "manifest_path": machine["manifest"]}
    loader, omitted_controls = source_loader_config(loader)
    if mode == "trace":
        loader.update(route_trace_path=str((output / "routes.raw.jsonl").resolve()),
                      route_trace_limit=suite["trace_limit"])
    argv = [sys.executable, "-m", "sglang.launch_server", "--model-path", machine["checkpoint"],
            "--revision", machine["model_revision"], "--host", "127.0.0.1", "--port", str(machine["port"]),
            "--model-loader-extra-config", json.dumps(loader), *candidate["server_args"],
            *machine.get("deployment_args", [])]
    # Deployment argv may add remote ViT/memory options; resolved assertions remain authoritative.
    env = subprocess_environment()
    # Independently check what a new interpreter actually imports, before any model load.
    result = {**unit, "fingerprint": fingerprint, "mode": mode, "status": "failed", "requests": [], "warmup": [],
              "command": argv, "requested_loader": loader, "state": case["state"],
              "cache_state": suite["cache_state"], "telemetry_before": telemetry(),
              "correctness": "completion/length/counts; semantic and generation parity require GPU review",
              "retry_policy": "none; failed units preserved and not silently rerun",
              "enforced_environment": env}
    result["omitted_unsupported_neutral_controls"] = omitted_controls
    process = None
    log_path = output / "server.log"
    try:
        probe = subprocess.run([sys.executable, "-c", "import json,sglang,sglang.srt.layers.moe.qwen38_cache_threaded as c; print(json.dumps({'sglang':sglang.__file__,'runtime':c.__file__}))"],
                               cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
        write(output / "import_probe.json", {"returncode": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr})
        if probe.returncode:
            raise RuntimeError("import probe failed; inspect import_probe.json")
        imports = json.loads(probe.stdout.strip().splitlines()[-1])
        if any(not Path(p).resolve().is_relative_to(ROOT / "python/sglang") for p in imports.values()):
            raise RuntimeError("server imports another installed source tree")
        result["actual_imports"] = imports
        assert_port_available(machine["port"])
        with log_path.open("w") as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            url = f"http://127.0.0.1:{machine['port']}"
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {process.returncode}")
                if time.monotonic() - started > suite["startup_timeout_s"]:
                    raise TimeoutError("startup timeout")
                try:
                    with urllib.request.urlopen(url + "/ready", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                time.sleep(0.5)
            info = request(url + "/server_info")
            write(output / "effective_server_info.json", info)
            check_effective(info, candidate, loader)
            if info.get("model_path") != machine["checkpoint"] or info.get("revision") != machine["model_revision"]:
                raise ValueError("effective model identity mismatch")
            for i, turn in enumerate(case["warmup_requests"]):
                sample = generate(url, token_file(Path(inputs) / turn["tokens"]), turn,
                                  suite["request_timeout_s"], output / f"warmup-{i}.json")
                result["warmup"].append(sample)
                if sample["status"] != "ok":
                    raise RuntimeError("warmup failed")
            if mode == "profile":
                request(url + "/start_profile", {"output_dir": str((output / "profiler").resolve())}, timeout=60)
            history = []
            try:
                for i, turn in enumerate(case["turns"]):
                    ids = history + token_file(Path(inputs) / turn["tokens"])
                    begin = log_snapshots(log_path)
                    sample = generate(url, ids, turn, suite["request_timeout_s"], output / f"request-{i}.json")
                    end = log_snapshots(log_path)
                    sample["cache_statistics"] = {
                        "alignment": "periodic log snapshots; no exact request counter delta available",
                        "before": begin[-1:] or None, "after": end[-1:] or None,
                        "request_log_snapshots": end[len(begin):], "exact_delta": None}
                    result["requests"].append(sample)
                    write(output / f"request-{i}.json", sample)
                    if sample["status"] != "ok":
                        raise RuntimeError(f"request {i} failed")
                    answer = sample["output_ids"].copy()
                    while answer and answer[-1] in suite.get("session_strip_tail_ids", []):
                        answer.pop()
                    history = ids + answer
            finally:
                if mode == "profile":
                    request(url + "/stop_profile", {}, timeout=60)
            result["status"] = "ok"
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        stop_owned(process)
        result["telemetry_after"] = telemetry()
        result["all_cache_snapshots"] = log_snapshots(log_path)
        observed = result["all_cache_snapshots"][-1:] or []
        result["runtime_slot_observation"] = {
            "status": "observed" if observed else "not_available",
            "layer_cache_slots": observed[-1]["stats"].get("layer_cache_slots") if observed else None,
            "expected": loader["layer_cache_slots"]}
        if observed and observed[-1]["stats"].get("layer_cache_slots") != loader["layer_cache_slots"]:
            result["status"] = "failed"
            result["error"] = "runtime slot vector differs from frozen configuration"
        write(output / "unit.json", result)
    return result
