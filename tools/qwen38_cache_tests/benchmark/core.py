"""CPU-only protocol validation, provenance, streaming metrics and pairing."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path

from . import VERSION

ROOT = Path(__file__).resolve().parents[3]
# Minimal child environment: unknown inherited variables (including credentials
# and unlisted runtime toggles) are not forwarded. All forwarded values are audited.
ENV_KEYS = (
    "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LD_LIBRARY_PATH",
    "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
    "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_HOME", "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_MODULE_LOADING", "CUDA_CACHE_PATH",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
    "SGLANG_ENABLE_TORCH_COMPILE", "SGLANG_QWEN38_ELASTIC_L2",
    "SGLANG_QWEN38_EXPERT_PACK_MANIFEST", "QWEN38_READY_TIMEOUT_SECONDS",
)


def subprocess_environment(source=None):
    source = os.environ if source is None else source
    effective = {key: source[key] for key in ENV_KEYS if key in source}
    effective.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                     PYTHONPATH=str(ROOT / "python"))
    return effective


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def command(args, cwd=ROOT):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def code_identity():
    diff = command(["git", "diff", "HEAD", "--binary"])["stdout"]
    # Include new Python sources too: an untracked benchmark must invalidate resume.
    sources = {str(p.relative_to(ROOT)): file_hash(p)
               for base in (ROOT / "python/sglang", ROOT / "tools/qwen38_cache_tests")
               for p in sorted(base.rglob("*.py"))}
    return {"sha": command(["git", "rev-parse", "HEAD"])["stdout"],
            "tree": command(["git", "rev-parse", "HEAD^{tree}"])["stdout"],
            "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "source_sha256": digest(sources), "sources": sources, "dirty_diff": diff}


def machine_identity():
    import importlib.metadata
    versions = {}
    for name in ("torch", "sglang", "flashinfer-python", "transformers", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"system": platform.platform(), "cpu": platform.processor(),
            "python": sys.version, "executable": sys.executable, "versions": versions,
            "cpu_info": command(["lscpu"]),
            "gpu": command(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                            "--format=csv,noheader"]),
            "effective_child_env": subprocess_environment()}


def telemetry():
    return {"gpu": command(["nvidia-smi", "--query-gpu=uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem",
                             "--format=csv,noheader"]),
            "processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader"])}


def token_file(path):
    ids = read(path)
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError(f"invalid saved token array: {path}")
    return ids


def named(items, what):
    names = [x["id"] for x in items]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"empty or duplicate {what}")
    if any(not n or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in n)
           for n in names):
        raise ValueError(f"unsafe {what} id")


def validate(suite, machine, inputs=None):
    if suite["version"] != VERSION:
        raise ValueError("suite version mismatch")
    named(suite["cases"], "case")
    named(suite["candidates"], "candidate")
    if suite["baseline"] not in [c["id"] for c in suite["candidates"]]:
        raise ValueError("baseline missing")
    if type(suite["rounds"]) is not int or suite["rounds"] < 1:
        raise ValueError("positive independent rounds required")
    if type(suite["seed"]) is not int:
        raise ValueError("integer seed required")
    for key in ("startup_timeout_s", "request_timeout_s", "replay_timeout_s"):
        if not isinstance(suite[key], (int, float)) or suite[key] <= 0:
            raise ValueError(f"positive {key} required")
    if machine["container_image"] in ("", "REPLACE_WITH_LOCAL_IMAGE_DIGEST"):
        raise ValueError("record actual container image identity")
    if not Path(machine["checkpoint"]).is_absolute() or not Path(machine["manifest"]).is_absolute():
        raise ValueError("checkpoint/manifest must be absolute local paths")
    if not machine["model_revision"]:
        raise ValueError("model revision required")
    if not 1024 <= machine["port"] <= 65535:
        raise ValueError("invalid port")
    for candidate in suite["candidates"]:
        loader = candidate["loader"]
        if loader.get("adaptive_cache") is not False:
            raise ValueError("isolation suite requires adaptive_cache=false")
        if loader.get("runtime_mode") != "threaded":
            raise ValueError("stage A production replay requires threaded")
        slots = loader["layer_cache_slots"]
        if len(slots) != suite["num_layers"] or any(type(n) is not int or not suite["top_k"] <= n <= suite["num_experts"] for n in slots):
            raise ValueError("invalid frozen slot vector")
        if any(k in loader for k in ("route_trace_path", "manifest_path")):
            raise ValueError("manifest and trace paths are managed by the suite")
        effective = candidate["expected_effective"]
        for key in ("chunked_prefill_size", "kv_cache_dtype", "max_running_requests",
                    "cuda_graph_backend_prefill", "cuda_graph_backend_decode",
                    "disable_overlap_schedule", "mem_fraction_static", "context_length",
                    "ple_offload_backend", "ple_offload_embedding", "disable_radix_cache",
                    "incremental_streaming_output", "stream_interval", "load_format"):
            if key not in effective:
                raise ValueError(f"missing effective assertion {key}")
        if effective["incremental_streaming_output"] or effective["stream_interval"] != 1 or effective["max_running_requests"] != 1:
            raise ValueError("requires cumulative output IDs, interval=1, concurrency=1")
        if effective["load_format"] != "qwen38_nvfp4_cache":
            raise ValueError("unexpected loader")
        if not isinstance(candidate["server_args"], list) or any(not isinstance(x, str) for x in candidate["server_args"]):
            raise ValueError("server_args must be argv array")
        managed = ("--host", "--port", "--model-path", "--model-loader-extra-config", "--revision")
        if any(x.split("=", 1)[0] in managed for x in candidate["server_args"]):
            raise ValueError("managed server option duplicated")
    layouts = {digest(c["loader"]["layer_cache_slots"]) for c in suite["candidates"]}
    if len(layouts) != 1:
        raise ValueError("candidate slot layouts must match; create a new suite for capacity changes")
    for case in suite["cases"]:
        if case["split"] not in ("tune", "holdout") or case["weight"] <= 0:
            raise ValueError("invalid split/weight")
        if case["state"] not in ("process_restart", "warm_sequence"):
            raise ValueError("state must be process_restart or warm_sequence")
        if case["state"] == "process_restart" and case["warmup_requests"]:
            raise ValueError("process_restart case cannot run request warmup")
        if case["state"] == "warm_sequence" and not case["warmup_requests"]:
            raise ValueError("warm_sequence requires explicit warmup")
        if not case["turns"]:
            raise ValueError("empty session")
        for turn in case["turns"] + case["warmup_requests"]:
            if Path(turn["tokens"]).name != turn["tokens"] or not turn["tokens"].endswith(".json"):
                raise ValueError("token files must be simple JSON filenames")
            sampling = turn["sampling"]
            if type(sampling["max_new_tokens"]) is not int or sampling["max_new_tokens"] < 1:
                raise ValueError("invalid output length")
            if sampling.get("sampling_seed") != suite["seed"] or "temperature" not in sampling:
                raise ValueError("freeze seed and sampling")
            if turn["length_policy"] not in ("fixed", "eos"):
                raise ValueError("invalid length policy")
            if turn["length_policy"] == "fixed" and sampling.get("ignore_eos") is not True:
                raise ValueError("fixed length requires ignore_eos")
            if inputs:
                ids = token_file(Path(inputs) / turn["tokens"])
                if len(ids) != turn["input_tokens"]:
                    raise ValueError("saved input length mismatch")
        # Continuation uses full previous input + generated IDs + saved suffix.
        bound = 0
        for turn in case["turns"]:
            bound += turn["input_tokens"] + turn["sampling"]["max_new_tokens"]
            if any(bound > c["expected_effective"]["context_length"] for c in suite["candidates"]):
                raise ValueError("session exceeds frozen context length")
    for arg in machine.get("deployment_args", []):
        if arg.split("=", 1)[0] in ("--host", "--port", "--model-path", "--model-loader-extra-config", "--revision"):
            raise ValueError("deployment_args duplicate managed option")
    return schedule(suite)


def schedule(suite):
    rng = random.Random(suite["seed"])
    rows = []
    for r in range(suite["rounds"]):
        cases = list(suite["cases"])
        rng.shuffle(cases)
        for case in cases:
            candidates = list(suite["candidates"])
            rng.shuffle(candidates)
            for order, candidate in enumerate(candidates):
                rows.append({"round": r, "case": case["id"], "candidate": candidate["id"],
                             "order": order, "pair": f"{r}:{case['id']}"})
    return rows


def check_effective(info, candidate, loader):
    for key, expected in candidate["expected_effective"].items():
        if info.get(key) != expected:
            raise ValueError(f"effective {key}: expected {expected!r}, got {info.get(key)!r}")
    actual = info.get("model_loader_extra_config")
    if isinstance(actual, str):
        actual = json.loads(actual)
    if actual != loader:
        raise ValueError("effective loader config mismatch")


class StreamMeasurement:
    """Cumulative /generate IDs; client receipt timestamps, never chunk counts."""
    def __init__(self):
        self.ids = []
        self.events = []
        self.first = self.last = None
        self.meta = {}
        self.text = None

    def feed(self, value, seconds):
        ids = value.get("output_ids")
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("stream lacks output token IDs")
        if ids[:len(self.ids)] != self.ids:
            raise ValueError("non-cumulative or changed stream IDs")
        added = len(ids) - len(self.ids)
        if added:
            if self.first is None:
                self.first = seconds
            self.last = seconds
        self.ids = ids.copy()
        self.meta = value.get("meta_info", {})
        self.text = value.get("text") or self.text
        self.events.append({"seconds": seconds, "cumulative_tokens": len(ids),
                            "new_tokens": added, "raw": value})

    def finish(self, elapsed, turn, input_count):
        reason = self.meta.get("finish_reason")
        kind = reason.get("type") if isinstance(reason, dict) else None
        n = len(self.ids)
        errors = []
        if kind not in ("length", "stop") or (kind == "length" and n != turn["sampling"]["max_new_tokens"]):
            errors.append("incomplete_or_failed_finish")
        if self.meta.get("completion_tokens") != n:
            errors.append("completion_count_mismatch")
        if self.meta.get("prompt_tokens") != input_count:
            errors.append("prompt_count_mismatch")
        if not n:
            errors.append("empty_output")
        if turn["length_policy"] == "fixed" and n != turn["sampling"]["max_new_tokens"]:
            errors.append("fixed_length_mismatch")
        return {"status": "failed" if errors else "ok", "errors": errors,
                "ttft_s": self.first, "tpot_s": (self.last - self.first) / (n - 1) if n >= 2 else None,
                "duration_s": elapsed, "input_tokens": input_count, "output_tokens": n,
                "output_ids": self.ids, "output_sha256": digest(self.ids), "text": self.text,
                "finish_reason": reason, "events": self.events,
                "timing_boundary": "client receipt; tokens in one chunk share a timestamp",
                "coalesced_chunks": sum(e["new_tokens"] > 1 for e in self.events)}


def describe(values):
    if not values:
        return {"n": 0, "median": None, "p95": None, "stdev": None, "tail_unstable": True}
    ordered = sorted(values)
    return {"n": len(values), "median": statistics.median(values),
            "p95": ordered[max(0, math.ceil(0.95 * len(values)) - 1)],
            "stdev": statistics.stdev(values) if len(values) >= 2 else None,
            "min": ordered[0], "max": ordered[-1], "tail_unstable": len(values) < 100}


def summarize(samples, suite, mode="benchmark"):
    # One request per session turn per independent process/round. Do not pool turns.
    rows, paired = [], []
    index = {(s["round"], s["case"], s["candidate"]): s for s in samples}
    for case in suite["cases"]:
        for candidate in suite["candidates"]:
            units = [s for s in samples if s["case"] == case["id"] and s["candidate"] == candidate["id"]]
            for turn_id in range(len(case["turns"])):
                valid = [s["requests"][turn_id] for s in units if s["status"] == "ok" and len(s["requests"]) > turn_id]
                row = {"case": case["id"], "candidate": candidate["id"], "turn": turn_id,
                       "expected_rounds": suite["rounds"], "failed_units": sum(s["status"] != "ok" for s in units),
                       "missing_units": suite["rounds"] - len(units),
                       "metrics": {m: describe([v[m] for v in valid if v[m] is not None])
                                   for m in ("ttft_s", "tpot_s", "duration_s")}}
                rows.append(row)
                if candidate["id"] == suite["baseline"]:
                    continue
                ratios, deviations = [], []
                for r in range(suite["rounds"]):
                    a = index.get((r, case["id"], candidate["id"]))
                    b = index.get((r, case["id"], suite["baseline"]))
                    if not a or not b or a["status"] != "ok" or b["status"] != "ok":
                        continue
                    x, y = a["requests"][turn_id], b["requests"][turn_id]
                    deviations.append({"round": r, "output_equal": x["output_ids"] == y["output_ids"]})
                    if x["tpot_s"] is not None and y["tpot_s"] is not None and y["tpot_s"] > 0:
                        ratios.append(x["tpot_s"] / y["tpot_s"])
                # Deterministic paired bootstrap; exploratory, five samples still weak.
                rng = random.Random(suite["seed"])
                boot = sorted(statistics.mean(rng.choices(ratios, k=len(ratios))) for _ in range(2000)) if ratios else []
                paired.append({"case": case["id"], "candidate": candidate["id"], "turn": turn_id,
                               "tpot_ratios": ratios, "ratio_stats": describe(ratios),
                               "mean_ratio_bootstrap_95": [boot[49], boot[1949]] if len(ratios) >= 5 else None,
                               "output_checks": deviations,
                               "correctness_review_required": any(not d["output_equal"] for d in deviations)})
    return {"version": VERSION, "mode": mode, "rows": rows, "paired": paired, "raw_samples": samples,
            "failed_samples": [s for s in samples if s["status"] != "ok"],
            "approved_for_deployment": False}


def resume_guard(path, identity):
    if Path(path).exists():
        old = read(path)
        if old["fingerprint"] != digest(identity):
            raise ValueError("resume fingerprint mismatch: code/input/environment/config changed")
        return old
    value = {"fingerprint": digest(identity), "identity": identity}
    write(path, value)
    return value
