"""Run from the repository root: python -m tools.qwen38_cache_tests.benchmark."""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

from .core import (ROOT, code_identity, file_hash, machine_identity,
                   read, resume_guard, schedule, summarize, token_file, validate, write, subprocess_environment)
from .runner import run_unit, stop_owned
from .replay import validate_trace

RAFT = "纯逻辑推理，不调用任何工具回答：Raft 算法的整体流程。"
ARBITER = "纯逻辑推理，不调用任何工具回答：如果要实现一个 arbiter（只参与投票和成员变更，不参与普通日志复制和提交），应该怎么调整 Raft 算法？请结合前面的流程分析安全性和实现细节。"


def identity(suite, machine, inputs, extra=None):
    manifest = read(machine["manifest"])
    payload = Path(manifest["payload"])
    if not payload.is_absolute():
        payload = Path(machine["manifest"]).parent / payload
    # Full payload hash is expensive but necessary: cached at preparation by operator?
    # Always hash here so resume cannot silently accept a replaced pack.
    value = {"suite": suite, "machine_config": machine, "code": code_identity(),
             "environment": machine_identity(), "manifest_sha256": file_hash(machine["manifest"]),
             "payload_sha256": file_hash(payload),
             "model_metadata": {p.name: file_hash(p) for p in sorted(Path(machine["checkpoint"]).glob("*.json"))},
             "checkpoint_files": {str(p.relative_to(machine["checkpoint"])): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                 for p in sorted(Path(machine["checkpoint"]).rglob("*")) if p.is_file()},
             "checkpoint_identity_scope": "declared model revision + JSON hashes + all file size/mtime signatures; checkpoint weights not fully hashed",
             "inputs": {str(p.relative_to(inputs)): file_hash(p) for p in sorted(Path(inputs).rglob("*")) if p.is_file()},
             "input_artifacts": {str(p.relative_to(inputs)): {"path": str(p.resolve()), "sha256": file_hash(p)}
                 for p in sorted(Path(inputs).rglob("*")) if p.is_file()},
             "extra": extra}
    if manifest["checkpoint_revision"] != machine["model_revision"]:
        raise ValueError("model revision differs from pack")
    return value


def prepare(args):
    from transformers import AutoTokenizer
    suite = read(args.suite)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("prepare requires a new empty directory; inputs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    # Some tokenizer versions return BatchEncoding for tokenize=True.
    # Rendering first keeps frozen input files as flat token-ID lists.
    chat = lambda text: encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))
    first = chat(RAFT)
    write(out / "raft.json", first)
    sentinel = "BENCHMARK_ASSISTANT_SENTINEL_71ecca"
    rendered = tokenizer.apply_chat_template([
        {"role": "user", "content": RAFT}, {"role": "assistant", "content": sentinel},
        {"role": "user", "content": ARBITER}], tokenize=False, add_generation_prompt=True)
    if rendered.count(sentinel) != 1:
        raise ValueError("cannot derive tokenizer chat continuation boundary")
    suffix = encode(rendered.split(sentinel, 1)[1])
    write(out / "arbiter-suffix.json", suffix)
    # eos before a saved assistant/user boundary is removed once to avoid duplicate separators.
    suite["session_strip_tail_ids"] = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    warm = chat("纯逻辑推理，不调用工具：简述多数派投票的作用。")
    write(out / "warmup.json", warm)
    for case in suite["cases"]:
        for turn in case["turns"] + case["warmup_requests"]:
            name = turn["tokens"]
            if name in ("raft.json", "arbiter-suffix.json", "warmup.json"):
                turn["input_tokens"] = len(token_file(out / name))
                continue
            target = turn["input_tokens"]
            phrase = ("Raft 节点记录任期、投票、日志索引及提交位置。请分析网络分区和恢复。\n"
                      if case["split"] == "tune" else
                      "考虑节点重启、消息乱序和成员配置切换；逐步检查共识协议的安全条件。\n")
            # Pad inside the user message, preserving the closing chat/template suffix.
            marker = "BENCHMARK_PADDING_92bdb"
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": marker + RAFT}], tokenize=False, add_generation_prompt=True)
            prefix, tail = rendered.split(marker)
            a, b, fill = encode(prefix), encode(tail), encode(phrase)
            count = target - len(a) - len(b)
            if count < 0 or not fill:
                raise ValueError("context too short for chat template")
            ids = a + (fill * ((count + len(fill) - 1) // len(fill)))[:count] + b
            write(out / name, ids)
    write(out / "suite.json", suite)
    write(out / "preparation.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
          "prompts": [RAFT, ARBITER], "kind": "synthetic repeated text for long contexts; user reasoning prompts for short/session",
          "tokenizer_metadata": {p.name: file_hash(p) for p in Path(args.checkpoint).glob("*token*.json")},
          "files": {p.name: file_hash(p) for p in out.glob("*.json")}})
    print(str(out / "suite.json"))


def freeze_answer(args):
    suite = read(args.suite)
    unit = read(args.source_unit)
    if unit["status"] != "ok" or unit["case"] != "raft-session":
        raise ValueError("requires a successful raft-session source unit")
    first = unit["requests"][0]
    if first["status"] != "ok":
        raise ValueError("first answer failed")
    ids = first["output_ids"].copy()
    strip = suite.get("session_strip_tail_ids", [])
    while ids and ids[-1] in strip:
        ids.pop()
    frozen = first["input_ids"] + ids + token_file(Path(args.inputs) / "arbiter-suffix.json")
    destination = Path(args.inputs) / "raft-answer-prefill.json"
    if destination.exists():
        raise ValueError("refusing to overwrite frozen answer")
    write(destination, frozen)
    case = copy.deepcopy(next(c for c in suite["cases"] if c["id"] == "raft-session"))
    case.update(id="raft-answer-prefill", state="process_restart", warmup_requests=[],
                purpose="saved baseline first answer + arbiter question; fixed input for prefill comparisons")
    turn = copy.deepcopy(case["turns"][1])
    turn.update(tokens=destination.name, input_tokens=len(frozen))
    case["turns"] = [turn]
    suite["cases"].append(case)
    write(args.output_suite, suite)
    write(destination.with_suffix(".provenance.json"), {"source_unit_sha256": file_hash(args.source_unit),
          "source_round": unit["round"], "source_candidate": unit["candidate"],
          "answer_sha256": first["output_sha256"], "frozen_input_sha256": file_hash(destination),
          "answer_finish_reason": first["finish_reason"],
          "requires_semantic_review": True})


def full_run(args):
    suite, machine = read(args.suite), read(args.machine)
    validate(suite, machine, args.inputs)
    if args.rounds is not None and not 1 <= args.rounds <= suite["rounds"]:
        raise ValueError("rounds must be in configured range")
    rows = schedule(suite)
    if args.case:
        rows = [u for u in rows if u["case"] == args.case]
        if not rows:
            raise ValueError("unknown case")
    if args.rounds is not None:
        rows = [u for u in rows if u["round"] < args.rounds]
    out = Path(args.results)
    if out.exists() and any(p.name != ".lock" for p in out.iterdir()) and not args.resume:
        raise ValueError("results exist; use --resume after fingerprint validation")
    record = resume_guard(out / "run.json", identity(suite, machine, args.inputs,
                          {"mode": args.mode, "schedule": rows}))
    write(out / "schedule.json", rows)
    samples = []
    for unit in rows:
        path = out / "units" / f"r{unit['round']}-{unit['case']}-{unit['candidate']}"
        if (path / "unit.json").exists():
            sample = read(path / "unit.json")
            if sample.get("fingerprint") != record["fingerprint"]:
                raise ValueError("unit fingerprint mismatch")
        else:
            print(f"{unit['pair']} {unit['candidate']} ({args.mode})", flush=True)
            sample = run_unit(unit, suite, machine, args.inputs, path, args.mode, record["fingerprint"])
            sample["fingerprint"] = record["fingerprint"]
            write(path / "unit.json", sample)
        samples.append(sample)
    report = summarize(samples, suite, args.mode)
    report["fingerprint"] = record["fingerprint"]
    write(out / "summary.json", report)
    report_text(out, report)
    if any(s["status"] != "ok" for s in samples):
        return 1
    return 0


def report_text(out, report):
    lines = ["# Qwen3.8 benchmark", "", f"Mode: {report['mode']}; deployment approval: false.", "",
             "Client receipt timing. p95 with fewer than 100 samples is unstable.", "",
             "| Case | Candidate | Turn | n | TTFT median s | TPOT median s | failed | missing |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        lines.append(f"| {row['case']} | {row['candidate']} | {row['turn']} | {row['metrics']['ttft_s']['n']} | {row['metrics']['ttft_s']['median']} | {row['metrics']['tpot_s']['median']} | {row['failed_units']} | {row['missing_units']} |")
    (Path(out) / "report.md").write_text("\n".join(lines) + "\n")


def summarize_run(args):
    record = read(Path(args.results) / "run.json")
    samples = [read(p) for p in sorted(Path(args.results).glob("units/*/unit.json"))]
    if any(s.get("fingerprint") != record["fingerprint"] for s in samples):
        raise ValueError("mixed result fingerprints")
    suite = record["identity"]["suite"]
    report = summarize(samples, suite, record["identity"]["extra"]["mode"])
    report["fingerprint"] = record["fingerprint"]
    write(Path(args.results) / "summary.json", report)
    report_text(args.results, report)


def seal_trace(args):
    suite, machine = read(args.suite), read(args.machine)
    unit = read(args.source_unit)
    if unit["mode"] != "trace" or unit["status"] != "ok":
        raise ValueError("trace must come from a successful separate collection")
    raw = Path(args.source_unit).parent / "routes.raw.jsonl"
    offset, rows = 0, []
    last_count = None
    for i, line in enumerate(raw.read_text().splitlines()):
        event = json.loads(line)
        if event["layer"] != i % suite["num_layers"]:
            raise ValueError("raw trace incomplete/interleaved; collect single-request text")
        if event["layer"] == 0 and i:
            offset += last_count
        event["token"] = offset
        rows.append(event)
        last_count = len(event["routes"])
    out = Path(args.output)
    if out.exists():
        raise ValueError("refusing to overwrite sealed trace")
    out.write_text("".join(json.dumps(e) + "\n" for e in rows))
    meta = {"kind": "real", "events": len(rows), "trace_sha256": file_hash(out),
            "manifest_sha256": file_hash(machine["manifest"]), "model_revision": machine["model_revision"],
            "collection_unit_sha256": file_hash(args.source_unit),
            "collection_fingerprint": unit["fingerprint"],
            "boundaries": "ordered layer sweeps, token row offsets; source unit identifies request order but no exact route/request join",
            "truncated": len(rows) >= suite["trace_limit"],
            "collection_mode": "profiling only; route capture synchronizes and writes CPU JSONL"}
    validate_trace(out, meta, suite, file_hash(machine["manifest"]))
    write(str(out) + ".meta.json", meta)


def synthetic_trace(args):
    suite, machine = read(args.suite), read(args.machine)
    if args.tokens < 1:
        raise ValueError("positive synthetic token count required")
    rng = random.Random(suite["seed"])
    rows = [{"layer": layer, "token": token, "routes": [rng.sample(range(suite["num_experts"]), suite["top_k"])]}
            for token in range(args.tokens) for layer in range(suite["num_layers"])]
    path = Path(args.output)
    if path.exists():
        raise ValueError("refusing to overwrite trace")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write(str(path) + ".meta.json", {"kind": "synthetic", "events": len(rows), "seed": suite["seed"],
          "trace_sha256": file_hash(path), "manifest_sha256": file_hash(machine["manifest"]),
          "model_revision": machine["model_revision"], "boundaries": "one decode row per layer per token"})


def validate_replay_result(value, mode, candidate, trace_hash, warmup, events, version):
    if not isinstance(value, dict):
        raise ValueError("replay result missing")
    expected = {"status": "ok", "mode": mode, "candidate": candidate,
                "trace_sha256": trace_hash, "warmup_events": warmup,
                "measured_event_range": [0, events], "version": version,
                "effective_child_env": subprocess_environment()}
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"replay result {key} mismatch")
    for key in ("before", "after", "counter_delta", "effective_runtime", "correctness"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"replay result missing {key}")
    if value["effective_runtime"].get("stats_log_interval") != 0:
        raise ValueError("replay periodic logging must be disabled")
    import math
    wall = value.get("wall_s")
    if mode == "benchmark":
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("replay benchmark wall_s invalid")
    elif wall is not None:
        raise ValueError("diagnostic replay wall_s must be null")
    if mode == "correctness" and value["correctness"].get("status") != "passed":
        raise ValueError("replay correctness not passed")


def validate_replay_unit(unit, fingerprint, round_id, candidate, order, mode, trace_hash, warmup, events, version):
    expected = {"fingerprint": fingerprint, "round": round_id, "candidate": candidate,
                "order": order, "pair": f"replay:{round_id}", "mode": mode,
                "trace_sha256": trace_hash, "warmup_events": warmup}
    for key, wanted in expected.items():
        if unit.get(key) != wanted:
            raise ValueError(f"replay unit {key} mismatch")
    if unit.get("status") not in ("ok", "failed"):
        raise ValueError("replay unit status invalid")
    if unit["status"] == "ok":
        validate_replay_result(unit.get("result"), mode, candidate, trace_hash, warmup, events, version)
    elif not unit.get("error"):
        raise ValueError("failed replay unit lacks error")


def check_replay_l2(result, candidate):
    if result["effective_runtime"].get("decode_submit_experts", 1) != candidate["loader"].get("decode_submit_experts", 1):
        raise ValueError("replay effective decode submission mismatch")
    for field in ("packed_decode", "packed_prefill"):
        if result["effective_runtime"].get(field, "direct") != candidate["loader"].get(field, "direct"):
            raise ValueError("replay effective packed configuration mismatch")
    enabled = any(candidate["loader"].get(field, "direct") != "direct" for field in ("packed_decode", "packed_prefill"))
    if result["effective_runtime"].get("packed_attached", False) != enabled:
        raise ValueError("replay packed attachment mismatch")
    mode = candidate["loader"].get("l2_mode", "off")
    effective = result["effective_runtime"]
    if effective.get("l2_mode", "off") != mode or effective.get("l2_attached", False) != (mode != "off"):
        raise ValueError("replay effective L2 configuration mismatch")
    if mode != "off" and not result.get("before", {}).get("l2"):
        raise ValueError("replay L2 boundary state missing")


def replay_matrix(args):
    suite, machine = read(args.suite), read(args.machine)
    validate(suite, machine, args.inputs)
    mode = "correctness" if args.correctness else "profile" if args.profile else "benchmark"
    meta = read(args.trace_meta)
    events = validate_trace(args.trace, meta, suite, file_hash(machine["manifest"]))
    if args.warmup_events < 0 or args.warmup_events > len(events) or args.warmup_events % suite["num_layers"]:
        raise ValueError("warmup must be a complete prefix of layer sweeps")
    out = Path(args.results)
    if out.exists() and any(p.name != ".lock" for p in out.iterdir()) and not args.resume:
        raise ValueError("results exist; use --resume")
    record = resume_guard(out / "run.json", identity(suite, machine, args.inputs,
           {"mode": "replay-" + mode, "effective_child_env": subprocess_environment(),
            "trace_sha256": file_hash(args.trace), "meta": meta, "warmup_events": args.warmup_events}))
    rng = random.Random(suite["seed"])
    rows, failures = [], 0
    for r in range(suite["rounds"] if not args.correctness else 1):
        candidates = list(suite["candidates"])
        rng.shuffle(candidates)
        for order, candidate in enumerate(candidates):
            target = out / f"r{r}-{candidate['id']}"
            target.mkdir(parents=True, exist_ok=True)
            if (target / "unit.json").exists():
                result = read(target / "unit.json")
                validate_replay_unit(result, record["fingerprint"], r, candidate["id"], order,
                                     mode, meta["trace_sha256"], args.warmup_events, len(events), suite["version"])
                if result["status"] == "ok":
                    check_replay_l2(result["result"], candidate)
            else:
                # No committed unit means incomplete work, never trust an orphan report.
                orphan = target / "replay.json"
                if orphan.exists():
                    index = 0
                    while (target / f"replay.orphan-{index}.json").exists():
                        index += 1
                    orphan.rename(target / f"replay.orphan-{index}.json")
                argv = [sys.executable, "-m", "tools.qwen38_cache_tests.benchmark", "replay-worker",
                        "--suite", args.suite, "--machine", args.machine, "--trace", args.trace,
                        "--trace-meta", args.trace_meta, "--candidate", candidate["id"],
                        "--warmup-events", str(args.warmup_events), "--output", str(target / "replay.json")]
                if args.correctness:
                    argv.append("--correctness")
                if args.profile:
                    argv.append("--profile")
                process = None
                error = None
                with (target / "worker.log").open("w") as log:
                    try:
                        process = subprocess.Popen(argv, cwd=ROOT, env=subprocess_environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        process.wait(timeout=suite["replay_timeout_s"])
                        if process.returncode:
                            error = f"worker exit {process.returncode}"
                    except subprocess.TimeoutExpired:
                        error = "replay total deadline exceeded"
                    except KeyboardInterrupt:
                        error = "interrupted"
                        raise
                    finally:
                        stop_owned(process)
                worker_result = None
                try:
                    worker_result = read(target / "replay.json")
                    if error is None:
                        validate_replay_result(worker_result, mode, candidate["id"], meta["trace_sha256"],
                                               args.warmup_events, len(events), suite["version"])
                        check_replay_l2(worker_result, candidate)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    error = error or f"invalid worker result: {exc}"
                result = {"mode": mode, "trace_sha256": meta["trace_sha256"], "warmup_events": args.warmup_events,
                          "round": r, "candidate": candidate["id"], "order": order, "pair": f"replay:{r}",
                          "fingerprint": record["fingerprint"], "command": argv,
                          "status": "failed" if error else "ok", "error": error,
                          "result": worker_result}
                write(target / "unit.json", result)
            rows.append(result)
            failures += result["status"] != "ok"
    from .core import describe
    paired = []
    for c in suite["candidates"]:
        if c["id"] == suite["baseline"] or mode != "benchmark":
            continue
        ratios = []
        for r in range(suite["rounds"]):
            a = next((u for u in rows if u["round"] == r and u["candidate"] == c["id"] and u["status"] == "ok"), None)
            b = next((u for u in rows if u["round"] == r and u["candidate"] == suite["baseline"] and u["status"] == "ok"), None)
            if a and b and a["result"]["wall_s"] is not None and b["result"]["wall_s"]:
                ratios.append({"round": r, "wall_ratio": a["result"]["wall_s"] / b["result"]["wall_s"]})
        paired.append({"candidate": c["id"], "pairs": ratios, "stats": describe([v["wall_ratio"] for v in ratios])})
    summary = {"paired": paired, "mode": record["identity"]["extra"]["mode"], "scope": "transport-only", "trace_kind": meta["kind"],
               "samples": rows, "wall_s": {c["id"]: describe([u["result"]["wall_s"] for u in rows
                  if u["candidate"] == c["id"] and u["status"] == "ok" and u["result"]["wall_s"] is not None]) for c in suite["candidates"]}}
    write(out / "summary.json", summary)
    return 1 if failures else 0


def profile_candidate(args):
    record = read(Path(args.results) / "run.json")
    summary = read(Path(args.results) / "summary.json")
    suite = record["identity"]["suite"]
    candidate = next(c for c in suite["candidates"] if c["id"] == args.candidate)
    # Stage A creates diagnostic profiles only. Promotion awaits GPU baseline review.
    profile = {"version": suite["version"], "status": "candidate", "approved": False,
               "candidate": candidate, "baseline": next(c for c in suite["candidates"] if c["id"] == suite["baseline"]),
               "fingerprint": record["fingerprint"], "identity": record["identity"],
               "summary_sha256": file_hash(Path(args.results) / "summary.json"),
               "results": str(Path(args.results).resolve()),
               "reason": "diagnostic export; GPU correctness, independent holdout and baseline review required",
               "failed_samples": len(summary.get("failed_samples", []))}
    write(args.output, profile)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "campaign":
        from .campaign_cli import main as campaign_main
        try:
            return campaign_main(sys.argv[2:])
        except (Exception, KeyboardInterrupt) as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    a = subs.add_parser("prepare-inputs")
    a.add_argument("--suite", required=True); a.add_argument("--checkpoint", required=True); a.add_argument("--output", required=True)
    a = subs.add_parser("freeze-answer")
    for flag in ("suite", "inputs", "source-unit", "output-suite"):
        a.add_argument("--" + flag, required=True)
    for name in ("validate", "run", "replay"):
        a = subs.add_parser(name)
        for flag in ("suite", "machine", "inputs"):
            a.add_argument("--" + flag, required=True)
        if name == "run":
            a.add_argument("--results", required=True); a.add_argument("--resume", action="store_true")
            a.add_argument("--case"); a.add_argument("--rounds", type=int)
            a.add_argument("--mode", choices=("benchmark", "profile", "trace"), default="benchmark")
        if name == "replay":
            a.add_argument("--results", required=True); a.add_argument("--resume", action="store_true")
            for flag in ("trace", "trace-meta"):
                a.add_argument("--" + flag, required=True)
            a.add_argument("--warmup-events", type=int, default=0,
                       help="repeat first N trace events as warmup, then measure the entire trace [0,end)")
            modes = a.add_mutually_exclusive_group()
            modes.add_argument("--correctness", action="store_true")
            modes.add_argument("--profile", action="store_true", help="diagnostic per-layer events; no benchmark speed ratios")
    for name in ("seal-trace", "synthetic-trace"):
        a = subs.add_parser(name)
        for flag in ("suite", "machine", "output"):
            a.add_argument("--" + flag, required=True)
        if name == "seal-trace":
            a.add_argument("--source-unit", required=True)
        else:
            a.add_argument("--tokens", type=int, default=16)
    a = subs.add_parser("replay-worker")
    for flag in ("suite", "machine", "trace", "trace-meta", "candidate", "output"):
        a.add_argument("--" + flag, required=True)
    a.add_argument("--warmup-events", type=int, default=0)
    modes = a.add_mutually_exclusive_group()
    modes.add_argument("--correctness", action="store_true")
    modes.add_argument("--profile", action="store_true")
    a = subs.add_parser("summarize"); a.add_argument("--results", required=True)
    a = subs.add_parser("candidate-profile")
    for flag in ("results", "candidate", "output"):
        a.add_argument("--" + flag, required=True)
    args = p.parse_args()
    import signal
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    lock = None
    try:
        if args.action in ("run", "replay"):
            import fcntl
            Path(args.results).mkdir(parents=True, exist_ok=True)
            lock = open(Path(args.results) / ".lock", "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("results directory already in use")
        if args.action == "prepare-inputs": prepare(args)
        elif args.action == "freeze-answer": freeze_answer(args)
        elif args.action == "validate":
            print(json.dumps(validate(read(args.suite), read(args.machine), args.inputs), indent=2))
        elif args.action == "run": return full_run(args)
        elif args.action == "replay": return replay_matrix(args)
        elif args.action == "seal-trace": seal_trace(args)
        elif args.action == "synthetic-trace": synthetic_trace(args)
        elif args.action == "summarize": summarize_run(args)
        elif args.action == "candidate-profile": profile_candidate(args)
        elif args.action == "replay-worker":
            from .replay import run
            run(args)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
