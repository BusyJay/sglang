"""Staged GPU tuning and fail-closed machine profiles; CPU fixtures stay synthetic."""
from __future__ import annotations
import copy
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from .core import ROOT, digest, file_hash, read, write, validate, schedule, check_effective, subprocess_environment

GATES = {"screen_rounds": 1, "tune_rounds": 3, "holdout_rounds": 5,
         "shortlist": 3, "minimum_gain": 0.03, "case_regression": 0.03,
         "ttft_regression": 0.03, "duration_regression": 0.03,
         "bootstrap_samples": 2000, "confidence": 0.95}


def safe_baseline(candidate):
    l = candidate["loader"]
    return (l.get("l2_mode", "off") == "off" and l.get("packed_decode", "direct") == "direct"
            and l.get("packed_prefill", "direct") == "direct" and l.get("decode_submit_experts", 1) == 1
            and candidate["expected_effective"].get("disable_overlap_schedule") is False)


def matrix(source):
    """Canonical unique settings; packed submission groups intentionally redundant."""
    base = copy.deepcopy(source)
    prototype = next(c for c in source["candidates"] if c["id"] == source["baseline"])
    if not safe_baseline(prototype):
        raise ValueError("deployment baseline must be overlap-on L2-off/direct/expert")
    candidates, seen = [], set()
    for mode in ("off", "fixed", "dynamic"):
        for overlap in ((False, True) if mode == "off" else (True,)):
            for packed_decode in ("direct", "packed"):
                for packed_prefill in ("direct", "packed"):
                    # Packed decode remains single-expert; don't screen 1/2/0
                    # duplicates that cannot merge DMA on the packed path.
                    for group in ((1, 2, 0) if packed_decode == "direct" else (2,)):
                        c = copy.deepcopy(prototype)
                        l = c["loader"]
                        for k in list(l):
                            if k.startswith("l2_") or k.startswith("packed_") or k == "decode_submit_experts":
                                del l[k]
                        l.update(l2_mode=mode, packed_decode=packed_decode, packed_prefill=packed_prefill, decode_submit_experts=group)
                        if mode == "fixed":
                            l["l2_fixed_seed"] = copy.deepcopy(source["replay_l2"]["seed"])
                        if mode == "dynamic":
                            l.update(l2_min_score=2.0, l2_margin=0.1, l2_demotion_budget=2, l2_prefill_budget=8)
                        c["server_args"] = [v for v in c["server_args"] if v.split("=", 1)[0] != "--disable-overlap-schedule"]
                        if overlap:
                            c["server_args"].append("--disable-overlap-schedule")
                        c["expected_effective"]["disable_overlap_schedule"] = overlap
                        key = digest([l, c["server_args"], c["expected_effective"]])
                        if key in seen:
                            continue
                        seen.add(key)
                        baseline = mode == "off" and not overlap and packed_decode == packed_prefill == "direct" and group == 1
                        c["id"] = source["baseline"] if baseline else f"l2-{mode}-overlap-{'off' if overlap else 'on'}-decode-{packed_decode}-prefill-{packed_prefill}-group-{group}"
                        candidates.append(c)
    base["candidates"] = candidates
    return base


def validate_gates(gates):
    if set(gates) != set(GATES):
        raise ValueError("unknown/missing selection gate")
    for k in ("screen_rounds", "tune_rounds", "holdout_rounds", "shortlist", "bootstrap_samples"):
        if type(gates[k]) is not int or gates[k] < 1:
            raise ValueError("positive integer gates required")
    if gates["holdout_rounds"] < 5 or gates["bootstrap_samples"] < 1000:
        raise ValueError("holdout >=5 paired rounds and bootstrap >=1000 required")
    for k in ("minimum_gain", "case_regression", "ttft_regression", "duration_regression"):
        if type(gates[k]) not in (float, int) or not 0 <= gates[k] < 1:
            raise ValueError("invalid ratio gate")
    if gates["confidence"] != 0.95:
        raise ValueError("current selector implements 95% paired bootstrap")


def new_plan(suite, gates=None):
    gates = copy.deepcopy(gates or GATES)
    validate_gates(gates)
    if not safe_baseline(next(c for c in suite["candidates"] if c["id"] == suite["baseline"])):
        raise ValueError("unsafe baseline")
    if not any(c["split"] == "tune" for c in suite["cases"]) or not any(c["split"] == "holdout" for c in suite["cases"]):
        raise ValueError("independent tune and holdout cases required")
    tune_files = {t["tokens"] for c in suite["cases"] if c["split"] == "tune" for t in c["turns"]}
    holdout_files = {t["tokens"] for c in suite["cases"] if c["split"] == "holdout" for t in c["turns"]}
    if tune_files & holdout_files:
        raise ValueError("tune and holdout measured token files must be independent")
    value = {"schema": "qwen38-campaign-v2", "suite": suite, "gates": gates,
             "scope": "same-tree ablations, real GPU performance pending", "status": "prepared"}
    value["fingerprint"] = digest(value)
    return value


def check_plan(plan):
    if plan.get("schema") != "qwen38-campaign-v2":
        raise ValueError("campaign schema mismatch; regenerate v2 evidence")
    raw = {k: v for k, v in plan.items() if k != "fingerprint"}
    if plan.get("fingerprint") != digest(raw):
        raise ValueError("campaign plan fingerprint mismatch")
    validate_gates(plan["gates"])


def binding(identity):
    """Same code/machine/model/all input bytes across staged subsets."""
    return {k: identity[k] for k in ("code", "environment", "machine_config", "manifest_sha256", "payload_sha256",
                                    "model_metadata", "checkpoint_files", "inputs", "input_artifacts")}


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def check_artifact(value):
    if file_hash(value["path"]) != value["sha256"]:
        raise ValueError("evidence artifact hash mismatch")
    return read(value["path"])


def stage_suite(plan, stage, previous=None):
    check_plan(plan)
    if stage not in ("baseline", "screen", "tune", "holdout"):
        raise ValueError("unknown stage")
    suite = copy.deepcopy(plan["suite"])
    split = "holdout" if stage == "holdout" else "tune"
    if stage != "baseline":
        suite["cases"] = [c for c in suite["cases"] if c["split"] == split]
    if stage == "screen":
        suite["cases"] = [min(suite["cases"], key=lambda c: sum(t["input_tokens"] for t in c["turns"]))]
    if stage == "baseline":
        suite["candidates"] = [c for c in suite["candidates"] if c["id"] == suite["baseline"]]
    if previous is None and stage not in ("screen", "baseline"):
        raise ValueError("next stage requires previous selection")
    if previous:
        if previous["plan_fingerprint"] != plan["fingerprint"] or previous["stage"] != ("screen" if stage == "tune" else "tune"):
            raise ValueError("selection stage/plan mismatch")
        check_artifact(previous["results_record"])
        suite["candidates"] = [c for c in suite["candidates"] if c["id"] == suite["baseline"] or c["id"] in previous["shortlist"]]
    suite["rounds"] = plan["gates"]["holdout_rounds" if stage == "baseline" else stage + "_rounds"]
    suite["campaign"] = {"plan_fingerprint": plan["fingerprint"], "stage": stage,
                         "previous_selection_sha256": digest(previous) if previous else None}
    return suite


def load_run(path, plan, stage, previous=None):
    """Validate raw units, actual launch config, requests, pair schedule and hashes."""
    path = Path(path).resolve()
    record = read(path / "run.json")
    identity = record["identity"]
    if record["fingerprint"] != digest(identity):
        raise ValueError("run fingerprint mismatch")
    expected = stage_suite(plan, stage, previous)
    # Locator and byte hash are part of the run/stage/profile binding.
    from .core import token_file
    locators = identity.get("input_artifacts", {})
    if set(locators) != set(identity["inputs"]):
        raise ValueError("missing frozen input artifacts")
    for name, item in locators.items():
        if item.get("sha256") != identity["inputs"][name] or not Path(item["path"]).is_absolute():
            raise ValueError("frozen input artifact binding mismatch")
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("frozen input artifact hash mismatch")
    frozen_inputs = {t["tokens"]: token_file(locators[t["tokens"]]["path"])
                     for c in plan["suite"]["cases"] for t in c["turns"] + c["warmup_requests"]}
    tune_hashes = {identity["inputs"][t["tokens"]] for c in plan["suite"]["cases"] if c["split"] == "tune" for t in c["turns"]}
    holdout_hashes = {identity["inputs"][t["tokens"]] for c in plan["suite"]["cases"] if c["split"] == "holdout" for t in c["turns"]}
    if tune_hashes & holdout_hashes:
        raise ValueError("holdout input bytes duplicate tune inputs")
    if identity["suite"] != expected or identity["extra"]["mode"] != "benchmark":
        raise ValueError("foreign suite/stage or profiling results")
    rows = schedule(expected)
    if identity["extra"]["schedule"] != rows or read(path / "schedule.json") != rows:
        raise ValueError("foreign/missing schedule")
    units, artifacts = [], [artifact(path / "run.json"), artifact(path / "schedule.json")]
    expected_paths = set()
    for row in rows:
        directory = path / "units" / f"r{row['round']}-{row['case']}-{row['candidate']}"
        target = directory / "unit.json"
        expected_paths.add(target)
        if not target.exists():
            raise ValueError("missing unit: " + str(target))
        unit = read(target)
        if any(unit.get(k) != v for k, v in row.items()) or unit.get("fingerprint") != record["fingerprint"] or unit.get("mode") != "benchmark":
            raise ValueError("foreign unit fingerprint/pair")
        artifacts.append(artifact(target))
        c = next(c for c in expected["candidates"] if c["id"] == row["candidate"])
        case = next(c for c in expected["cases"] if c["id"] == row["case"])
        if unit.get("status") not in ("ok", "failed") or (unit["status"] == "failed" and not unit.get("error")):
            raise ValueError("invalid unit failure status")
        if unit["status"] == "ok":
            imports = unit.get("actual_imports", {})
            if set(imports) != {"sglang", "runtime"} or any(not Path(v).resolve().is_relative_to(ROOT / "python/sglang") for v in imports.values()):
                raise ValueError("actual runtime imports missing/foreign")
            probe = read(directory / "import_probe.json")
            if probe.get("returncode") != 0:
                raise ValueError("actual import probe failed")
            artifacts.append(artifact(directory / "import_probe.json"))
            info = read(directory / "effective_server_info.json")
            check_effective(info, c, {**c["loader"], "manifest_path": identity["machine_config"]["manifest"]})
            if info.get("model_path") != identity["machine_config"]["checkpoint"] or info.get("revision") != identity["machine_config"]["model_revision"]:
                raise ValueError("effective model mismatch")
            if unit.get("enforced_environment") != identity["environment"]["effective_child_env"]:
                raise ValueError("unit environment mismatch")
            artifacts.extend([artifact(directory / "effective_server_info.json"), artifact(directory / "server.log")])
            history = []
            for kind, turns in (("warmup", case["warmup_requests"]), ("request", case["turns"])):
                samples = unit.get("warmup" if kind == "warmup" else "requests", [])
                if len(samples) != len(turns):
                    raise ValueError("missing request/warmup evidence")
                for i, (turn, request) in enumerate(zip(turns, samples)):
                    name = turn["tokens"]
                    frozen = frozen_inputs[name]
                    if len(frozen) != turn["input_tokens"]:
                        raise ValueError("frozen input length mismatch")
                    expected_ids = ([] if kind == "warmup" else history) + frozen
                    raw_path = directory / f"{kind}-{i}.json"
                    raw = read(raw_path)
                    if raw != request or raw.get("status") != "ok" or raw.get("errors"):
                        raise ValueError("request/warmup raw evidence mismatch/error")
                    ids = raw["output_ids"]
                    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids) or len(ids) != raw["output_tokens"] or not ids or raw["output_sha256"] != digest(ids):
                        raise ValueError("request output count/hash mismatch")
                    finish = raw.get("finish_reason", {})
                    if finish.get("type") not in ("stop", "length") or (turn["length_policy"] == "fixed" and len(ids) != turn["sampling"]["max_new_tokens"]):
                        raise ValueError("incomplete request length/finish")
                    if not isinstance(raw.get("input_ids"), list) or any(type(i) is not int for i in raw["input_ids"]) or raw["input_ids"] != expected_ids or raw.get("input_tokens") != len(expected_ids):
                        raise ValueError("input tokens differ from frozen file/session history")
                    from .core import StreamMeasurement
                    measurement = StreamMeasurement()
                    for event in raw.get("events", []):
                        measurement.feed(event["raw"], event["seconds"])
                    reconstructed = measurement.finish(raw["duration_s"], turn, raw["input_tokens"])
                    if reconstructed["status"] != "ok" or (any(reconstructed[k] != raw[k] for k in ("output_ids", "output_tokens", "finish_reason")) or any(reconstructed[k] is None or not math.isclose(reconstructed[k], raw[k], rel_tol=1e-12, abs_tol=1e-12) for k in ("tpot_s", "ttft_s"))):
                        raise ValueError("timing/output not reproducible from raw stream events")
                    for metric in ("tpot_s", "ttft_s", "duration_s"):
                        if type(raw[metric]) not in (int, float) or not math.isfinite(raw[metric]) or raw[metric] <= 0:
                            raise ValueError("missing/invalid request metric")
                    artifacts.append(artifact(raw_path))
                    if kind == "request":
                        answer = ids.copy()
                        while answer and answer[-1] in expected.get("session_strip_tail_ids", []):
                            answer.pop()
                        history = expected_ids + answer
        units.append(unit)
    if set(path.glob("units/*/unit.json")) != expected_paths:
        raise ValueError("unexpected units in results")
    return record, units, artifacts


def paired_scores(units, suite, gates):
    """Paired workload geometric mean; bootstrap independent process rounds."""
    index = {(u["round"], u["case"], u["candidate"]): u for u in units}
    baseline = suite["baseline"]
    scores = []
    for candidate in suite["candidates"]:
        if candidate["id"] == baseline:
            continue
        ratios, case_rows, errors = [], [], []
        for case in suite["cases"]:
            per_case = []
            for turn in range(len(case["turns"])):
                values = {k: [] for k in ("tpot_s", "ttft_s", "duration_s")}
                for r in range(suite["rounds"]):
                    a, b = index.get((r, case["id"], candidate["id"])), index.get((r, case["id"], baseline))
                    if not a or not b or a["status"] != "ok" or b["status"] != "ok":
                        errors.append(f"failed/missing pair {case['id']}/{turn}/{r}")
                        continue
                    for k in values:
                        x, y = a["requests"][turn][k], b["requests"][turn][k]
                        if x is None or y is None or x <= 0 or y <= 0:
                            errors.append("invalid metric")
                            continue
                        values[k].append(x / y)
                per_case.append(values["tpot_s"])
                row = {"case": case["id"], "turn": turn, "ratios": values,
                       "medians": {k: statistics.median(v) if v else None for k, v in values.items()}}
                case_rows.append(row)
                for k, limit in (("tpot_s", gates["case_regression"]), ("ttft_s", gates["ttft_regression"]), ("duration_s", gates["duration_regression"])):
                    if len(values[k]) != suite["rounds"] or row["medians"][k] > 1 + limit:
                        errors.append(f"case regression/incomplete {case['id']}/{turn}/{k}")
            ratios.append((case["weight"], per_case))
        rounds = []
        if not errors:
            for r in range(suite["rounds"]):
                rounds.append(math.exp(sum(weight * statistics.mean(math.log(v[r]) for v in turns)
                                          for weight, turns in ratios) / sum(w for w, _ in ratios)))
        ratio = math.exp(statistics.mean(map(math.log, rounds))) if rounds else None
        rng = random.Random(suite["seed"])
        boots = sorted(math.exp(statistics.mean(map(math.log, rng.choices(rounds, k=len(rounds)))))
                       for _ in range(gates["bootstrap_samples"])) if rounds else []
        ci = [boots[int(len(boots) * 0.025)], boots[min(len(boots)-1, int(len(boots)*0.975))]] if len(rounds) >= 5 else None
        scores.append({"candidate": candidate["id"], "ratio": ratio, "paired_round_ratios": rounds,
                       "bootstrap_95": ci, "case_metrics": case_rows, "errors": errors,
                       "passes": not errors and ratio <= 1 - gates["minimum_gain"] and ci is not None and ci[1] <= 1 - gates["minimum_gain"]})
    return scores


def select_stage(plan, stage, results, previous=None):
    if stage not in ("screen", "tune", "holdout"):
        raise ValueError("selection requires screen/tune/holdout stage")
    record, units, artifacts = load_run(results, plan, stage, previous)
    if previous and binding(record["identity"]) != previous["binding"]:
        raise ValueError("foreign code/machine/model/input fingerprint between stages")
    scores = paired_scores(units, record["identity"]["suite"], plan["gates"])
    eligible = sorted((s for s in scores if not s["errors"] and s["ratio"] is not None and s["ratio"] < 1), key=lambda s: s["ratio"])
    selected = [s["candidate"] for s in eligible[:plan["gates"]["shortlist"]]] if stage != "holdout" else [s["candidate"] for s in eligible if s["passes"]][:1]
    return {"schema": "qwen38-selection-v2", "status": "candidate", "stage": stage,
            "plan_fingerprint": plan["fingerprint"], "shortlist": selected, "scores": scores,
            "binding": binding(record["identity"]), "results_record": artifact(Path(results) / "run.json"),
            "artifacts": artifacts, "failed_units": [{k:u.get(k) for k in ("case","candidate","round","status","error")} for u in units if u["status"] != "ok"], "previous": previous,
            "reason": "no qualifying gain: retain baseline" if not selected else "GPU correctness and independent evidence gate still required"}


def gpu_available(identity):
    gpu = identity["environment"]["gpu"]
    return gpu.get("returncode") == 0 and bool(gpu.get("stdout", "").strip())


def verify_selection(selection, plan):
    check_plan(plan)
    previous = selection.get("previous")
    if previous:
        verify_selection(previous, plan)
    record_path = check_artifact(selection["results_record"])
    path = Path(selection["results_record"]["path"]).parent
    rebuilt = select_stage(plan, selection["stage"], path, previous)
    if rebuilt != selection:
        raise ValueError("selection modified or raw evidence changed")
    for item in selection["artifacts"]:
        # server.log is plain text, not JSON
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("selection artifact changed")
    return record_path


def seal_correctness(plan, selection, gpu_small, replay_results, semantic):
    """Bind actual GPU test/report artifacts and reviewed model output IDs."""
    record = verify_selection(selection, plan)
    if not selection.get("previous") or not selection["previous"].get("previous"):
        raise ValueError("complete screen/tune/independent holdout chain required")
    if selection["stage"] != "holdout" or not selection["shortlist"]:
        raise ValueError("qualifying independent holdout selection required")
    if not gpu_available(record["identity"]):
        raise ValueError("real GPU hardware evidence required")
    small = read(gpu_small)
    if (small.get("status") != "passed" or small.get("tests", 0) <= 0 or small.get("skipped") != 0
            or small.get("failures") != 0 or small.get("errors") != 0 or small.get("execution_kind") != "cuda"
            or small.get("capability") != [12, 0] or small.get("code") != selection["binding"]["code"]
            or small.get("environment") != selection["binding"]["environment"]):
        raise ValueError("GPU small-test evidence invalid/foreign; CPU/mock cannot validate")
    replay_path = Path(replay_results)
    replay_record = read(replay_path / "run.json")
    if replay_record.get("fingerprint") != digest(replay_record["identity"]):
        raise ValueError("replay fingerprint mismatch")
    if binding(replay_record["identity"]) != selection["binding"] or replay_record["identity"]["extra"].get("mode") != "replay-correctness":
        raise ValueError("foreign correctness replay binding/mode")
    if replay_record["identity"]["extra"]["meta"].get("kind") != "real":
        raise ValueError("synthetic trace cannot validate machine defaults")
    from .__main__ import validate_replay_unit, check_replay_l2
    from .replay import validate_trace
    expected = replay_record["identity"]["suite"]
    final_candidates = {c["id"]: c for c in record["identity"]["suite"]["candidates"]}
    candidates = {c["id"]: c for c in expected["candidates"]}
    needed = [expected["baseline"], selection["shortlist"][0]]
    artifacts = [artifact(gpu_small), artifact(replay_path / "run.json")]
    seen = set()
    for path in sorted(replay_path.glob("r0-*/unit.json")):
        u = read(path)
        validate_replay_unit(u, replay_record["fingerprint"], 0, u["candidate"], u["order"], "correctness",
                             replay_record["identity"]["extra"]["trace_sha256"], replay_record["identity"]["extra"]["warmup_events"],
                             replay_record["identity"]["extra"]["meta"]["events"], expected["version"])
        if u["status"] != "ok" or u["candidate"] in seen or u["candidate"] not in candidates:
            raise ValueError("failed/duplicate/foreign correctness unit")
        if read(path.parent / "replay.json") != u["result"]:
            raise ValueError("raw replay result differs from unit")
        artifacts.append(artifact(path.parent / "replay.json"))
        check_replay_l2(u["result"], candidates[u["candidate"]])
        if candidates[u["candidate"]] != final_candidates.get(u["candidate"]):
            raise ValueError("correctness candidate config differs from holdout")
        argv = u["command"]
        trace_path = argv[argv.index("--trace") + 1]
        meta_path = argv[argv.index("--trace-meta") + 1]
        meta = read(meta_path)
        if meta != replay_record["identity"]["extra"]["meta"]:
            raise ValueError("raw trace metadata differs from correctness run")
        validate_trace(trace_path, meta, expected, selection["binding"]["manifest_sha256"])
        artifacts.extend([artifact(trace_path), artifact(meta_path), artifact(path.parent / "worker.log")])
        evidence = u["result"]["correctness"]["evidence"]
        if len(evidence) != u["result"]["trace_meta"]["events"] or not all(e["matches"] and all(e["matches"].values()) for e in evidence):
            raise ValueError("missing/failed routed payload evidence")
        artifacts.append(artifact(path)); seen.add(u["candidate"])
    if not set(needed) <= seen:
        raise ValueError("baseline/candidate correctness missing")
    review = read(semantic)
    required_outputs = {a["path"]: a["sha256"] for a in selection["artifacts"] if Path(a["path"]).name.startswith("request-")}
    if (review.get("selection_sha256") != digest(selection) or review.get("reviewed_request_artifacts") != required_outputs
            or review.get("status") != "passed" or not review.get("reviewer") or not review.get("notes")):
        raise ValueError("semantic/output/boundary review missing or unbound")
    artifacts.append(artifact(semantic))
    return {"schema": "qwen38-gpu-correctness-v2", "execution_kind": "cuda", "status": "passed",
            "selection_sha256": digest(selection), "binding": selection["binding"], "artifacts": artifacts,
            "gpu_small": artifact(gpu_small), "replay_results": str(replay_path.resolve()), "semantic": artifact(semantic)}


def make_profile(plan, selection=None, correctness=None):
    check_plan(plan)
    baseline = next(c for c in plan["suite"]["candidates"] if c["id"] == plan["suite"]["baseline"])
    profile = {"schema": "qwen38-machine-profile-v2", "status": "baseline_pending", "config": baseline,
               "fallback": baseline, "plan": plan, "selection": selection, "correctness": correctness,
               "binding": None, "reason": "no real GPU evidence: baseline retained; not validated"}
    if selection:
        verify_selection(selection, plan)
        profile["binding"] = selection["binding"]
        if selection["stage"] != "holdout":
            profile["status"] = "candidate"
            profile["reason"] = "independent holdout not completed"
        elif selection["shortlist"]:
            profile["status"] = "candidate"
            profile["config"] = next(c for c in plan["suite"]["candidates"] if c["id"] == selection["shortlist"][0])
            profile["reason"] = "qualifying holdout, correctness evidence still required"
            if correctness:
                rebuilt = seal_correctness(plan, selection, correctness["gpu_small"]["path"], correctness["replay_results"], correctness["semantic"]["path"])
                if rebuilt != correctness:
                    raise ValueError("correctness seal changed")
                profile["status"] = "validated"
                profile["reason"] = "real GPU correctness + independent paired holdout gates passed"
        else:
            profile["status"] = "baseline"
            profile["reason"] = "no qualifying gain; baseline retained"
    profile["fingerprint"] = digest(profile)
    return profile


def check_profile(profile, current):
    raw = {k: v for k, v in profile.items() if k != "fingerprint"}
    if digest(raw) != profile.get("fingerprint"):
        raise ValueError("profile fingerprint mismatch")
    rebuilt = make_profile(profile["plan"], profile["selection"], profile["correctness"])
    if rebuilt != profile:
        raise ValueError("profile status/config/evidence inconsistent")
    if current["suite"] != profile["plan"]["suite"]:
        raise ValueError("profile workload/config scope mismatch")
    if profile["binding"] != binding(current):
        raise ValueError("profile code/environment/model/input fingerprint mismatch")
    if profile["status"] != "validated":
        raise ValueError("profile is not validated; use explicit baseline fallback")
    return profile["config"]


def launch_command(profile, current, fallback=False, loader_overrides=None, server_overrides=None):
    """Explicit settings win; an override is always marked unvalidated."""
    if fallback:
        if digest({k: v for k, v in profile.items() if k != "fingerprint"}) != profile.get("fingerprint"):
            raise ValueError("profile fingerprint mismatch")
        check_plan(profile["plan"])
        canonical = next(c for c in profile["plan"]["suite"]["candidates"] if c["id"] == profile["plan"]["suite"]["baseline"])
        if profile["fallback"] != canonical:
            raise ValueError("fallback differs from plan baseline")
        candidate = copy.deepcopy(profile["fallback"])
        if not safe_baseline(candidate):
            raise ValueError("unsafe fallback")
        status = "baseline_fallback_unvalidated"
    else:
        candidate = copy.deepcopy(check_profile(profile, current))
        status = "validated"
    loader_overrides = loader_overrides or {}
    server_overrides = server_overrides or []
    if any(k in loader_overrides for k in ("manifest_path", "route_trace_path")):
        raise ValueError("managed loader field override")
    candidate["loader"].update(loader_overrides)
    machine = current["machine_config"]
    # Merge complete argv groups, preserving one occurrence per key. Explicit
    # overrides win over deployment settings, which win over profile settings.
    from .server_options import parse_options
    merged = parse_options(candidate["server_args"])
    machine_options = parse_options(machine.get("deployment_args", []))
    explicit = parse_options(server_overrides)
    merged.update(machine_options)
    merged.update(explicit)
    assertions = candidate["expected_effective"]
    original_assertions = copy.deepcopy(assertions)
    for _, expected in merged.values():
        assertions.update(expected)
    candidate["server_args"] = [v for argv, _ in merged.values() for v in argv]
    if loader_overrides or server_overrides:
        status = "explicit_override_unvalidated"
    elif any(assertions[k] != v for k, v in original_assertions.items()):
        status = "machine_override_unvalidated"
    effective_overlap = assertions["disable_overlap_schedule"]
    if candidate["loader"].get("l2_mode", "off") != "off" and not effective_overlap:
        raise ValueError("L2 requires overlap off")
    from .core import validate as validate_suite
    suite = copy.deepcopy(profile["plan"]["suite"])
    candidate["expected_effective"]["disable_overlap_schedule"] = effective_overlap
    suite["candidates"], suite["baseline"] = [candidate], candidate["id"]
    validate_suite(suite, machine)
    loader = {**candidate["loader"], "manifest_path": machine["manifest"]}
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", machine["checkpoint"],
               "--revision", machine["model_revision"], "--host", "127.0.0.1", "--port", str(machine["port"]),
               "--model-loader-extra-config", __import__("json").dumps(loader), *candidate["server_args"]]
    return {"status": status, "command": command, "environment": subprocess_environment(),
            "config": candidate, "fingerprint": digest(current)}


def experimental_launch(current, loader_overrides=None, server_overrides=None):
    """Normal user-authorized test launch; no GPU validation/profile required."""
    suite = current["suite"]
    candidate = copy.deepcopy(next(c for c in suite["candidates"] if c["id"] == suite["baseline"]))
    candidate["id"] = "experimental-defaults"
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("qwen38_launch_defaults", ROOT / "python/sglang/srt/layers/moe/qwen38_l2_config.py")
    module = module_from_spec(spec); spec.loader.exec_module(module)
    candidate["loader"].update(module.EXPERIMENTAL_DEFAULTS)
    # Start from explicit experimental settings. Caller may disable individual
    # features; the L2 overlap dependency is coordinated, never silently off.
    overrides = loader_overrides or {}
    candidate["loader"].update(overrides)
    overlap = candidate["loader"]["l2_mode"] != "off"
    candidate["server_args"] = [a for a in candidate["server_args"] if a.split("=",1)[0] != "--disable-overlap-schedule"]
    if overlap:
        candidate["server_args"].append("--disable-overlap-schedule")
    candidate["expected_effective"]["disable_overlap_schedule"] = overlap
    plan_suite = copy.deepcopy(suite)
    # Reuse canonical merging/validation, without pretending an experiment is
    # a validated profile. The temporary fallback baseline is explicitly off.
    baseline = copy.deepcopy(next(c for c in suite["candidates"] if c["id"] == suite["baseline"]))
    plan_suite["candidates"] = [baseline]
    profile = make_profile(new_plan(plan_suite))
    combined = {**candidate["loader"], **overrides}
    result = launch_command(profile, current, True, combined,
                            (["--disable-overlap-schedule"] if overlap else []) + (server_overrides or []))
    result["config"]["id"] = candidate["id"]
    result["status"] = "experimental_unvalidated"
    result["GPU"] = "not_run"
    return result
