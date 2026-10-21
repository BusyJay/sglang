# SPDX-License-Identifier: Apache-2.0
"""Paired end-to-end branch calibration on the target SM120 GPU and CPU."""

import argparse
import collections
import hashlib
import json
import mmap
import os
import random
import statistics
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def trim_pairs(pairs, margin, spread_limit):
    """Accept only stable, measured wins; otherwise leave the route on GPU."""
    cpu = [p[0] for p in pairs]
    gpu = [p[1] for p in pairs]
    if len(pairs) < 5 or min(cpu + gpu) <= 0:
        return None
    cpu_med, gpu_med = statistics.median(cpu), statistics.median(gpu)
    if max(cpu) / min(cpu) > spread_limit or max(gpu) / min(gpu) > spread_limit:
        return None
    if cpu_med >= gpu_med * (1 - margin):
        return None
    return cpu_med, gpu_med


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cores", required=True, help="physical CPU cores, e.g. 0,2,4,6")
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--spread-limit", type=float, default=2.0)
    p.add_argument(
        "--source-mode", choices=["file_cow", "shared_thp"], default="file_cow"
    )
    p.add_argument("--epilogue", choices=["fused", "merge"], default="merge")
    args = p.parse_args()
    if args.samples < 5 or args.warmup < 1 or not 0 <= args.margin < 1:
        p.error("samples >= 5, warmup >= 1 and margin in [0,1) required")
    if args.spread_limit <= 1:
        p.error("spread-limit must exceed 1")
    calibration_path = args.output.with_suffix(".calibration.json")
    if calibration_path.exists():
        p.error(f"refusing to replace an existing calibration: {calibration_path}")
    try:
        cores = tuple(map(int, args.cores.split(",")))
    except ValueError:
        p.error("cores must be comma-separated CPU numbers")
    if not cores or not set(cores) <= os.sched_getaffinity(0):
        p.error("CPU cores must be available")
    fill_cta = os.getenv("SGLANG_QWEN38_FILL_CTA", "8")
    if fill_cta not in tuple(str(i) for i in range(1, 9)) + ("adaptive",):
        p.error("SGLANG_QWEN38_FILL_CTA must be 1..8 or adaptive")
    fill_cta = 0 if fill_cta == "adaptive" else int(fill_cta)
    if os.getenv("SGLANG_QWEN38_PREFILL_REGISTERED_DMA", "1") != "1":
        p.error("calibration requires registered DMA, as used by this benchmark")

    import torch
    from cuda.bindings import driver
    from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4
    from sglang.kernels.ops.moe.qwen38_expert_cache import _module as fill_module
    from sglang.kernels.ops.moe.qwen38_hybrid_plan import (
        COST_COLUMNS,
        hybrid_merge,
        hybrid_plan_stage_indexed,
    )
    from sglang.srt.layers.moe.qwen38_heap_index import build_heap_index
    from sglang.srt.layers.moe.qwen38_hybrid import HybridCache
    from sglang.srt.layers.moe.qwen38_hybrid_device import (
        DeviceHybridPlanner,
        _MappedBatchPage,
    )
    from sglang.srt.layers.moe.qwen38_hybrid_policy import HybridCostModel
    from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
        Qwen38ExpertPack,
        load_manifest,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("full-path calibration requires a local SM120 GPU")
    manifest = load_manifest(args.manifest)
    selected = json.loads(args.cases.read_text())
    if not selected.get("cases") or not selected.get("complete_keys"):
        raise ValueError("route replay produced no complete workload keys")

    def normalized_key(key):
        if len(key) != 6:
            raise ValueError("invalid workload key in route replay")
        return (*key[:4], tuple(sorted(key[4])), tuple(sorted(key[5])))

    complete_keys = {normalized_key(key) for key in selected["complete_keys"]}
    case_keys = {normalized_key(case["key"]) for case in selected["cases"]}
    if not case_keys or case_keys != complete_keys:
        raise ValueError("route replay keys are incomplete or inconsistent")
    root = Path(__file__).resolve().parents[4]
    for source, expected in selected["provenance"].items():
        if hashlib.sha256((root / source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"case source changed since route replay: {source}")
    if selected["geometry"] != dict(
        hidden_size=manifest.hidden_size,
        intermediate_size=manifest.intermediate_size,
        num_experts=manifest.num_experts,
        num_layers=manifest.num_layers,
        top_k=manifest.top_k,
    ):
        raise ValueError("case geometry does not match pack")
    pack = Qwen38ExpertPack(manifest, source_mode=args.source_mode)
    cpu = CpuNvfp4(args.library, pack, cores, gpu_backend="b12x")
    source = torch.frombuffer(pack.cuda_source_mapping(), dtype=torch.uint8)
    fill = fill_module()
    source_address = fill.mailbox_register(source)
    page = _MappedBatchPage(manifest.hidden_size, manifest.top_k, torch.device("cuda"))
    failure_map = mmap.mmap(-1, mmap.PAGESIZE, flags=mmap.MAP_SHARED)
    failure_bytes = torch.frombuffer(failure_map, dtype=torch.uint8)
    failure_head = torch.frombuffer(failure_map, dtype=torch.int32, count=1)
    failed_address = fill.mailbox_register(failure_bytes)
    result_stream = torch.cuda.Stream()
    state = SimpleNamespace(
        gpu_backend="b12x",
        mapped_batch_pages=[page],
        cpu=cpu,
        failure=None,
        failure_head=failure_head,
        stats=collections.Counter(),
        result_stream=result_stream,
        runtime=SimpleNamespace(
            poisoned=False,
            miss_mailbox=SimpleNamespace(
                _abort=lambda error: state.failure_head.fill_(1)
            ),
        ),
    )
    planner = DeviceHybridPlanner.__new__(DeviceHybridPlanner)
    planner.cache, planner.driver = state, driver
    planner.flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    stopped = threading.Event()

    def worker():
        try:
            with torch.cuda.device("cuda"):
                while not stopped.is_set():
                    request = planner.poll()
                    if request is not None:
                        planner.cpu_task(*request)
        except BaseException as error:
            state.failure = error
            state.failure_head.fill_(1)
        finally:
            try:
                cpu.release_worker()
            except BaseException as error:
                state.failure = error
                state.failure_head.fill_(1)

    thread = threading.Thread(target=worker, name="qwen38-gate-cpu", daemon=True)
    names = (
        "w13_weight",
        "w2_weight",
        "w13_blockscale_swizzled",
        "w2_blockscale_swizzled",
        "g1_alphas",
        "g2_alphas",
    )
    dtypes = dict(
        w13_weight=torch.uint8,
        w2_weight=torch.uint8,
        w13_blockscale_swizzled=torch.float8_e4m3fn,
        w2_blockscale_swizzled=torch.float8_e4m3fn,
        g1_alphas=torch.float32,
        g2_alphas=torch.float32,
    )
    # Kernel launch geometry and alpha-refresh work depend on the actual pool
    # capacity. A tiny synthetic pool would understate the serving GPU cost.
    capacity = selected["budget"]
    if capacity < 1 + 3 * manifest.top_k:
        raise ValueError("benchmark budget cannot hold the active verify routes")
    pool = {
        name: torch.zeros(
            (capacity, *manifest.tensors[0][name].shape[1:]),
            dtype=dtypes[name],
            device="cuda",
        )
        for name in names
    }
    backend = CachedB12xW4A16(
        torch.device("cuda"),
        manifest.hidden_size,
        manifest.intermediate_size,
        capacity,
        enable_fp32=True,
    )
    report = dict(
        status="running",
        cases=[],
        rejected=[],
        epilogue=args.epilogue,
        source_mode=args.source_mode,
        warmup=args.warmup,
        samples=args.samples,
    )
    key_measurements = collections.defaultdict(list)
    metadata = dict(
        backend="b12x",
        hidden=manifest.hidden_size,
        width=manifest.intermediate_size,
        top_k=manifest.top_k,
        cores=len(cores),
        gpu=torch.cuda.get_device_name(),
        cpu=next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        transport="registered_dma",
        runtime_version=HybridCache._cost_version(
            SimpleNamespace(
                cpu=cpu, gpu_backend="b12x", runtime=SimpleNamespace(manifest=manifest)
            )
        ),
    )
    table = dict(schema_version=1, margin=args.margin, samples=[], **metadata)
    rng = random.Random(617)
    sequence = 0
    thread.start()
    try:
        for case in selected["cases"]:
            if state.failure is not None:
                raise RuntimeError("CPU worker failed") from state.failure
            layer = case["layer"]
            routes = case["rows"]
            tokens = len(routes)
            present = {e for row in routes for e in row if e >= 0}
            cpu_experts = set(case["cpu_experts"])
            initial = (
                present - set(case["admitted_experts"]) - set(case["rejected_experts"])
            )
            if not cpu_experts or not cpu_experts <= set(case["rejected_experts"]):
                raise ValueError("invalid CPU subset")
            weights = torch.full(
                (tokens, manifest.top_k), 0.1, dtype=torch.float32, device="cuda"
            )
            x = torch.randn(
                tokens, manifest.hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            original_ids = torch.tensor(routes, dtype=torch.int32, device="cuda")
            rows = manifest.tensors[layer]
            layout = torch.tensor(
                [
                    (
                        pool[name].data_ptr(),
                        rows[name].nbytes // manifest.num_experts,
                        source_address + rows[name].offset,
                    )
                    for name in names
                ],
                dtype=torch.int64,
                device="cuda",
            )
            scales = {
                name: torch.frombuffer(
                    pack.cuda_source_mapping(),
                    dtype=torch.float32,
                    count=1,
                    offset=rows[name].offset,
                )
                .clone()
                .to("cuda")
                for name in ("w13_input_scale_quant", "w2_input_scale_quant")
            }
            tensors = {**pool, **scales}

            def make_plan(experts, slots):
                plan = torch.zeros(
                    4 + 2 * max(1, len(experts)), dtype=torch.int32, device="cuda"
                )
                if experts:
                    plan[1] = len(experts)
                    for i, expert in enumerate(experts):
                        plan[4 + 2 * i] = expert
                        plan[5 + 2 * i] = slots[expert]
                return plan

            slots = {e: i + 1 for i, e in enumerate(sorted(present))}
            if len(slots) >= capacity:
                raise ValueError("case exceeds GPU pool")
            # Preload the same initially resident experts before both branches.
            initial_experts = sorted(initial)
            initial_plan = make_plan(initial_experts, slots)
            fill.mailbox_fill_grid(
                initial_plan,
                layout,
                max(1, len(initial_experts)),
                failed_address,
                fill_cta,
            )
            torch.cuda.synchronize()
            # Rebuild the same LFU state before each paired branch. The free
            # slots admit the recorded experts; hot dummy residents keep the
            # recorded rejected set out of the cache until forced placement.
            lookup = torch.full(
                (manifest.num_layers, manifest.num_experts),
                -1,
                dtype=torch.int32,
                device="cuda",
            )
            frequency = torch.zeros_like(lookup, dtype=torch.float64)
            owners = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
            keys = torch.full_like(owners, -1)
            owners[0] = -2
            for expert in initial:
                slot = slots[expert]
                lookup[layer, expert] = slot
                frequency[layer, expert] = 10
                owners[slot], keys[slot] = layer, expert
            for expert in case["admitted_experts"]:
                frequency[layer, expert] = 2.0
            free_slots = {slots[expert] for expert in case["admitted_experts"]}
            dummy_pairs = (
                (other_layer, expert)
                for other_layer in range(manifest.num_layers)
                if other_layer != layer
                for expert in range(manifest.num_experts)
            )
            for slot in range(1, capacity):
                if owners[slot] >= 0 or slot in free_slots:
                    continue
                other_layer, expert = next(dummy_pairs)
                owners[slot], keys[slot] = other_layer, expert
                lookup[other_layer, expert] = slot
                frequency[other_layer, expert] = 100
            policy_state = (lookup, frequency, owners, keys)
            baseline = tuple(t.clone() for t in policy_state)
            index = build_heap_index(
                frequency.cpu().tolist(),
                owners.cpu().tolist(),
                keys.cpu().tolist(),
                manifest.num_experts,
                capacity,
            ).to("cuda")
            baseline_index = index.clone()
            counts = collections.Counter(e for row in routes for e in row if e >= 0)
            rejected = sorted(case["rejected_experts"], key=lambda e: (counts[e], e))
            cpu_count = sum(counts[e] for e in cpu_experts)
            profile = [counts[e] for e in rejected]
            cpu_profile = profile[: len(cpu_experts)]
            cost_row = (
                [
                    tokens,
                    sum(counts.values()) - cpu_count,
                    cpu_count,
                    len(cpu_experts),
                    len(rejected),
                    0,
                ]
                + cpu_profile
                + [0] * (32 - len(cpu_profile))
                + profile
                + [0] * (32 - len(profile))
            )
            if (
                len(cost_row) != COST_COLUMNS
                or set(rejected[: len(cpu_experts)]) != cpu_experts
            ):
                raise ValueError("route case does not match planner cost profile")
            mixed_costs = torch.tensor([cost_row], dtype=torch.float64, device="cuda")
            gpu_costs = torch.empty(
                (0, COST_COLUMNS), dtype=torch.float64, device="cuda"
            )
            torch.cuda.synchronize()

            completed = torch.cuda.Event()

            def invoke(mixed):
                nonlocal sequence
                cache_ids = torch.empty_like(original_ids)
                cpu_ids = torch.empty_like(original_ids)
                plan = torch.empty(
                    4 + 2 * original_ids.numel(), dtype=torch.int32, device="cuda"
                )
                stats = torch.zeros(7, dtype=torch.int64, device="cuda")
                sequence += 1
                hybrid_plan_stage_indexed(
                    original_ids,
                    *policy_state,
                    mixed_costs if mixed else gpu_costs,
                    cache_ids,
                    cpu_ids,
                    plan,
                    stats,
                    layer,
                    0.95,
                    1.9,
                    x,
                    weights,
                    *page.batch.device_addresses[:3],
                    page.batch.device_addresses[4],
                    sequence,
                    index,
                )
                fill.mailbox_fill_grid(
                    plan, layout, original_ids.numel(), failed_address, fill_cta
                )
                result = backend.run(
                    x,
                    cache_ids,
                    weights,
                    tensors,
                    output_dtype=torch.float32 if args.epilogue == "merge" else None,
                    hybrid=(
                        (
                            plan,
                            page.batch.device_addresses[5],
                            page.batch.device_addresses[3],
                            failed_address,
                            sequence,
                        )
                        if args.epilogue == "fused"
                        else None
                    ),
                )
                if args.epilogue == "merge":
                    output = torch.empty_like(x)
                    hybrid_merge(
                        cpu_ids,
                        result,
                        planner.flag,
                        output,
                        page.batch.device_addresses[3],
                        sequence,
                        failed_address,
                        completed_address=page.batch.device_addresses[5],
                    )
                    result = output
                completed.record(torch.cuda.current_stream())
                completed.synchronize()
                if state.failure is not None:
                    raise RuntimeError("CPU worker failed") from state.failure
                if not hasattr(invoke, "checked"):
                    invoke.checked = set()
                if mixed not in invoke.checked:
                    actual_cpu = set(cpu_ids[cpu_ids >= 0].tolist())
                    if actual_cpu != (cpu_experts if mixed else set()):
                        raise ValueError("planner chose a different CPU subset")
                    actual_fills = set(plan[4 : 4 + 2 * plan[1].item() : 2].tolist())
                    expected_fills = (
                        present - initial - (cpu_experts if mixed else set())
                    )
                    if actual_fills != expected_fills:
                        raise ValueError("planner generated different GPU fills")
                    invoke.checked.add(mixed)
                return result

            measurements = []
            outputs = {}
            for step in range(args.warmup + args.samples):
                order = [False, True]
                rng.shuffle(order)
                pair = {}
                for mixed in order:
                    # Starting cache content is identical. Nonresident slots
                    # may contain stale bytes, but the fill overwrites them.
                    for target, initial_value in zip(policy_state, baseline):
                        target.copy_(initial_value)
                    index.copy_(baseline_index)
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    result = invoke(mixed)
                    elapsed = (time.perf_counter_ns() - start) / 1000
                    pair["cpu" if mixed else "gpu"] = elapsed
                    outputs[mixed] = result
                torch.testing.assert_close(
                    outputs[True],
                    outputs[False],
                    rtol=0.025,
                    atol=0.003,
                )
                if step >= args.warmup:
                    measurements.append((pair["cpu"], pair["gpu"]))
            gate = trim_pairs(measurements, args.margin, args.spread_limit)
            row = dict(
                key=case["key"],
                layer=layer,
                count=case["count"],
                samples_us=measurements,
                cpu_us=statistics.median(p[0] for p in measurements),
                gpu_us=statistics.median(p[1] for p in measurements),
                accepted=gate is not None,
                paired_frame=case.get("paired_frame"),
                cpu_experts=case["cpu_experts"],
                admitted_experts=case["admitted_experts"],
                rejected_experts=case["rejected_experts"],
            )
            report["cases"].append(row)
            raw_key = case["key"]
            key = normalized_key(raw_key)
            key_measurements[key].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                case["key"],
                "cpu",
                row["cpu_us"],
                "gpu",
                row["gpu_us"],
                "accept",
                row["accepted"],
                flush=True,
            )
        # Exact runtime keys cannot distinguish layer or admitted/rejected GPU
        # mix. Admit a key only if every measured representative wins; use the
        # worst measured paired ratio to keep the gate conservative. Mixing
        # CPU time from one workload with GPU time from another destroys the
        # paired comparison (different layers can have different fill counts).
        for key, representatives in key_measurements.items():
            if key not in complete_keys:
                report["rejected"].append(
                    dict(key=key, reason="truncated trace representatives")
                )
                continue
            if not all(row["accepted"] for row in representatives):
                report["rejected"].append(
                    dict(key=key, reason="unstable or nonwinning representative")
                )
                continue
            worst = max(representatives, key=lambda row: row["cpu_us"] / row["gpu_us"])
            cpu_us, gpu_us = worst["cpu_us"], worst["gpu_us"]
            table["samples"].append(
                dict(
                    tokens=key[0],
                    hit_routes=key[1],
                    miss_routes=key[2],
                    miss_experts=key[3],
                    miss_tokens=list(sorted(key[4])),
                    rejected_tokens=list(sorted(key[5])),
                    cpu_us=cpu_us,
                    gpu_us=gpu_us,
                )
            )
        # Validate the exact runtime schema before making the table available.
        HybridCostModel(table, **metadata)
    finally:
        stopped.set()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("CPU worker did not stop; mappings retained")
        if state.failure is not None:
            raise RuntimeError("CPU worker failed") from state.failure
        torch.cuda.synchronize()
        fill.mailbox_unregister(failure_bytes)
        fill.mailbox_unregister(source)
        page.close(torch.device("cuda"))
        state.failure_head = None
        del failure_head, failure_bytes
        failure_map.close()
        del source
        cpu.close()
        pack.close()
    # Publish only after all GPU work and the CPU worker have stopped cleanly.
    # A failed or interrupted write must never leave a loadable partial table.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=calibration_path.parent, prefix=".qwen38-gate-", delete=False
    ) as output:
        temporary_path = Path(output.name)
        os.fchmod(output.fileno(), 0o644)
        output.write(json.dumps(table, indent=2) + "\n")
    try:
        os.link(temporary_path, calibration_path)
    finally:
        temporary_path.unlink()
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
