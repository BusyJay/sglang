#!/usr/bin/env python3
"""Replay actual planner bodies on recorded routes; no CUDA or timing claims."""

import argparse
import ctypes as ct
import gzip
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GLOBAL = ROOT / "python/sglang/kernels/jit/csrc/moe/qwen38_hybrid_plan.cuh"
HEAP = ROOT / "python/sglang/kernels/jit/csrc/moe/qwen38_heap_index.cuh"
LOCAL = ROOT / "python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu"
I, D, F, Q = ct.c_int32, ct.c_double, ct.c_float, ct.c_int64


def compile_planners(directory, global_source=None):
    global_source = GLOBAL.read_text() if global_source is None else global_source
    local_source = LOCAL.read_text()
    body = (
        global_source[
            global_source.index(
                "__global__ void hybrid_plan_kernel"
            ) : global_source.index("// Only a selected CPU contribution")
        ]
        .replace("Indexed", "(index_data != nullptr)")
        .replace("void hybrid_plan_kernel(", "void hybrid_plan_impl_replay(")
        .replace(
            'asm volatile("trap;");',
            'throw std::runtime_error("invalid route or insufficient capacity");',
        )
    )
    helpers = global_source[
        global_source.index("struct HybridVictim") : global_source.index(
            "// Global planners and all preceding GPU readers"
        )
    ]
    helpers = helpers.replace('#include "qwen38_heap_index.cuh"', HEAP.read_text())
    baseline = local_source[
        local_source.index("__device__ void apply_plan(") : local_source.index(
            "__global__ void plan_kernel("
        )
    ]
    prefix = """#include <cstdint>
#include <limits>
#include <stdexcept>
#define __global__
#define __shared__
#define __restrict__
#define __device__
#define __forceinline__ inline
struct Index { int x; }; Index threadIdx{0}, blockDim{1};
void __syncthreads() {}
void __threadfence_system() {}
void atomicAdd(int32_t* p,int32_t n) { *p+=n; }
void atomicExch(int32_t* p,int32_t n) { *p=n; }
double __dadd_rn(double a,double b) { return a+b; }
double __dmul_rn(double a,double b) { return a*b; }
constexpr int kRoutes=32, kCostColumns=70;
extern "C" {
"""
    wrapper = """
void hybrid_plan_kernel(const int32_t* ids, int32_t* lookup, double* frequency,
    int32_t* owners, int32_t* keys, const double* costs, int32_t* cache_ids,
    int32_t* cpu_ids, int32_t* cache_plan, int64_t* stats, int32_t tokens,
    int32_t routes, int32_t experts, int32_t slots, int32_t samples,
    int32_t layer, double decay, double admission) {
  hybrid_plan_impl_replay(ids, lookup, frequency, owners, keys, costs,
      cache_ids, cpu_ids, cache_plan, stats, tokens, routes, experts,
      slots, samples, layer, decay, admission);
}
void hybrid_plan_indexed_kernel(const int32_t* ids, int32_t* lookup,
    double* frequency, int32_t* owners, int32_t* keys, const double* costs,
    int32_t* cache_ids, int32_t* cpu_ids, int32_t* cache_plan, int64_t* stats,
    int32_t tokens, int32_t routes, int32_t experts, int32_t slots,
    int32_t samples, int32_t layer, double decay, double admission,
    int32_t* index, int32_t layers, int32_t index_slots) {
  hybrid_plan_impl_replay(ids, lookup, frequency, owners, keys, costs,
      cache_ids, cpu_ids, cache_plan, stats, tokens, routes, experts,
      slots, samples, layer, decay, admission, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, 0, 0, index, layers, index_slots);
}
"""
    cpp = directory / "planners.cpp"
    cpp.write_text(prefix + helpers + body + wrapper + baseline + "}\n")
    library = directory / "planners.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            str(cpp),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ct.CDLL(str(library))
    lib.hybrid_plan_kernel.argtypes = (
        [
            ct.POINTER(I),
            ct.POINTER(I),
            ct.POINTER(D),
            ct.POINTER(I),
            ct.POINTER(I),
            ct.POINTER(D),
        ]
        + [ct.POINTER(I)] * 3
        + [ct.POINTER(Q)]
        + [I] * 6
        + [D] * 2
    )
    lib.hybrid_plan_indexed_kernel.argtypes = lib.hybrid_plan_kernel.argtypes + (
        [ct.POINTER(I), I, I]
    )
    lib.apply_plan.argtypes = (
        [ct.POINTER(I)] * 3
        + [ct.POINTER(F)]
        + [ct.POINTER(Q)] * 3
        + [ct.POINTER(I)]
        + [I] * 2
        + [F]
        + [I] * 2
        + [ct.c_bool, ct.POINTER(I), I]
    )
    return lib


def read_cases(path, layers, experts, topk):
    cases, current, pending = {}, None, []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if "event" in row:
                if pending:
                    raise ValueError("request boundary inside layer sweep")
                if row["event"] == "start":
                    if current is not None or row["case"] in cases:
                        raise ValueError("duplicate or nested case")
                    current = row["case"]
                    cases[current] = []
                elif row["event"] == "end":
                    if current != row["case"]:
                        raise ValueError("mismatched request boundary")
                    current = None
                else:
                    raise ValueError("unknown boundary")
                continue
            routes = row["routes"]
            if row["phase"] != "verify" or row["layer"] != len(pending):
                raise ValueError("expected ordered complete verify sweeps")
            if not 1 <= len(routes) <= 3 or any(len(r) != topk for r in routes):
                raise ValueError("invalid route shape")
            if any(
                type(e) is not int or not -1 <= e < experts for r in routes for e in r
            ):
                raise ValueError("invalid expert")
            if pending and len(routes) != len(pending[0]):
                raise ValueError("inconsistent shape in layer sweep")
            pending.append(routes)
            if len(pending) == layers:
                # Unlabelled graph-setup calls and warmup are not request evidence.
                if current is not None:
                    cases[current].append(pending)
                pending = []
    if pending or current is not None:
        raise ValueError("incomplete trace")
    return {k: v for k, v in cases.items() if k != "warmup"}


def replay(lib, sweeps, budget, geometry, warmup):
    layers, experts, topk = geometry
    layout = [budget // layers + (l < budget % layers) for l in range(layers)]
    if max(layout) > experts:
        raise ValueError("baseline slots exceed per-layer experts")
    if min(layout) < 3 * topk:
        raise ValueError("per-layer capacity cannot hold worst-case active set")
    lookup = (I * (layers * experts))(*([-1] * (layers * experts)))
    heat = (D * (layers * experts))()
    owners = (I * budget)(*([-2] + [-1] * (budget - 1)))  # Existing B12X zero slot.
    keys = (I * budget)(*([-1] * budget))
    costs = (D * 0)()
    stats = (Q * 7)()
    local_states = []
    for slots in layout:
        local_states.append(
            (
                (I * experts)(*([-1] * experts)),
                (I * slots)(*([-1] * slots)),
                (F * slots)(),
                (Q * slots)(),
                (Q * slots)(),
                (Q * 1)(),
                (I * (4 + 4 * slots))(),
            )
        )
    result = {name: Counter() for name in ["per_layer_gpu", "global_gpu"]}
    per_layer = {name: [Counter() for _ in range(layers)] for name in result}
    occupancy = None
    for sweep, events in enumerate(sweeps):
        for layer, rows in enumerate(events):
            flat = sum(rows, [])
            counts = Counter(e for e in flat if e >= 0)
            ids = (I * len(flat))(*flat)
            gpu, cpu = (I * len(flat))(), (I * len(flat))()
            plan = (I * (4 + 2 * len(flat)))()
            previous = list(stats)
            lib.hybrid_plan_kernel(
                ids,
                lookup,
                heat,
                owners,
                keys,
                costs,
                gpu,
                cpu,
                plan,
                stats,
                len(rows),
                len(flat),
                experts,
                budget,
                0,
                layer,
                0.95,
                1.9,
            )
            histogram = (I * (experts + 1))(*[counts[e] for e in range(experts)], 0)
            state = local_states[layer]
            slots = layout[layer]
            lib.apply_plan(
                histogram, *state, experts, slots, 0.95, 0, 0, False, None, -1
            )
            if state[-1][0] or state[-1][3]:
                raise ValueError("baseline failed to place active experts")
            assert all(
                cpu[i] == -1 and (e < 0 or gpu[i] >= 0) for i, e in enumerate(flat)
            )
            if sweep < warmup:
                continue
            for name, misses in [
                ("per_layer_gpu", state[-1][1]),
                ("global_gpu", plan[1]),
            ]:
                values = Counter(
                    calls=1,
                    unique_accesses=len(counts),
                    gpu_fills=misses,
                    all_hit_calls=int(misses == 0),
                )
                if name == "global_gpu":
                    values.update(
                        lfu_admitted=stats[3] - previous[3],
                        forced_placements=stats[6] - previous[6],
                        evictions=stats[0] - previous[0],
                    )
                result[name].update(values)
                per_layer[name][layer].update(values)
        if sweep == warmup - 1:
            occupancy = [
                sum(owners[s] == l and keys[s] >= 0 for s in range(budget))
                for l in range(layers)
            ]
    for name, row in result.items():
        row["hit_fraction"] = 1 - row["gpu_fills"] / row["unique_accesses"]
    return dict(
        layout=layout,
        global_reserved_slots=1,
        scored_sweeps=len(sweeps) - warmup,
        warmup_sweeps=warmup,
        results=result,
        per_layer=per_layer,
        global_warmup_occupancy=occupancy,
        global_final_occupancy=[
            sum(owners[s] == l and keys[s] >= 0 for s in range(budget))
            for l in range(layers)
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT
        / "tools/qwen38_cache_tests/verify_reuse_diagnosis_20261018/service/verify-routes.jsonl.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "tools/qwen38_cache_tests/cache_policy_experiments_20261018/live_fp4_results/pack-manifest.json",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[6388, 6390, 6643, 7006]
    )
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    m = manifest["model"]
    geometry = (m["num_layers"], m["num_experts"], m["top_k"])
    if not 1 <= m["top_k"] <= 10 or not 1 <= m["num_experts"] <= 512:
        raise ValueError("unsupported planner geometry")
    payloads = [
        sum(
            t["nbytes"] // m["num_experts"]
            for t in tensors.values()
            if len(t["shape"]) > 1
        )
        for tensors in manifest["layers"].values()
    ]
    if len(set(payloads)) != 1:
        raise ValueError("replay requires homogeneous expert payload sizes")
    expert_bytes = payloads[0]
    cases = read_cases(args.trace, *geometry)
    if args.warmup < 0 or any(len(s) <= args.warmup for s in cases.values()):
        raise ValueError("invalid warmup")
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        lib = compile_planners(Path(tmp))
        for budget in args.budgets:
            for name, sweeps in cases.items():
                row = replay(lib, sweeps, budget, geometry, args.warmup)
                row.update(case=name, budget=budget)
                for metrics in row["results"].values():
                    metrics["expert_h2d_bytes"] = metrics["gpu_fills"] * expert_bytes
                baseline = row["results"]["per_layer_gpu"]["gpu_fills"]
                new = row["results"]["global_gpu"]["gpu_fills"]
                row["transfer_delta_fraction"] = new / baseline - 1
                reports.append(row)
                print(
                    budget,
                    name,
                    "per-layer",
                    baseline,
                    "global",
                    new,
                    "delta",
                    f"{100 * (new / baseline - 1):+.2f}%",
                    flush=True,
                )
    output = dict(
        status="actual_planner_host_replay",
        geometry=m,
        expert_payload_bytes=expert_bytes,
        reports=reports,
        provenance={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.trace, args.manifest, LOCAL, GLOBAL]
        },
        limits=[
            "Actual CUDA planner bodies executed serially on host; no CUDA execution or throughput prediction.",
            "Baseline is non-global per-layer LFU, FP32 resident heat and LRU tie break; fixed balanced slot allocation.",
            "New planner uses FP64 per-expert heat and forced global placement, empty calibration: all GPU, no CPU transfers.",
            "Same total expert slot budget; global planner additionally reserves its existing B12X zero slot.",
            "Each request starts from empty and replays its own prefix; synthetic setup and separate warmup case excluded.",
            "Trace only contains target verify calls: prefill, draft, initial production cache, adaptive rebalance and resize are not recorded.",
            "H2D payload covers expert weights and block scales; excludes metadata, shared scales, CPU activations and physical bus overhead.",
        ],
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
