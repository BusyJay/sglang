"""Frozen-state replay: exact state hashes and graph reset+planner GPU timings.
The reset is included in graph timings identically for both revisions. A small
CUDA-only profiler trace separately records the planner's kernel durations.
"""

import hashlib, json, mmap, statistics, sys
from pathlib import Path
import torch
from sglang.kernels.ops.moe.qwen38_expert_cache import _module
from sglang.srt.layers.moe.qwen38_cache_buffers import AdaptiveMetadataBuffers

r = Path("/results")
name = sys.argv[1]
mod = _module()
torch.manual_seed(42)
report = {"revision": name, "cases": []}
snapshots = {}
page = mmap.mmap(-1, 65536, flags=mmap.MAP_SHARED)
host = torch.frombuffer(page, dtype=torch.uint8)
address = mod.mailbox_register(host)
for slots in [16, 64, 146, 256]:
    for tokens in [1, 3]:
        for kind in ["hit", "cpu", "hot", "over-limit", "free"]:
            for limit in [0, 1]:
                metadata = AdaptiveMetadataBuffers(1, 512, slots, torch.device("cuda"))
                state = metadata.reset(0, slots)
                perm = torch.randperm(512).tolist()
                nr = slots - 1 if kind == "free" else slots
                resident = perm[:nr]
                missing = perm[nr:]
                metadata.restore(
                    0, state, [(e, 10.0, j + 1) for j, e in enumerate(resident)]
                )
                ids = torch.tensor(
                    [resident[:10]] * tokens, dtype=torch.int32, device="cuda"
                )
                heat = torch.zeros(512, device="cuda")
                if kind != "hit":
                    ids[:, -1] = missing[0]
                if kind in ["over-limit", "free"]:
                    ids[:, -3:] = torch.tensor(missing[:3], device="cuda")
                if kind == "hot":
                    heat[missing[0]] = 20
                if kind == "free":
                    heat[missing[1]] = 2
                    heat[missing[2]] = 4
                x = torch.ones((tokens, 2560), device="cuda", dtype=torch.bfloat16)
                weights = torch.ones((tokens, 10), device="cuda")
                outputs = [
                    torch.empty_like(ids),
                    torch.empty_like(ids),
                    torch.empty_like(weights),
                    torch.empty(2, device="cuda", dtype=torch.int32),
                ]
                remap, cpu, gpuw, desc = outputs
                stats = torch.zeros(5, device="cuda", dtype=torch.int64)
                table = torch.empty((0, 47), device="cuda", dtype=torch.int32)
                initial = {
                    k: v.clone()
                    for k, v in state.items()
                    if isinstance(v, torch.Tensor) and k != "prefill_plan"
                }
                heat0 = heat.clone()

                def reset():
                    for k, v in initial.items():
                        state[k].copy_(v)
                    heat.copy_(heat0)
                    stats.zero_()

                def plan():
                    mod.decode_plan_layered_lfu(
                        ids.view(1, -1),
                        remap.view(1, -1),
                        *[
                            state[k]
                            for k in [
                                "counts",
                                "lookup",
                                "keys",
                                "frequency",
                                "ages",
                                "generation",
                                "clock",
                                "plan",
                            ]
                        ],
                        0.95,
                        0,
                        0.7,
                        0,
                        0,
                        2,
                        x,
                        weights,
                        table,
                        gpuw,
                        cpu,
                        desc,
                        stats,
                        address,
                        0,
                        1,
                        -1,
                        1,
                        0,
                        heat,
                        limit,
                    )

                reset()
                plan()
                torch.cuda.synchronize()
                # Only defined plan prefixes are hashed; unused capacity is unspecified.
                planv = state["plan"].cpu()
                misses = int(planv[1])
                active = int(planv[2])
                h = hashlib.sha256()
                values = {
                    k: v
                    for k, v in state.items()
                    if isinstance(v, torch.Tensor) and k not in ("plan", "prefill_plan")
                }
                values.update(
                    heat=heat,
                    remap=remap,
                    cpu=cpu,
                    gpuw=gpuw,
                    desc=desc,
                    stats=stats,
                    plan_header=planv[:4],
                    plan_misses=planv[4 : 4 + 2 * misses],
                    plan_slots=planv[4 + 2 * slots : 4 + 2 * slots + active],
                    plan_experts=planv[4 + 3 * slots : 4 + 3 * slots + active],
                )
                for k, v in sorted(values.items()):
                    h.update(k.encode())
                    h.update(v.cpu().numpy().tobytes())
                snapshots[(slots, tokens, kind, limit)] = {
                    k: v.cpu().clone() for k, v in values.items()
                }
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        reset()
                        plan()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    reset()
                    plan()
                for _ in range(10):
                    graph.replay()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(7):
                    begin.record()
                    for _ in range(100):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(begin.elapsed_time(end) * 10)
                report["cases"].append(
                    dict(
                        slots=slots,
                        tokens=tokens,
                        kind=kind,
                        n=limit,
                        state_sha256=h.hexdigest(),
                        cpu_experts=int(desc[0]),
                        graph_median_us=statistics.median(times),
                        graph_samples_us=times,
                    )
                )
                if slots == 146 and tokens == 3 and limit == 1:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        for _ in range(31):
                            graph.replay()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(r / f"{name}-{kind}.trace.json"))
                (r / f"{name}-planner.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
        print(name, slots, tokens, "done", flush=True)
# Multi-step replay also catches arithmetic drift that changes later admission.
report["chains"] = []
for slots in [16, 146, 256]:
    for limit in [0, 1, 3, 30]:
        metadata = AdaptiveMetadataBuffers(1, 512, slots, torch.device("cuda"))
        state = metadata.reset(0, slots)
        perm = torch.randperm(512).tolist()
        resident = perm[:slots]
        frequencies = (4 + torch.rand(slots) * 16).tolist()
        metadata.restore(
            0, state, [(e, frequencies[j], j + 1) for j, e in enumerate(resident)]
        )
        heat = torch.rand(512, device="cpu").cuda() * 4
        ids = torch.empty((3, 10), dtype=torch.int32, device="cuda")
        x = torch.ones((3, 2560), device="cuda", dtype=torch.bfloat16)
        weights = torch.ones((3, 10), device="cuda")
        remap = torch.empty_like(ids)
        cpu = torch.empty_like(ids)
        gpuw = torch.empty_like(weights)
        desc = torch.empty(2, device="cuda", dtype=torch.int32)
        stats = torch.zeros(5, device="cuda", dtype=torch.int64)
        h = hashlib.sha256()
        cpu_total = 0
        for step in range(64):
            domain = perm if slots >= 32 else resident[:8] + perm[slots : slots + 8]
            routes = torch.tensor(domain + [-1], dtype=torch.int32)[
                torch.randint(0, len(domain) + 1, (3, 10))
            ]
            routes[:, 0] = resident[0]
            ids.copy_(routes)
            plan()
            torch.cuda.synchronize()
            planv = state["plan"].cpu()
            misses = int(planv[1])
            active = int(planv[2])
            cpu_total += int(desc[0])
            values = {
                k: v
                for k, v in state.items()
                if isinstance(v, torch.Tensor) and k not in ("plan", "prefill_plan")
            }
            values.update(
                heat=heat,
                remap=remap,
                cpu=cpu,
                gpuw=gpuw,
                desc=desc,
                stats=stats,
                plan_header=planv[:4],
                plan_misses=planv[4 : 4 + 2 * misses],
                plan_slots=planv[4 + 2 * slots : 4 + 2 * slots + active],
                plan_experts=planv[4 + 3 * slots : 4 + 3 * slots + active],
            )
            for k, v in sorted(values.items()):
                h.update(k.encode())
                h.update(v.cpu().numpy().tobytes())
        snapshots[("chain", slots, limit)] = {
            k: v.cpu().clone() for k, v in values.items()
        }
        report["chains"].append(
            dict(
                slots=slots,
                n=limit,
                steps=64,
                cpu_total=cpu_total,
                state_sha256=h.hexdigest(),
            )
        )
        print(name, "chain", slots, limit, "done", flush=True)
torch.save(snapshots, r / f"{name}-snapshots.pt")
mod.mailbox_unregister(host)
del host
page.close()
report["status"] = "passed"
(r / f"{name}-planner.json").write_text(json.dumps(report, indent=2) + "\n")
