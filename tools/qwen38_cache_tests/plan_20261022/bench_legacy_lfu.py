"""Time the pre-hybrid layer-local LFU planner, including heat and reclaim.

Use the same 30 flattened routes as M=3/top-k=10 in the global benchmark.
Local slots and FP32 heat follow the legacy serving implementation; this is
an algorithm-scope comparison, not equivalent global cache decisions.
"""

import statistics
import torch
from sglang.kernels.ops.moe.qwen38_expert_cache import _module

for case in ("all_hit", "full_miss", "free_miss"):
    slots = 146
    resident = slots - (64 if case == "free_miss" else 0)
    lookup = torch.full((512,), -1, dtype=torch.int32, device="cuda")
    lookup[:resident] = torch.arange(resident, device="cuda", dtype=torch.int32)
    keys = torch.full((slots,), -1, dtype=torch.int32, device="cuda")
    keys[:resident] = torch.arange(resident, device="cuda", dtype=torch.int32)
    freq = torch.zeros(slots, device="cuda")
    freq[:resident] = 10 + torch.arange(resident, device="cuda") / 512
    ages = torch.zeros(slots, dtype=torch.int64, device="cuda")
    generations = torch.zeros_like(ages)
    clock = torch.zeros(1, dtype=torch.int64, device="cuda")
    source = (lookup, keys, freq, ages, generations, clock)
    state = tuple(torch.empty_like(t) for t in source)
    base = 0 if case == "all_hit" else resident
    ids = torch.tensor(
        [[base + r + t for t in range(3) for r in range(10)]],
        device="cuda",
        dtype=torch.int32,
    )
    output = torch.empty_like(ids)
    counts = torch.empty(513, device="cuda", dtype=torch.int32)
    plan = torch.empty(4 + 4 * slots, device="cuda", dtype=torch.int32)
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for i in range(120):
        for dst, src in zip(state, source):
            dst.copy_(src)
        begin.record()
        _module().decode_plan_parallel(
            ids, output, counts, *state, plan, 0.95, 0, 0.7, 0, 0, 2
        )
        end.record()
        end.synchronize()
        if i >= 20:
            times.append(begin.elapsed_time(end) * 1000)
    print(case, {"legacy_lfu_us": round(statistics.median(times), 3)}, flush=True)
