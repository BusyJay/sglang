"""Fixed-input GPU/host measurements supplement the serving ablation.

Run while the inference server is stopped. GPU events measure captured work;
CUDA API/host enqueue time is reported separately for U2/U4. C1/C2 totals and
post-fill consumers are separate: moving work does not eliminate its GPU cost.
"""

import argparse
import ast
import gc
import importlib.util
import json
import statistics
import subprocess
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.activation.softcap import cast_to_float32_logits
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split
from sglang.kernels.ops.elementwise.hc_combine_decode import (
    hc_combine_apply,
    hc_combine_gate,
)
from sglang.kernels.ops.moe.shared_expert_gate import (
    shared_expert_gate,
    shared_expert_gate_add,
)
from sglang.srt.layers.attention.qsa.kernel import gather_qsa_compressed_keys
from sglang.srt.utils.custom_op import register_custom_op

BASE = "cd748677bb6bd87c3162b24c8a4d3591dce2c8ea"
REPO = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_grad_enabled(False)
torch.manual_seed(20261025)
results = [
    dict(
        component="environment",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
    )
]

# Probe the integration contract as well as the row-contiguous kernel. The
# previous copy_ accepted column-strided logits; an optimized caller must keep
# that fallback even when the GPU cast is enabled.
from sglang.srt.layers.logits_processor import LogitsProcessor

column_input = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
column_output = torch.empty(3, 64, device="cuda", dtype=torch.float32)
try:
    actual = LogitsProcessor._copy_logits_to_buffer(
        SimpleNamespace(vocab_size=64),
        column_input,
        SimpleNamespace(next_token_logits_buffer=column_output),
    )
    torch.testing.assert_close(actual, column_input.float(), rtol=0, atol=0)
    results.append(dict(component="U3 column-stride integration", status="passed"))
except AssertionError as error:
    results.append(
        dict(
            component="U3 column-stride integration",
            status="regression",
            error=str(error),
        )
    )
flush = torch.ones(32 * 1024 * 1024, dtype=torch.float32, device="cuda")


def stats(values):
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def gpu_time(fn, cold=False, repetitions=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graphs, events = [], []
    work = 1 if cold else 100
    if cold:
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                flush.add_(1)
                start.record()
                fn()
                end.record()
            graphs.append(graph)
            events.append((start, end))
        for graph in graphs:
            graph.replay()
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(work):
                fn()
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            events.append((start, end))
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) * 1000 / work for start, end in events]
    del graphs, events, graph
    gc.collect()
    return stats(values)


def compare(component, shape, legacy, candidate):
    old, new = legacy().clone(), candidate()
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    for cold in (False, True):
        # ABBA counterbalances clocks/cache warmup. Both graphs do identical inputs.
        series = {"legacy": [], "candidate": []}
        for label, fn in (
            ("legacy", legacy),
            ("candidate", candidate),
            ("candidate", candidate),
            ("legacy", legacy),
        ):
            series[label].append(gpu_time(fn, cold))
        before = statistics.median(x["median_us"] for x in series["legacy"])
        after = statistics.median(x["median_us"] for x in series["candidate"])
        row = dict(
            component=component,
            shape=shape,
            cache="cold-L2" if cold else "warm-L2",
            legacy_us=before,
            candidate_us=after,
            latency_change_percent=100 * (after / before - 1),
            rounds=series,
            parity="bitwise",
        )
        results.append(row)
        print(
            component,
            shape,
            row["cache"],
            round(before, 3),
            round(after, 3),
            flush=True,
        )
    args.output.write_text(json.dumps(results, indent=2))


for dtype in (torch.bfloat16, torch.float8_e4m3fn):
    lengths = [25560]
    ratio, heads, dim = 4, 1, 128
    blocks = [n // ratio for n in lengths]
    source = torch.randn(
        (sum(blocks) + 64, heads, dim), device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    table = torch.arange(lengths[0], device="cuda", dtype=torch.int32)[None, :]
    cu = torch.tensor([0, blocks[0]], device="cuda", dtype=torch.int32)

    def old_gather():
        return torch.cat(
            [
                source.index_select(
                    0, table[i, : count * ratio : ratio].long() // ratio
                )
                for i, count in enumerate(blocks)
            ],
            dim=0,
        )

    compare(
        "U1 gather",
        dict(lengths=lengths, heads=heads, dim=dim, dtype=str(dtype)),
        old_gather,
        lambda: gather_qsa_compressed_keys(
            source, table, cu, sum(blocks), max(blocks), ratio
        ),
    )

for rows in (1, 3):
    for strided in (False, True):
        width = 248320
        logits = torch.randn(
            (rows, width + (128 if strided else 0)), device="cuda", dtype=torch.bfloat16
        )[:, :width]
        output = torch.empty(
            (rows, width + (128 if strided else 0)), device="cuda", dtype=torch.float32
        )[:, :width]
        compare(
            "U3 logits cast",
            dict(rows=rows, width=width, strided=strided),
            lambda: output.copy_(logits),
            lambda: cast_to_float32_logits(logits, output),
        )

for rows in (1, 3):
    hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16) * 0.02
    shared = torch.randn_like(hidden)
    routed = torch.randn_like(hidden)
    baseline_out, candidate_out = routed.clone(), routed.clone()

    def old_shared():
        baseline_out.copy_(routed)
        fused_gate_sigmoid_mul_add(hidden, weight, shared, baseline_out)
        return baseline_out

    def new_shared():
        candidate_out.copy_(routed)
        gate = shared_expert_gate(hidden, weight)
        return shared_expert_gate_add(gate, shared, candidate_out)

    compare("C2 shared gate total", dict(rows=rows), old_shared, new_shared)
    gate = shared_expert_gate(hidden, weight)

    def post_shared():
        candidate_out.copy_(routed)
        return shared_expert_gate_add(gate, shared, candidate_out)

    results.append(
        dict(
            component="C2 post-fill consumer",
            rows=rows,
            includes_reset_copy=True,
            timing=gpu_time(post_shared),
        )
    )

    residual = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    normed = torch.randn_like(residual)
    injection = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16) * 0.02
    compare(
        "C1 HC gate total",
        dict(rows=rows),
        lambda: hc_combine_split(hidden, residual, normed, injection, 4, 2560),
        lambda: hc_combine_apply(hidden, residual, hc_combine_gate(normed, injection)),
    )
    partials = hc_combine_gate(normed, injection)
    results.append(
        dict(
            component="C1 post-fill consumer",
            rows=rows,
            timing=gpu_time(lambda: hc_combine_apply(hidden, residual, partials)),
        )
    )


@register_custom_op(
    op_name="qwen38_ablation_eager_add",
    mutates_args=[],
    fake_impl=lambda x: torch.empty_like(x),
)
def eager_add(x: torch.Tensor) -> torch.Tensor:
    return x + 1


x = torch.randn(8, device="cuda")
registered = torch.ops.sglang.qwen38_ablation_eager_add.default
torch.testing.assert_close(eager_add(x), registered(x), rtol=0, atol=0)


def host_time(fn, repeats=1000):
    values = []
    for _ in range(12):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        values.append((time.perf_counter() - start) * 1e6 / repeats)
        torch.cuda.synchronize()
    return stats(values)


results.append(
    dict(
        component="U2 host enqueue",
        legacy=host_time(lambda: registered(x)),
        candidate=host_time(lambda: eager_add(x)),
        parity="bitwise",
    )
)

# Run the real old/new prefill planner with the existing fixture. Only the
# benchmark fixture's return is extended to expose its batch; production is intact.
from sglang.srt.managers import schedule_batch as batch_module

path = REPO / "test/registered/unit/managers/test_mamba_checkpoint_depth.py"
spec = importlib.util.spec_from_file_location("mamba_fixture", path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
tree = ast.parse(path.read_text())
helper = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_prepare_extend"
)
helper.body[-1].value.elts.append(ast.Name(id="batch", ctx=ast.Load()))
ns = dict(vars(fixture))
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
        str(path),
        "exec",
    ),
    ns,
)
_, req, batch = ns["_prepare_extend"](tree_page=64, prefix_len=0, extend_len=8192)
req.kv.mamba_ping_pong_track_buffer = torch.tensor(
    [11, 12], device="cuda", dtype=torch.int64
)
base_source = subprocess.check_output(
    [
        "git",
        "-c",
        "safe.directory=" + str(REPO),
        "show",
        BASE + ":python/sglang/srt/managers/schedule_batch.py",
    ],
    cwd=REPO,
    text=True,
)
old_method = next(
    n
    for n in ast.walk(ast.parse(base_source))
    if isinstance(n, ast.FunctionDef)
    and n.name == "_mamba_radix_cache_v2_req_prepare_for_extend"
)
old_ns = dict(vars(batch_module))
old_ns["_MambaRadixCacheV2TrackEntry"] = namedtuple(
    "LegacyTrackEntry", "track_mask track_index track_seqlen"
)
old_ns["get_parallel"] = lambda: SimpleNamespace(dcp_enabled=False)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[old_method], type_ignores=[])),
        "legacy_schedule_batch.py",
        "exec",
    ),
    old_ns,
)


def old_plan():
    req.kv.mamba_next_track_idx = 0
    return old_ns[old_method.name](batch, req)


def new_plan():
    req.kv.mamba_next_track_idx = 0
    return batch._mamba_radix_cache_v2_req_prepare_for_extend(req)


with patch.object(
    batch_module, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
):
    before, after = old_plan(), new_plan()
    assert (
        before.track_mask == after.track_mask
        and before.track_seqlen == after.track_seqlen
    )
    assert (
        before.track_index
        == req.kv.mamba_ping_pong_track_buffer[after.track_position].item()
    )
    results.append(
        dict(
            component="U4 real prefill planning host",
            legacy=host_time(old_plan, 500),
            candidate=host_time(new_plan, 500),
            parity="same slot and checkpoint",
        )
    )

args.output.write_text(json.dumps(results, indent=2))
print("Saved", args.output, flush=True)
