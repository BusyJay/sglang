"""Real E512 pack, fixed routes: split-CUTLASS calculation and peak memory.

All weights are resident. This isolates calculation after route planning, and
makes no claim about copy overlap or whole-service speedup.
"""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.qwen38_cache_buffers import GrowableSwigluBuffers
from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedNvFp4MoeMethod
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
    LAYER_TENSORS,
    REQUIRED_TENSORS,
    Qwen38ExpertPack,
    load_manifest,
)
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.layer_ut_utils import init_single_process_dist

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
base_commit = "a03f36b70d"
source = subprocess.check_output(
    [
        "git",
        "-c",
        "safe.directory=" + str(repo),
        "show",
        base_commit + ":python/sglang/srt/layers/quantization/qwen38_nvfp4_cache.py",
    ],
    cwd=repo,
)
baseline = types.ModuleType("qwen38_prefill_baseline")
baseline.__file__ = "/tmp/qwen38_prefill_baseline.py"
sys.modules[baseline.__name__] = baseline
exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)
init_single_process_dist()
torch.set_grad_enabled(False)
torch.manual_seed(20261026)
manifest = load_manifest(args.manifest)
assert (
    manifest.num_experts,
    manifest.hidden_size,
    manifest.intermediate_size,
    manifest.top_k,
) == (512, 2560, 640, 10)
assert not manifest.scale_encoding
with Qwen38ExpertPack(manifest, source_mode="file_cow") as pack:
    tensors = {}
    for name, record, view in pack.layer_tensors(0):
        tensors[name] = (
            torch.frombuffer(bytearray(view), dtype=getattr(torch, record.dtype))
            .reshape(record.shape)
            .cuda()
        )
        if name in LAYER_TENSORS:
            tensors[name] = tensors[name].reshape(())
        view.release()
cache = {
    name: value[:128] if name in REQUIRED_TENSORS else value
    for name, value in tensors.items()
}
lookup = torch.full((512,), -1, device="cuda", dtype=torch.int32)
lookup[:128] = torch.arange(128, device="cuda", dtype=torch.int32)
placement = SimpleNamespace(moe_tp_size=1, moe_tp_rank=0, moe_ep_size=1, moe_ep_rank=0)
config = MoeRunnerConfig(
    num_experts=512,
    num_local_experts=512,
    hidden_size=2560,
    intermediate_size_per_partition=640,
    layer=placement,
    top_k=10,
    params_dtype=torch.bfloat16,
    inplace=False,
)
rows = []
with (
    get_context().override_server_args(model_path="dummy"),
    get_flags().moe.override(
        runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS,
        a2a_backend=MoeA2ABackend.NONE,
    ),
    get_parallel().override(
        tp_size=1, tp_rank=0, moe_tp_size=1, moe_tp_rank=0, moe_ep_size=1, moe_ep_rank=0
    ),
):
    for count in (984, 2048, 2049, 4101, 8192):
        x = torch.randn(count, 2560, device="cuda", dtype=torch.bfloat16) * 0.1
        ids = torch.rand(count, 512, device="cuda").topk(10, dim=1).indices.int()
        weights = torch.full(ids.shape, 0.1, device="cuda", dtype=torch.float32)
        prepared = PreparedExperts(
            tensors,
            ids,
            512,
            True,
            cache_tensors=cache,
            cache_experts={i: i for i in range(128)},
            arena_experts=list(range(128, 512)),
            cache_lookup=lookup,
            cache_slot_limit=128,
        )
        methods = {}
        for arm, cls in [
            ("old", baseline.Qwen38CachedNvFp4MoeMethod),
            ("new", Qwen38CachedNvFp4MoeMethod),
        ]:
            method = cls(manifest, "fixture")
            method.layer_id = 0
            method.prefill_use_w4a16 = False
            method.moe_runner_config = config
            method.swiglu_buffers = GrowableSwigluBuffers(
                config, torch.device("cuda"), 512
            )
            method.runtime = SimpleNamespace(
                run=lambda _layer, _ids, execute, **kwargs: execute(prepared)
            )
            methods[arm] = method

        def run(arm):
            return methods[arm]._run_experts(
                x, weights, ids, None, execution_phase="prefill"
            )

        reference = run("old").clone()
        numerical = []
        for arm in methods:
            for _ in range(3):
                actual = run(arm)
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
                difference = (actual.float() - reference.float()).norm()
                relative_l2 = (difference / reference.float().norm()).item()
                assert relative_l2 < 0.001, (arm, count, relative_l2)
                numerical.append(dict(arm=arm, relative_l2=relative_l2))
        del actual
        observations = []
        for cycle in range(3):
            for arm in ("old", "new", "new", "old"):
                torch.cuda.synchronize()
                gc.collect()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                started = time.perf_counter()
                actual = run(arm)
                end.record()
                end.synchronize()
                wall = (time.perf_counter() - started) * 1000
                observations.append(
                    dict(
                        arm=arm,
                        gpu_ms=begin.elapsed_time(end),
                        wall_ms=wall,
                        peak_extra_bytes=torch.cuda.max_memory_allocated() - before,
                    )
                )
                del actual
        medians = {
            arm: {
                key: statistics.median(v[key] for v in observations if v["arm"] == arm)
                for key in ("gpu_ms", "wall_ms", "peak_extra_bytes")
            }
            for arm in methods
        }
        result = dict(
            tokens=count,
            medians=medians,
            observations=observations,
            parity="passed rtol=atol=0.02",
            relative_l2_limit=0.001,
            numerical=numerical,
            hits=128,
            source_experts=512,
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "observations"}),
            flush=True,
        )
        rows.append(result)
        args.output.write_text(
            json.dumps(
                dict(
                    baseline_commit=base_commit,
                    source_sha256=hashlib.sha256(
                        (
                            repo
                            / "python/sglang/srt/layers/quantization/qwen38_nvfp4_cache.py"
                        ).read_bytes()
                    ).hexdigest(),
                    results=rows,
                ),
                indent=2,
            )
            + "\n"
        )
