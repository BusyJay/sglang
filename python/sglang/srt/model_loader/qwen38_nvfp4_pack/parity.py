# SPDX-License-Identifier: Apache-2.0
"""SM120 fixed-route cached vs resident CUTLASS comparison; no downloads."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path


def check_exported_layer(
    layer,
    exported,
    mode="sync",
    *,
    slots=56,
    rtol=0.02,
    atol=0.02,
    decode_graph=False,
    graph_width=1,
):
    import torch

    if decode_graph and mode != "threaded":
        raise ValueError("decode graph parity requires threaded mode")
    if graph_width not in (1, 2, 3, 4):
        raise ValueError("graph width must be 1/2/3/4")

    from sglang.srt.layers.moe.qwen38_cache_runtime import (
        Qwen38SynchronousExpertRuntime,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
        Qwen38CachedNvFp4MoeMethod,
    )

    from .format import (
        PACK_TENSORS,
        Qwen38ExpertPack,
        Qwen38ExpertPackManifest,
        TensorRecord,
    )

    exported = Path(exported)
    metadata = json.loads((exported / "metadata.json").read_text())
    # Assemble a compact one-layer fixture without copying the other 47 layers.
    with tempfile.TemporaryDirectory(prefix="qwen38-parity-") as temporary:
        payload = Path(temporary) / "experts.bin"
        records = {}
        import shutil

        with payload.open("wb") as stream:
            for name in PACK_TENSORS:
                info = metadata[name]
                source = exported / info["file"]
                offset = stream.tell()
                with source.open("rb") as tensor_file:
                    shutil.copyfileobj(tensor_file, stream, length=1 << 20)
                records[name] = TensorRecord(
                    offset, stream.tell() - offset, tuple(info["shape"]), info["dtype"]
                )
        config = layer.moe_runner_config
        manifest = Qwen38ExpertPackManifest(
            payload,
            payload.stat().st_size,
            "parity-fixture",
            "parity-fixture",
            "flashinfer_cutlass",
            "sm120",
            1,
            layer.num_experts,
            config.top_k,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            {0: records},
        )
        manifest.validate()
        with Qwen38ExpertPack(manifest) as pack:
            runtime_class = Qwen38SynchronousExpertRuntime
            if mode == "async":
                from sglang.srt.layers.moe.qwen38_cache_async import (
                    Qwen38AsyncExpertRuntime,
                )

                runtime_class = Qwen38AsyncExpertRuntime
            if mode == "device":
                from sglang.srt.layers.moe.qwen38_cache_device import (
                    Qwen38DeviceExpertRuntime,
                )

                runtime_class = Qwen38DeviceExpertRuntime
            if mode == "threaded":
                from sglang.srt.layers.moe.qwen38_cache_threaded import (
                    Qwen38ThreadedExpertRuntime,
                )

                runtime_class = Qwen38ThreadedExpertRuntime
            runtime = runtime_class(
                pack, "cuda", slots_per_layer=min(slots, layer.num_experts)
            )
            runtime.reserve()
            cached = Qwen38CachedNvFp4MoeMethod(manifest, "model.layers.0.mlp.experts")
            cached.layer_id = 0
            cached.create_moe_runner(layer, replace(config, layer_id=0))
            cached.bind_runtime(runtime)
            reports = []
            generator = torch.Generator(device="cuda").manual_seed(731)
            graph = None
            # Miss, repeat hit, disjoint routes, then enough unique IDs for arena.
            cases = [
                ("decode_miss", 1, 0),
                ("decode_hit", 1, 0),
                ("decode_eviction", 1, config.top_k),
                (
                    "prefill_arena",
                    max(2, (min(slots, layer.num_experts) // config.top_k) + 2),
                    0,
                ),
            ]
            # Force more than capacity distinct decode routes before the arena case.
            capacity = min(slots, layer.num_experts)
            if capacity < layer.num_experts:
                cases[2:3] = [
                    (f"decode_fill_{offset}", 1, offset)
                    for offset in range(
                        config.top_k, capacity + config.top_k, config.top_k
                    )
                ]
            if decode_graph and graph_width > 1:
                cases += [
                    ("verify_miss", graph_width, 0),
                    ("verify_shift", graph_width, config.top_k),
                    ("verify_revisit", graph_width, 0),
                ]
            if decode_graph:
                cases += [
                    ("decode_after_arena", 1, layer.num_experts - config.top_k),
                    ("decode_revisit", 1, 0),
                    ("decode_revisit_hit", 1, 0),
                ]
            try:
                for label, tokens, offset in cases:
                    ids = (
                        (
                            (
                                torch.arange(tokens * config.top_k, device="cuda")
                                + offset
                            )
                            % layer.num_experts
                        )
                        .reshape(tokens, config.top_k)
                        .to(torch.int32)
                    )
                    weights = torch.rand(
                        tokens, config.top_k, device="cuda", generator=generator
                    )
                    weights /= weights.sum(-1, keepdim=True)
                    hidden = (
                        torch.randn(
                            tokens,
                            layer.hidden_size,
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=generator,
                        )
                        / 10
                    )
                    topk = StandardTopKOutput(
                        weights,
                        ids,
                        torch.empty(tokens, layer.num_experts, device="cuda"),
                    )
                    # Clone inputs independently, as kernels may modify workspaces.
                    resident = layer.quant_method.apply(
                        layer, StandardDispatchOutput(hidden.clone(), None, topk)
                    ).hidden_states
                    if decode_graph and tokens == graph_width:
                        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
                            BreakableCUDAGraph,
                            BreakableCUDAGraphCapture,
                        )

                        if graph is None:
                            static_hidden, static_weights, static_ids = (
                                hidden.clone(),
                                weights.clone(),
                                ids.clone(),
                            )
                            static_topk = StandardTopKOutput(
                                static_weights, static_ids, topk.router_logits
                            )
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                for _ in range(2):
                                    cached.apply(
                                        layer,
                                        StandardDispatchOutput(
                                            static_hidden, None, static_topk
                                        ),
                                    )
                            stream.synchronize()
                            graph = BreakableCUDAGraph()
                            capture_runs = runtime.run_invocations
                            with BreakableCUDAGraphCapture(graph, stream=stream):
                                # Exercise upstream and downstream GPU segments,
                                # not just an eager callback in an empty graph.
                                graph_hidden = static_hidden * 1.0
                                graph_topk = StandardTopKOutput(
                                    static_weights * 1.0,
                                    static_ids + 0,
                                    static_topk.router_logits,
                                )
                                graph_output = (
                                    cached._run_experts(
                                        graph_hidden,
                                        graph_topk.topk_weights,
                                        graph_topk.topk_ids,
                                        graph_topk.router_logits,
                                        execution_phase="verify"
                                        if graph_width > 1
                                        else "decode",
                                    )
                                    * 1.0
                                )
                            if runtime.run_invocations != capture_runs:
                                raise AssertionError(
                                    "capture enqueued host transport using stale routes"
                                )
                            torch.cuda.current_stream().wait_stream(stream)
                        static_hidden.copy_(hidden)
                        static_weights.copy_(weights)
                        static_ids.copy_(ids)
                        before = runtime.run_invocations
                        graph.replay()
                        if runtime.run_invocations != before + 1:
                            raise AssertionError(
                                "replay did not enqueue fresh host transport"
                            )
                        actual = graph_output
                    else:
                        actual = cached.apply(
                            layer, StandardDispatchOutput(hidden.clone(), None, topk)
                        ).hidden_states
                    torch.cuda.synchronize()
                    if (
                        not torch.isfinite(actual).all()
                        or not torch.isfinite(resident).all()
                    ):
                        raise AssertionError(f"nonfinite CUTLASS output in {label}")
                    torch.testing.assert_close(actual, resident, rtol=rtol, atol=atol)
                    reports.append(
                        {
                            "case": label,
                            "tokens": tokens,
                            "max_abs_error": (actual.float() - resident.float())
                            .abs()
                            .max()
                            .item(),
                        }
                    )
                return {
                    "status": "passed",
                    "runtime_mode": mode,
                    "decode_graph": decode_graph,
                    "graph_width": graph_width,
                    "rtol": rtol,
                    "atol": atol,
                    "cases": reports,
                    "hits": runtime.hits,
                    "misses": runtime.misses,
                    "arena_loads": runtime.arena_loads,
                    "memory_budget": runtime.memory_budget(),
                }
            finally:
                if hasattr(runtime, "close"):
                    runtime.close()
                elif hasattr(runtime, "synchronize"):
                    runtime.synchronize()
                torch.cuda.synchronize()
