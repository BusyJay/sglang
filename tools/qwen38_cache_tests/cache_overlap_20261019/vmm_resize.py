"""Real expert pack: migrate two layers before physically unmapping VMM tails."""

import json
import time
from dataclasses import replace

import torch

from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
    REQUIRED_TENSORS,
    Qwen38ExpertPack,
    load_manifest,
)

path = "/model-pack/manifest.json"
manifest = load_manifest(path)
manifest = replace(
    manifest, num_layers=2, tensors={i: manifest.tensors[i] for i in range(2)}
)
with Qwen38ExpertPack(manifest, source_mode="file_cow") as pack:
    runtime = Qwen38ThreadedExpertRuntime(
        pack,
        "cuda:0",
        slots_per_layer=20,
        prefill_arena=False,
        decode_device_gate=False,
        adaptive_cache=True,
        adaptive_cache_mode="sample",
        preserve_cache_on_resize=True,
    )
    runtime.kv_cache_stages = [80000, 160000, 220000]
    runtime.reserve()
    try:
        budgets = runtime.staged_pool.configure_loans([128 << 20, 64 << 20, 0])
        runtime.initialize_staged_layout()
        rows = []
        for cycle in range(5):
            for layer in range(2):
                for offset in (0, 10, 20, 20, 20):
                    ids = torch.arange(
                        offset, offset + 10, device="cuda:0", dtype=torch.int32
                    )[None]
                    runtime.run(
                        layer, ids, lambda prepared: None, execution_phase="decode"
                    )
            runtime.synchronize()
            before = runtime.staged_pool.arena.backed_bytes
            start = time.perf_counter()
            runtime.pending_kv_stage = 2
            runtime.staged_trim_enqueued = False
            runtime.enqueue_staged_trim()
            torch.cuda.synchronize()
            runtime.complete_staged_trim()
            runtime.staged_pool.release_stage(2)
            after = runtime.staged_pool.arena.backed_bytes
            assert after < before
            for layer in range(2):
                state = runtime.device_states[layer]
                keys = state["keys"].cpu().tolist()
                assert set(range(20, 30)).issubset(keys), keys
                for slot, expert in enumerate(keys):
                    if expert < 0:
                        continue
                    for name in REQUIRED_TENSORS:
                        expected = torch.frombuffer(
                            pack.expert_tensor_bytes(layer, expert, name),
                            dtype=torch.uint8,
                        ).clone()
                        actual = runtime.caches[layer][name][slot].cpu()
                        assert torch.equal(
                            actual.reshape(-1).view(torch.uint8), expected
                        ), (layer, slot, expert, name)
            runtime.staged_pool.grow_stage(0)
            runtime.grow_staged_layout(0)
            runtime.synchronize()
            rows.append(
                dict(
                    cycle=cycle,
                    before_bytes=before,
                    after_bytes=after,
                    resize_and_byte_validation_seconds=time.perf_counter() - start,
                )
            )
        print(
            json.dumps(
                dict(
                    status="passed",
                    budgets=budgets,
                    cycles=rows,
                    scope="two real layers, real VMM unmapping; no full-model inference",
                ),
                indent=2,
            )
        )
    finally:
        runtime.close()
