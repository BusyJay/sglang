# SPDX-License-Identifier: Apache-2.0
"""Check global compaction through actual expert VMM release/growth cycles.

No model or KV allocator: this exercises StagedExpertPool physical mappings.
"""

import collections
import gc
import os
import time
from types import SimpleNamespace


def main():
    os.nice(15)
    import torch
    from build_qwen38_cpu_hybrid import check_temperature

    from sglang.srt.layers.moe.qwen38_hybrid import HybridCache
    from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu
    from sglang.srt.layers.moe.qwen38_staged_pool import StagedExpertPool
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import TensorRecord

    records = {
        name: TensorRecord(0, 0, shape, dtype)
        for name, shape, dtype in (
            ("w13_weight", (8, 1024, 512), "uint8"),
            ("w2_weight", (8, 1024, 256), "uint8"),
            ("w13_blockscale_swizzled", (8, 1024, 64), "float8_e4m3fn"),
            ("w2_blockscale_swizzled", (8, 1024, 32), "float8_e4m3fn"),
            ("g1_alphas", (8,), "float32"),
            ("g2_alphas", (8,), "float32"),
        )
    }
    manifest = SimpleNamespace(num_experts=8, num_layers=4, tensors={0: records})
    pool = StagedExpertPool(manifest, "cuda:0", [4] * 4, 1)
    extra = pool.physical_bytes(32) - pool.physical_bytes(16)
    budgets = pool.configure_loans([extra, extra // 2, 0])
    policy = GlobalLfu(4, 8, [budgets[0] // 4] * 4)
    assert len(policy.keys) == budgets[0]
    for layer, state in enumerate(policy.layers):
        policy.seed(layer, {e: e for e in range(budgets[0] // 4)})
        policy.frequency[layer] = [float((layer * 13 + e * 7) % 19) for e in range(8)]
    for slot in range(budgets[0]):
        for tensor in pool.tensors.values():
            tensor[slot : slot + 1].view(torch.uint8).fill_(slot + 1)
    expected = {key: slot + 1 for slot, key in enumerate(policy.keys)}
    subject = HybridCache.__new__(HybridCache)
    subject.lookups, subject.policy = {}, policy
    subject._seed, subject.drain = lambda: None, lambda: None
    subject.fill_stream = torch.cuda.Stream()
    subject.stats = collections.Counter()
    subject.runtime = SimpleNamespace(
        staged_pool=pool,
        layer_computed={},
        device=torch.device("cuda:0"),
        transfer=torch.cuda.Stream(),
        manifest=manifest,
        _refresh_staged_arena=lambda: None,
        adaptive_migrated_experts=0,
        adaptive_evicted_experts=0,
        poisoned=False,
    )
    try:
        for stage in (1, 2, 0, 2, 1, 0):
            check_temperature()
            if stage < pool.stage_index:
                pool.grow_stage(stage)
                subject.resize(budgets[stage])
            else:
                subject.resize(budgets[stage])
                pool.release_stage(stage)
            for slot, key in enumerate(policy.keys):
                if key is not None:
                    for tensor in subject.pool.values():
                        assert bool(
                            (
                                tensor[slot : slot + 1].view(torch.uint8)
                                == expected[key]
                            ).all()
                        )
            torch.cuda.synchronize()
            print(f"PASS real VMM stage={stage} budget={budgets[stage]}", flush=True)
            time.sleep(0.2)
    finally:
        subject.pool = subject.runtime.cache_pool = None
        subject.runtime.caches = {}
        del tensor
        gc.collect()
        pool.close()


if __name__ == "__main__":
    main()
