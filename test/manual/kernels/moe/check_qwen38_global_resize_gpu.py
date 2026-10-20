# SPDX-License-Identifier: Apache-2.0
"""Target GPU global slot movement check, without a model or actual VMM unmap."""

import collections
import os
import time
from types import SimpleNamespace


def main():
    os.nice(15)
    import torch
    from build_qwen38_cpu_hybrid import check_temperature

    from sglang.srt.layers.moe.qwen38_hybrid import HybridCache
    from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu

    policy = GlobalLfu(4, 8, [4] * 4)
    for layer in range(4):
        policy.seed(layer, {e: e for e in range(4)})
        policy.frequency[layer][:4] = [(layer * 13 + e * 7) % 19 for e in range(4)]
    payload = torch.arange(32 * 64, dtype=torch.int32, device="cuda").reshape(32, 64)
    originals = {key: payload[slot].clone() for slot, key in enumerate(policy.keys)}
    pool = SimpleNamespace(pool_views=lambda budget: {"weight": payload[:budget]})
    subject = HybridCache.__new__(HybridCache)
    subject.lookups = {}
    subject.policy = policy
    subject._seed = lambda: None
    subject.drain = lambda: None
    subject.fill_stream = torch.cuda.Stream()
    subject.stats = collections.Counter()
    subject.runtime = SimpleNamespace(
        staged_pool=pool,
        layer_computed={},
        device=torch.device("cuda"),
        transfer=torch.cuda.Stream(),
        manifest=SimpleNamespace(num_layers=4),
        _refresh_staged_arena=lambda: None,
        adaptive_migrated_experts=0,
        adaptive_evicted_experts=0,
        poisoned=False,
    )
    for budget in (11, 20, 6, 16, 4):
        check_temperature()
        subject.resize(budget)
        for slot, key in enumerate(policy.keys):
            if key is not None:
                torch.testing.assert_close(
                    payload[slot], originals[key], rtol=0, atol=0
                )
        torch.cuda.synchronize()
        print(
            f"PASS budget={budget} residents={sum(key is not None for key in policy.keys)}"
        )
        time.sleep(0.2)
    print(
        "GPU byte moves passed. Real VMM/KV loan and serving transitions still require validation."
    )


if __name__ == "__main__":
    main()
