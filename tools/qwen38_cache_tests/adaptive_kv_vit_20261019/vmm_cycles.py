"""Exercise real CUDA VMM expert mappings through repeated grow/return cycles."""

import json
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.qwen38_staged_pool import StagedExpertPool
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS


def main():
    device = "cuda:1"
    torch.cuda.set_device(device)
    manifest = SimpleNamespace(
        num_experts=16,
        tensors=[
            {
                n: SimpleNamespace(shape=(16, 262144), dtype="uint8")
                for n in REQUIRED_TENSORS
            }
        ],
    )
    pool = StagedExpertPool(manifest, device, [4, 4], 2)
    budgets = pool.configure_loans([24 << 20, 12 << 20, 0])
    addresses = {n: t.data_ptr() for n, t in pool.tensors.items()}
    expected_bytes = {}
    handles = {}
    for cycle in range(300):
        for stage in (0, 1, 2):
            if stage != pool.stage_index:
                pool.release_stage(stage)
            budget = budgets[stage]
            if stage == 0:
                for t in pool.tensors.values():
                    t[:budget].fill_(cycle % 251)
            for n, t in pool.tensors.items():
                assert t.data_ptr() == addresses[n]
                assert bool((t[:budget] == cycle % 251).all()), (cycle, stage, n)
            torch.cuda.synchronize(device)
            physical = pool.arena.backed_bytes
            count = len(pool.arena._allocation._mappings)
            assert physical == expected_bytes.setdefault(stage, physical)
            assert count == handles.setdefault(stage, count)
        pool.grow_stage(0)
    print(
        json.dumps(
            {
                "cycles": 300,
                "budgets": budgets,
                "backed_bytes": expected_bytes,
                "mapping_counts": handles,
                "retained_payload_checks": "passed",
                "memory_stats": torch.cuda.memory_stats(device),
            },
            indent=2,
        )
    )
    pool.close()


if __name__ == "__main__":
    main()
