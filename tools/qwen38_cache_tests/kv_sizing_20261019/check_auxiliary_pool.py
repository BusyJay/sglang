"""Real GPU check: small arrays share the normal allocator, not VMM pages."""

import json
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.qwen38_staged_pool import StagedExpertPool
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS


def main():
    device = "cuda:1"
    torch.cuda.set_device(device)
    records = {}
    for tiny_alpha in (False, True):
        manifest = SimpleNamespace(
            num_experts=16,
            tensors=[
                {
                    name: SimpleNamespace(
                        shape=(16,)
                        if tiny_alpha and name.startswith("g")
                        else (16, 262144),
                        dtype="float32"
                        if tiny_alpha and name.startswith("g")
                        else "uint8",
                    )
                    for name in REQUIRED_TENSORS
                }
            ],
        )
        pool = StagedExpertPool(manifest, device, [4, 4], 2)
        budgets = pool.configure_loans([24 << 20, 12 << 20, 0])
        pointers = {n: t.data_ptr() for n, t in pool.tensors.items()}
        physical, handles = {}, {}
        for cycle in range(300):
            for stage in (0, 1, 2):
                if stage != pool.stage_index:
                    pool.release_stage(stage)
                budget = budgets[stage]
                if stage == 0:
                    for t in pool.tensors.values():
                        t[:budget].fill_(cycle % 251)
                for name, t in pool.tensors.items():
                    assert t.data_ptr() == pointers[name]
                    assert bool((t[:budget] == cycle % 251).all())
                torch.cuda.synchronize(device)
                backed = pool.arena.backed_bytes
                count = len(pool.arena._allocation._mappings)
                assert backed == physical.setdefault(stage, backed)
                assert count == handles.setdefault(stage, count)
                assert pool.physical_bytes(budget) == backed + pool.auxiliary_bytes
            pool.grow_stage(0)
        # With tiny alpha rows only four weight tensors need VMM mappings.
        assert handles[2] == (4 if tiny_alpha else 6)
        records[str(tiny_alpha)] = dict(
            cycles=300,
            budgets=budgets,
            vmm_bytes=physical,
            mapping_counts=handles,
            auxiliary_bytes=pool.auxiliary_bytes,
            stable_pointers_and_payload=True,
        )
        pool.close()
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
