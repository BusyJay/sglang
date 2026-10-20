# SPDX-License-Identifier: Apache-2.0
"""Check direct registered DMA and staged global fills against immutable bytes."""

import collections
from types import SimpleNamespace

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import _module
from sglang.srt.layers.moe.qwen38_hybrid import HybridCache


def main():
    shapes = {"weights": (4, 1024), "alpha": (4, 4)}
    payload = bytearray(
        (i * 37) % 256 for i in range(sum(a * b for a, b in shapes.values()))
    )
    source = torch.frombuffer(payload, dtype=torch.uint8)
    records, offset = {}, 0
    for name, shape in shapes.items():
        count = shape[0] * shape[1]
        records[name] = SimpleNamespace(offset=offset, nbytes=count)
        offset += count
    subject = HybridCache.__new__(HybridCache)
    subject.pool = {
        name: torch.zeros(shape, dtype=torch.uint8, device="cuda")
        for name, shape in shapes.items()
    }
    subject.fill_stream = torch.cuda.Stream()
    subject.staging = {
        name: torch.empty(shape[1], dtype=torch.uint8, pin_memory=True)
        for name, shape in shapes.items()
    }
    subject.cpu = SimpleNamespace(mapping=payload)
    subject.stats = collections.Counter()
    direct = False
    subject.runtime = SimpleNamespace(
        manifest=SimpleNamespace(num_experts=4, tensors={0: records}),
        miss_mailbox=SimpleNamespace(pack_view=source),
        _registered_prefill_enabled=lambda: direct,
    )
    _module().mailbox_register(source)
    try:
        for direct in (False, True):
            for fills in ([], [(2, 3), (0, 1)], [(3, 0), (1, 2)]):
                done = torch.cuda.Event()
                done.record()
                copied = subject._copy(0, fills, done)
                copied.synchronize()
                for expert, slot in fills:
                    for name, tensor in subject.pool.items():
                        record = records[name]
                        count = record.nbytes // 4
                        expected = source[
                            record.offset + expert * count : record.offset
                            + (expert + 1) * count
                        ]
                        torch.testing.assert_close(
                            tensor[slot].cpu(), expected, rtol=0, atol=0
                        )
                print(f"PASS direct={direct} fills={fills}", flush=True)
    finally:
        subject.fill_stream.synchronize()
        _module().mailbox_unregister(source)


if __name__ == "__main__":
    main()
