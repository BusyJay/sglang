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
        device=torch.device("cuda"),
        manifest=SimpleNamespace(num_experts=4, tensors={0: records}),
        miss_mailbox=SimpleNamespace(pack_view=source),
        _registered_prefill_enabled=lambda: direct,
    )
    mapped = _module().mailbox_register(source)
    from cuda.bindings import driver

    from sglang.srt.utils.cuda_vmm_utils import check_drv

    failed = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    failed_address = int(
        check_drv(
            driver.cuMemHostGetDevicePointer(failed.data_ptr(), 0), "test failure flag"
        )
    )
    try:
        for direct in (False, True, "mailbox"):
            if direct == "mailbox":
                subject.runtime.miss_mailbox.pack_address = mapped
                subject.runtime.miss_mailbox.failed_address = failed_address
                subject.runtime.miss_mailbox.fill_cta = 0
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


def check_arena_seed():
    for dtype in (torch.uint8, torch.float8_e4m3fn, torch.bfloat16):
        for borrowed in (False, True):
            for fills in ([], [(2, 3), (0, 6)], [(3, 1), (1, 7)]):
                subject = HybridCache.__new__(HybridCache)
                subject.pool = {
                    name: torch.zeros((12, width), dtype=dtype, device="cuda")
                    for name, width in (("weights", 1024), ("alpha", 4))
                }
                arena = {
                    name: tensor[-4:] if borrowed else torch.zeros_like(tensor[:4])
                    for name, tensor in subject.pool.items()
                }
                for tensor in arena.values():
                    raw = tensor.view(torch.uint8)
                    raw.copy_(
                        torch.arange(raw.numel(), device="cuda")
                        .reshape(raw.shape)
                        .to(torch.uint8)
                    )
                expected = {
                    name: tensor.clone().view(torch.uint8)
                    for name, tensor in subject.pool.items()
                }
                for expert, slot in fills:
                    for name in expected:
                        expected[name][slot].copy_(
                            arena[name][expert].view(torch.uint8)
                        )
                subject.policy = SimpleNamespace(keys=[None] * 12)
                subject.runtime = SimpleNamespace(
                    device=torch.device("cuda"),
                    manifest=SimpleNamespace(num_experts=4, num_layers=1),
                    arena=arena,
                )
                subject.fill_stream = torch.cuda.Stream()
                subject.stats = collections.Counter()
                done = torch.cuda.Event()
                done.record()
                subject._seed_from_arena(0, fills, done).synchronize()
                for name, tensor in subject.pool.items():
                    torch.testing.assert_close(
                        tensor.view(torch.uint8), expected[name], rtol=0, atol=0
                    )
                assert subject.stats["fill_bytes"] == 0
                print(
                    f"PASS GPU arena seed dtype={dtype} borrowed={borrowed} fills={fills}",
                    flush=True,
                )


def check_crossover_dispatch():
    # Exercise the actual runtime dispatch with a linear GPU oracle, including
    # deferred CPU, full-GPU, and all-hit windows and the eager bootstrap path.
    for threshold in (1, 3, 33):
        for tokens in (1, 3):
            for mode in ("gpu", "cpu", "hit"):
                subject = HybridCache.__new__(HybridCache)
                subject._seed = subject._wait_arena = lambda: None
                subject._wait_layer = lambda _: None
                mapping = {0: 1, 1: 2}
                subject.current = (0, 1, mapping)
                subject.policy = SimpleNamespace(keys=[None] * 8)
                subject.threshold = threshold
                subject.timeout = 10
                subject.host_buffers, subject.lookups = {}, {}
                subject.stats = collections.Counter()
                subject.readback = torch.cuda.Stream()
                subject.result_stream = torch.cuda.Stream()
                subject.runtime = SimpleNamespace(
                    device=torch.device("cuda"),
                    manifest=SimpleNamespace(num_experts=4, num_layers=1),
                    layer_computed={},
                    _record_invocation=lambda *args: None,
                    finish_layer=lambda _: None,
                )
                calls = []

                def gpu(x, slots, weights, layer):
                    calls.append(slots.clone())
                    value = (slots.clamp_min(0).float() * weights).sum(-1, keepdim=True)
                    return value.expand_as(x).contiguous()

                subject._gpu = gpu

                def submit(job):
                    if mode == "cpu":
                        job[6].fill_(5)
                    if job[-2] is not None:
                        job[-2].set_result(mode)
                    job[-1].set_result(
                        (
                            mode,
                            {2} if mode != "hit" else set(),
                            {0: 1, 1: 2, 2: 3} if mode == "gpu" else None,
                        )
                    )

                subject.cpu_jobs = SimpleNamespace(put_nowait=submit)
                ids = torch.tensor(
                    [[0, 1, 2, -1]] * tokens, dtype=torch.int32, device="cuda"
                )
                x = torch.zeros((tokens, 32), dtype=torch.bfloat16, device="cuda")
                weights = torch.ones_like(ids, dtype=torch.float32)
                output = subject.run(0, x, ids, weights)
                expected = 6 if mode == "gpu" else 8 if mode == "cpu" else 3
                torch.testing.assert_close(
                    output, torch.full_like(output, expected), rtol=0, atol=0
                )
                assert len(calls) == (
                    2 if mode == "gpu" and threshold > ids.numel() else 1
                )
                print(
                    f"PASS crossover threshold={threshold} tokens={tokens} mode={mode}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
    check_arena_seed()
    check_crossover_dispatch()
