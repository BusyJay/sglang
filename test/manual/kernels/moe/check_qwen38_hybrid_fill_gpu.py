# SPDX-License-Identifier: Apache-2.0
"""Check direct registered DMA and staged global fills against immutable bytes."""

import collections
import time
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
        direct = False
        done = torch.cuda.Event()
        done.record()
        batches = [[(2, 3), (0, 1)], [(3, 0), (1, 2)]]
        pending = [subject._copy(0, fills, done) for fills in batches]
        deadline = time.monotonic() + 10
        while pending:
            pending = [copied for copied in pending if not copied.query()]
            if time.monotonic() > deadline:
                raise TimeoutError("staged copies did not retire")
            time.sleep(0.0001)
        for fills in batches:
            for expert, slot in fills:
                for name, tensor in subject.pool.items():
                    record = records[name]
                    count = record.nbytes // 4
                    begin = record.offset + expert * count
                    torch.testing.assert_close(
                        tensor[slot].cpu(),
                        source[begin : begin + count],
                        rtol=0,
                        atol=0,
                    )
        print("PASS concurrent staged buffer leases", flush=True)
    finally:
        subject.fill_stream.synchronize()
        _module().mailbox_unregister(source)


def check_reserved_resize():
    from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu

    policy = GlobalLfu(2, 8, [8, 8])
    for layer in range(2):
        policy.seed(layer, {e: e for e in range(8)})
        policy.frequency[layer] = [float(e + 8 * layer) for e in range(8)]
    payload = torch.arange(16 * 37, device="cuda").to(torch.uint8).reshape(16, 37)
    expected = {key: payload[slot].clone() for slot, key in enumerate(policy.keys)}
    for budget in (16, 12, 16, 10, 15):
        reserved = {0, *range(budget - 4, budget)}
        plan = policy.plan_resize(budget, reserved=reserved)
        for old, new in plan.moves:
            payload[new].copy_(payload[old])
        torch.cuda.synchronize()
        policy.commit_resize(plan)
        for slot, key in enumerate(policy.keys):
            if key is not None:
                assert slot not in reserved
                torch.testing.assert_close(payload[slot], expected[key], rtol=0, atol=0)
        print(
            f"PASS reserved resize budget={budget} moves={len(plan.moves)}", flush=True
        )


def check_hybrid_dispatch():
    # Real D2H/CPU/H2D and route partitioning, including repeated buffer reuse.
    import queue
    import threading

    for tokens in (1, 3):
        for mode in ("cpu", "hit", "all_cpu"):
            subject = HybridCache.__new__(HybridCache)
            subject._seed = lambda: None
            mapping = {} if mode == "all_cpu" else {0: 1, 1: 2}
            subject.policy = SimpleNamespace(keys=[None] * 8)
            subject.timeout = 10
            subject.host_buffers, subject.lookups = {}, {}
            subject.stats = collections.Counter()
            subject.failure = None
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

            def cpu(layer, x, ids, weights, result):
                value = ((ids + 1).clamp_min(0).float() * weights).sum(-1, keepdim=True)
                result.copy_(value.expand_as(x))

            subject._gpu = gpu
            subject.cpu = SimpleNamespace(run=cpu, release_worker=lambda: None)
            subject.cpu_jobs = queue.Queue()
            worker = threading.Thread(target=subject._cpu_work)
            worker.start()
            try:
                for iteration in range(3):
                    subject.current = (0, iteration + 1, mapping)
                    route = [0, 1, -1, -1] if mode == "hit" else [0, 1, 2, -1]
                    ids = torch.tensor(
                        [route] * tokens, dtype=torch.int32, device="cuda"
                    )
                    x = torch.zeros((tokens, 32), dtype=torch.bfloat16, device="cuda")
                    weights = torch.full_like(ids, iteration + 1, dtype=torch.float32)
                    output = subject.run(0, x, ids, weights)
                    expected = (3 if mode == "hit" else 6) * (iteration + 1)
                    torch.testing.assert_close(
                        output, torch.full_like(output, expected), rtol=0, atol=0
                    )
                assert len(calls) == (0 if mode == "all_cpu" else 3)
                assert subject.stats["cpu_calls"] == (0 if mode == "hit" else 3)
                print(f"PASS hybrid tokens={tokens} mode={mode}", flush=True)
            finally:
                subject.cpu_jobs.put(None)
                worker.join(timeout=10)
                assert not worker.is_alive()


if __name__ == "__main__":
    main()
    check_reserved_resize()
    check_hybrid_dispatch()
