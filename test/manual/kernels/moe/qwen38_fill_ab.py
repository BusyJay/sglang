#!/usr/bin/env python3
"""Exact-byte mailbox CTA vs registered DMA, including optional plan readback.

No model/service is changed. Plans are synthetic unless --plans supplies real
miss pairs. Bulk copy is opt-in and measures the full-layer pipeline bandwidth
budget; it is not a full-model pipeline implementation.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from cuda.bindings import driver

from sglang.kernels.ops.moe.qwen38_expert_cache import _module


def measure(fn, iterations):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    wall = time.perf_counter()
    for _ in range(iterations):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return dict(
        median_gpu_ms=statistics.median(samples),
        p95_gpu_ms=sorted(samples)[int(0.95 * (len(samples) - 1))],
        wall_ms_per_call=(time.perf_counter() - wall) * 1000 / iterations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--plans", type=Path, help="JSON list of lists of [expert,slot] pairs; slot <32"
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--overlap-rows",
        nargs="+",
        type=int,
        help="Optional independent BF16 GEMM rows, e.g. 16 128 1024; measure fill/compute contention",
    )
    parser.add_argument(
        "--prefill-sweep",
        action="store_true",
        help="Compare full-layer DMA with contiguous/fragmented sparse runs (1.32 GiB GPU)",
    )
    parser.add_argument(
        "--bulk-layer",
        action="store_true",
        help="Also allocate 1.32 GiB GPU full-layer buffer",
    )
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.iterations < 5 or args.rounds < 1:
        parser.error("iterations must be >=5 and rounds >=1")
    if args.overlap_rows and min(args.overlap_rows) < 1:
        parser.error("overlap rows must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("SM12x required")
    e, h, n, slots = 512, 2560, 640, 32
    sizes = [2 * n * h // 2, 2 * n * (h // 16), h * n // 2, h * (n // 16)]
    sizes = [int(v) for v in sizes]
    source = torch.empty(e * sum(sizes), dtype=torch.uint8, device="cpu")
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset += e * size
    # Distinguish all 512 experts, all four tensors and offsets within a row.
    # A constant across experts would miss wrong source addresses.
    pattern = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    for segment, (off, size) in enumerate(zip(offsets, sizes)):
        rows = source[off : off + e * size].view(e, -1, 256)
        rows.copy_(pattern)
        for expert in range(e):
            rows[expert, :, 0] = expert % 256
            rows[expert, :, 1] = expert // 256
            rows[expert, :, 2] = segment
    targets = [
        torch.zeros(slots * size, dtype=torch.uint8, device="cuda") for size in sizes
    ]
    module = _module()
    address = module.mailbox_register(source)
    try:
        failed = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        status, failure_address = driver.cuMemHostGetDevicePointer(failed.data_ptr(), 0)
        if status != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(status)
        layout = torch.tensor(
            [
                [target.data_ptr(), size, address + off]
                for target, size, off in zip(targets, sizes, offsets)
            ],
            device="cuda",
            dtype=torch.int64,
        )
        plan_host = torch.zeros(4 + 2 * 32, dtype=torch.int32, pin_memory=True)
        plan_gpu = plan_host.to("cuda")
        plan_readback = torch.empty_like(plan_host, pin_memory=True)
        if args.plans:
            plans = json.loads(args.plans.read_text())
        else:
            plans = [
                [[expert, expert] for expert in range(count)]
                for count in (0, 1, 2, 3, 4, 6, 8, 10, 20, 30)
            ]
            plan_rng = random.Random(args.seed)
            for count in (1, 2, 3, 4, 6, 8, 10, 20, 30):
                plans.append(
                    list(
                        zip(
                            plan_rng.sample(range(e), count),
                            plan_rng.sample(range(slots), count),
                        )
                    )
                )
        if not plans:
            raise ValueError("empty plan list")
        records = []
        overlap = []
        copy_stream = torch.cuda.Stream() if args.overlap_rows else None
        compute_cases = []
        if args.overlap_rows:
            matrix = torch.randn(h, h, device="cuda", dtype=torch.bfloat16)
            for rows in args.overlap_rows:
                x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
                compute_cases.append((rows, x, matrix, torch.empty_like(x)))
            torch.cuda.synchronize()
        variants = [("mapped_cta_" + str(cta), cta) for cta in range(1, 9)] + [
            ("mapped_adaptive", 0),
            ("dma_known_plan", None),
            ("dma_coalesced_plan", None),
            ("dma_plan_readback", None),
        ]
        rng = random.Random(args.seed)
        for index, pairs in enumerate(plans):
            if len(pairs) > 30 or len({s for _, s in pairs}) != len(pairs):
                raise ValueError("invalid miss count or duplicate slot")
            if any(not 0 <= ex < e or not 0 <= slot < slots for ex, slot in pairs):
                raise ValueError("plan IDs out of bounds")
            plan_host.zero_()
            plan_host[1] = len(pairs)
            if pairs:
                plan_host[4 : 4 + 2 * len(pairs)] = torch.tensor(
                    pairs, dtype=torch.int32
                ).flatten()
            plan_gpu.copy_(plan_host)
            torch.cuda.synchronize()
            descriptors = [
                (
                    target[slot * size : (slot + 1) * size],
                    source[off + expert * size : off + (expert + 1) * size],
                )
                for expert, slot in pairs
                for target, size, off in zip(targets, sizes, offsets)
            ]
            coalesced = []
            for expert, slot in sorted(pairs):
                if (
                    coalesced
                    and expert == coalesced[-1][0] + coalesced[-1][2]
                    and slot == coalesced[-1][1] + coalesced[-1][2]
                ):
                    coalesced[-1][2] += 1
                else:
                    coalesced.append([expert, slot, 1])
            merged_descriptors = [
                (
                    target[slot * size : (slot + count) * size],
                    source[off + expert * size : off + (expert + count) * size],
                )
                for expert, slot, count in coalesced
                for target, size, off in zip(targets, sizes, offsets)
            ]
            order = []
            for repeat in range(args.rounds):
                current_order = list(variants)
                rng.shuffle(current_order)
                order.extend((repeat, name, cta) for name, cta in current_order)
            for repeat, name, cta in order:

                def run():
                    if cta is not None:
                        module.mailbox_fill_grid(
                            plan_gpu, layout, 30, int(failure_address), cta
                        )
                    else:
                        if name == "dma_plan_readback":
                            plan_readback.copy_(plan_gpu, non_blocking=True)
                            torch.cuda.current_stream().synchronize()
                            current = plan_readback.tolist()
                            dynamic = [
                                (current[4 + 2 * i], current[5 + 2 * i])
                                for i in range(current[1])
                            ]
                            current_descriptors = [
                                (
                                    target[slot * size : (slot + 1) * size],
                                    source[
                                        off + expert * size : off + (expert + 1) * size
                                    ],
                                )
                                for expert, slot in dynamic
                                for target, size, off in zip(targets, sizes, offsets)
                            ]
                        elif name == "dma_coalesced_plan":
                            current_descriptors = merged_descriptors
                        else:
                            current_descriptors = descriptors
                        for dst, src in current_descriptors:
                            module.registered_copy(dst, src)

                for target in targets:
                    target.zero_()
                run()
                torch.cuda.synchronize()
                if failed.item():
                    raise RuntimeError("mailbox fill reported failure")
                for target, size, off in zip(targets, sizes, offsets):
                    expected = torch.zeros_like(target)
                    for expert, slot in pairs:
                        expected[slot * size : (slot + 1) * size].copy_(
                            source[off + expert * size : off + (expert + 1) * size]
                        )
                    torch.testing.assert_close(target, expected, atol=0, rtol=0)
                timing = measure(run, args.iterations)
                records.append(
                    dict(
                        plan=index,
                        repeat=repeat,
                        misses=len(pairs),
                        variant=name,
                        payload_bytes=len(pairs) * sum(sizes),
                        expert_slot_pairs=pairs,
                        dma_calls=(
                            (
                                len(merged_descriptors)
                                if name == "dma_coalesced_plan"
                                else len(descriptors)
                            )
                            if cta is None
                            else None
                        ),
                        **timing,
                    )
                )
                print("passed", index, name, len(pairs), timing, flush=True)
                if (
                    copy_stream is not None
                    and len(pairs) in (2, 3, 10)
                    and name
                    in (
                        "mapped_cta_8",
                        "mapped_adaptive",
                        "dma_known_plan",
                        "dma_plan_readback",
                        "dma_coalesced_plan",
                    )
                ):
                    for rows, x, matrix, out in compute_cases:
                        ready, done = torch.cuda.Event(), torch.cuda.Event()

                        def serial():
                            run()
                            torch.mm(x, matrix, out=out)

                        def concurrent():
                            compute_stream = torch.cuda.current_stream()
                            ready.record(compute_stream)
                            torch.mm(x, matrix, out=out)
                            with torch.cuda.stream(copy_stream):
                                copy_stream.wait_event(ready)
                                run()
                                done.record(copy_stream)
                            compute_stream.wait_event(done)

                        isolated = measure(
                            lambda: torch.mm(x, matrix, out=out), args.iterations
                        )
                        serial_timing = measure(serial, args.iterations)
                        concurrent_timing = measure(concurrent, args.iterations)
                        overlap.append(
                            dict(
                                plan=index,
                                misses=len(pairs),
                                variant=name,
                                repeat=repeat,
                                gemm_rows=rows,
                                compute=isolated,
                                serial=serial_timing,
                                concurrent=concurrent_timing,
                            )
                        )
                        print(
                            "overlap",
                            index,
                            name,
                            rows,
                            serial_timing,
                            concurrent_timing,
                            flush=True,
                        )
        bulk = None
        prefill = []
        if args.bulk_layer or args.prefill_sweep:
            # One contiguous layer in pack's tensor-major layout.
            full = torch.empty_like(source, device="cuda")
            bulk = measure(
                lambda: module.registered_copy(full, source), args.iterations
            )
            torch.testing.assert_close(full.cpu(), source, atol=0, rtol=0)
            bulk.update(
                layer_bytes=source.numel(),
                round_48_layers_bytes=source.numel() * 48,
                sustained_GB_s=source.numel() / bulk["median_gpu_ms"] / 1e6,
                min_48_layer_copy_ms=bulk["median_gpu_ms"] * 48,
                two_layer_compute_window_ms=2 * (22.873349 - 11.065178) / 48,
                required_GB_s_for_26_263ms_round=source.numel() * 48 / 0.026263 / 1e9,
            )
            if args.prefill_sweep:
                for count in (64, 128, 256, 384, 448, 512):
                    choices = dict(
                        contiguous=list(range(count)),
                        fragmented=sorted(rng.sample(range(e), count)),
                    )
                    for pattern_name, experts in choices.items():
                        runs = []
                        for expert in experts:
                            if runs and expert == runs[-1][0] + runs[-1][1]:
                                runs[-1][1] += 1
                            else:
                                runs.append([expert, 1])
                        variants_pf = ["sparse_rows", "sparse_runs", "dense"]
                        for repeat in range(args.rounds):
                            rng.shuffle(variants_pf)
                            for variant in variants_pf:
                                selected = (
                                    [(0, e)]
                                    if variant == "dense"
                                    else (
                                        runs
                                        if variant == "sparse_runs"
                                        else [(ex, 1) for ex in experts]
                                    )
                                )
                                desc = [
                                    (
                                        full[
                                            off + first * size : off
                                            + (first + length) * size
                                        ],
                                        source[
                                            off + first * size : off
                                            + (first + length) * size
                                        ],
                                    )
                                    for first, length in selected
                                    for off, size in zip(offsets, sizes)
                                ]

                                def copy_prefill():
                                    for dst, src in desc:
                                        module.registered_copy(dst, src)

                                full.zero_()
                                copy_prefill()
                                torch.cuda.synchronize()
                                expected = torch.zeros_like(source)
                                for first, length in selected:
                                    for off, size in zip(offsets, sizes):
                                        expected[
                                            off + first * size : off
                                            + (first + length) * size
                                        ].copy_(
                                            source[
                                                off + first * size : off
                                                + (first + length) * size
                                            ]
                                        )
                                torch.testing.assert_close(
                                    full.cpu(), expected, atol=0, rtol=0
                                )
                                timing = measure(copy_prefill, args.iterations)
                                prefill.append(
                                    dict(
                                        experts=count,
                                        pattern=pattern_name,
                                        runs=len(runs),
                                        variant=variant,
                                        repeat=repeat,
                                        dma_calls=len(desc),
                                        payload_bytes=sum(
                                            src.numel() for _, src in desc
                                        ),
                                        **timing,
                                    )
                                )
                                print(
                                    "prefill passed",
                                    count,
                                    pattern_name,
                                    variant,
                                    repeat,
                                    timing,
                                    flush=True,
                                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    status="passed",
                    gpu=torch.cuda.get_device_name(),
                    capability=list(torch.cuda.get_device_capability()),
                    torch=torch.__version__,
                    sizes=sizes,
                    source_bytes=source.numel(),
                    plans="synthetic" if not args.plans else str(args.plans),
                    seed=args.seed,
                    rounds=args.rounds,
                    reports=records,
                    bulk=bulk,
                    prefill=prefill,
                    overlap=overlap,
                    limits=[
                        "DMA known plan omits route readback and descriptor generation",
                        "DMA readback includes synchronous readback but omits production worker scheduling",
                        "Per-case isolated microbench, not service speedup or achieved overlap",
                        "Full-layer copy uses single buffer; more buffers do not reduce total PCIe bytes",
                        "Prefill sweep uses precomputed descriptors and a full-layer target; production staging/route readback omitted",
                        "Event spans include host issue gaps; wall also includes per-iteration synchronization",
                        "Overlap uses independent BF16 GEMM, not a dependency-correct model pipeline; copy stream waits and joins are timed",
                    ],
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        torch.cuda.synchronize()
        module.mailbox_unregister(source)


if __name__ == "__main__":
    main()
