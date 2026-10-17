#!/usr/bin/env python3
"""SM120 correctness-gated MTP and mapped packed-PLE microbenchmarks; no server."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.srt.layers.quantization.qwen38_fp4_mtp import fp4_mtp_moe
from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel


def measure(fn, iterations):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    for begin, end in pairs:
        begin.record()
        result = fn()
        end.record()
    torch.cuda.synchronize()
    samples = sorted(begin.elapsed_time(end) for begin, end in pairs)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
        "iterations": iterations,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "output_shape": list(result.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--experts", type=int, default=512)
    args = parser.parse_args()
    if args.iterations < 5 or args.experts < 10:
        parser.error("need iterations >= 5 and experts >= 10")
    # The existing check raises when CUDA/SM120 is absent or parity fails.
    import importlib.util

    check_path = Path(__file__).with_name("fp4_local_gpu_check.py")
    spec = importlib.util.spec_from_file_location("fp4_local_gpu_check", check_path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    correctness = checks.check()
    torch.manual_seed(123)
    e, h, n, k = args.experts, 2560, 640, 10
    w13 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((e, 2 * n, h // 16), 0.5, dtype=torch.float8_e4m3fn, device="cuda")
    s2 = torch.full((e, h, n // 16), 0.5, dtype=torch.float8_e4m3fn, device="cuda")
    g13 = torch.full((e, 2), 0.03, device="cuda")
    g2 = torch.full((e,), 0.03, device="cuda")
    mtp = []
    for tokens in (1, 2, 4, 16, 64):
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(tokens)])
        router = torch.full((tokens, k), 1 / k, device="cuda")
        record = measure(
            lambda: fp4_mtp_moe(x, ids, router, w13, w2, s13, s2, g13, g2),
            args.iterations,
        )
        record.update(tokens=tokens, experts=e, hidden=h, intermediate=n, topk=k)
        mtp.append(record)
    w13 = w2 = s13 = s2 = g13 = g2 = None
    torch.cuda.synchronize()
    rows, dim = 65536, 160
    host = torch.empty(rows, 90, dtype=torch.uint8, pin_memory=True)
    host[:, :80].random_(0, 256)
    host[:, 80:] = torch.full((rows, 10), 0.5, dtype=torch.float8_e4m3fn).view(
        torch.uint8
    )
    ple = []
    global_scale = torch.tensor([3.324236240587197e-5], device="cuda")
    for lookups in (1, 32, 512, 8192):
        ids = torch.randint(0, rows, (lookups,), device="cuda")
        out = torch.empty(lookups, dim, device="cuda", dtype=torch.bfloat16)

        def gather():
            _gather_ple_embedding_from_pinned_kernel[(lookups,)](
                host.data_ptr(),
                ids,
                out,
                embedding_dim=dim,
                tp_vocab_start=0,
                tp_vocab_end=rows,
                is_fp8=False,
                BLOCK_D=256,
                MASK_INVALID=True,
                IS_NVFP4=True,
                global_scale_ptr=global_scale,
            )
            return out

        record = measure(gather, args.iterations)
        record.update(lookups=lookups, storage_row_bytes=90, output_dim=dim)
        ple.append(record)
    report = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "correctness": correctness,
        "mtp": mtp,
        "ple_mapped_gather": ple,
        "scope": "Synthetic layer microbenchmarks; excludes checkpoint I/O, expert misses, service and e2e.",
        "timing": "CUDA events after warmup; MTP includes its per-call temporary allocations, PLE reuses output.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
