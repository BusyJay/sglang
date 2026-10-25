#!/usr/bin/env python3
"""Replay real MTP projection inputs; one quantized candidate resident at a time.

Input is a torch.save dict with BF16 weight[N,K] and hidden[S,K] captured
immediately before the MTP logits processor (draft and draft_extend separately).
No serving integration or static activation calibration is implied.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def measure_cuda_events(torch, operation, repeats):
    """Batch event pairs; the measured group has one final host wait."""
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    # Materialize event handles before the measured group.
    for start, stop in pairs:
        start.record()
        stop.record()
    pairs[-1][1].synchronize()
    result = None
    for start, stop in pairs:
        start.record()
        result = operation()
        stop.record()
    pairs[-1][1].synchronize()
    return result, [start.elapsed_time(stop) for start, stop in pairs]


def replay_sample(torch, project, x, warmup, repeats):
    for _ in range(warmup):
        project(x)
    eager, eager_times = measure_cuda_events(torch, lambda: project(x), repeats)
    # Correctness transfers are outside both measurement groups.
    eager_cpu = eager.cpu()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(3, warmup)):
            project(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = project(x)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(warmup):
        graph.replay()
    _, graph_times = measure_cuda_events(torch, graph.replay, repeats)
    return eager_cpu, graph_output.cpu(), eager_times, graph_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=("bf16", "fp8", "int8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Independent BF16 export; captured input logits take priority",
    )
    parser.add_argument(
        "--export-reference",
        type=Path,
        help="BF16 only: export full-vocabulary original projection outputs",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.repeats, args.rows_per_chunk) < 1:
        parser.error("invalid repeat/chunk counts")
    if args.export_reference is not None and args.format != "bf16":
        parser.error("--export-reference requires --format bf16")
    import torch

    data = torch.load(args.input, map_location="cpu", weights_only=True)
    weight, hidden = data["weight"], data["hidden"]
    assert weight.ndim == hidden.ndim == 2 and weight.shape[1] == hidden.shape[1]
    assert weight.dtype == hidden.dtype == torch.bfloat16
    assert hidden.shape[0] > 0 and weight.shape[0] >= 2
    assert torch.isfinite(weight).all() and torch.isfinite(hidden).all()
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    input_sha256 = digest.hexdigest()
    reference_logits = data.get("logits")
    oracle = "captured_original_logits" if reference_logits is not None else None
    if reference_logits is None and args.reference is not None:
        reference_data = torch.load(
            args.reference, map_location="cpu", weights_only=True
        )
        if reference_data["input_sha256"] != input_sha256:
            raise ValueError("reference input hash mismatch")
        reference_logits = reference_data["logits"]
        oracle = "independent_full_vocabulary_bf16_export"
    if reference_logits is None and args.format != "bf16":
        parser.error("quantized candidates require captured logits or --reference")
    if reference_logits is not None:
        if reference_logits.shape != (hidden.shape[0], weight.shape[0]):
            raise ValueError("reference logits shape mismatch")
        if not torch.isfinite(reference_logits).all():
            raise ValueError("reference contains non-finite values")
    hidden = hidden.cuda().contiguous()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.format == "bf16":
        quant = weight.cuda()
        scale = None
        backend = "torch.mm BF16"
        project = lambda x: torch.mm(x, quant.T)
    else:
        dtype = torch.int8 if args.format == "int8" else torch.float8_e4m3fn
        limit = 127.0 if args.format == "int8" else 448.0
        # FP8 FlashInfer BMM uses per-tensor scales. INT8 uses per-row scales.
        scale = (
            torch.empty((weight.shape[0], 1), device="cuda", dtype=torch.float32)
            if args.format == "int8"
            else torch.tensor(
                [max(float(weight.abs().max()) / limit, 1e-12)], device="cuda"
            )
        )
        quant = torch.empty(weight.shape, device="cuda", dtype=dtype)
        for begin in range(0, weight.shape[0], args.rows_per_chunk):
            end = begin + args.rows_per_chunk
            block = weight[begin:end].cuda().float()
            row_scale = (
                block.abs().amax(dim=1, keepdim=True).div(limit).clamp_min(1e-12)
                if args.format == "int8"
                else scale
            )
            values = block.div(row_scale).clamp(-limit, limit)
            if args.format == "int8":
                values = values.round()
                scale[begin:end].copy_(row_scale)
            quant[begin:end].copy_(values.to(dtype))
            del block, values, row_scale
        if args.format == "int8":
            from sgl_kernel import int8_scaled_mm

            from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

            backend = "sgl_kernel.int8_scaled_mm + per_token_quant_int8"

            def project(x):
                qx, sx = per_token_quant_int8(x)
                return int8_scaled_mm(qx, quant.T, sx, scale, out_dtype=torch.bfloat16)
        else:
            from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
            from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

            backend = "flashinfer.bmm_fp8 backend=cublas + dynamic scaled_fp8_quant"

            def project(x):
                qx, sx = scaled_fp8_quant(x)
                return flashinfer_bmm_fp8(qx, quant.T, sx, scale, torch.bfloat16)

    torch.cuda.synchronize()
    construction_seconds = time.perf_counter() - started
    construction_peak = torch.cuda.max_memory_allocated()
    if reference_logits is None or args.export_reference is not None:
        # A separate BF16 process holds the whole original head and executes
        # full-vocabulary GEMV, avoiding the chunked-GEMM reduction oracle.
        full_logits = torch.cat([project(x).cpu() for x in hidden.split(1)], dim=0)
        if args.export_reference is not None:
            args.export_reference.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    input_sha256=input_sha256,
                    logits=full_logits,
                    projection="full_vocabulary_bf16_per_sample",
                ),
                args.export_reference,
            )
        if reference_logits is None:
            reference_logits = full_logits
            oracle = "full_vocabulary_bf16_per_sample"
    errors, agreement, margins = [], [], []
    timings = {"eager": [], "graph": []}
    graph_agreement = []
    torch.cuda.reset_peak_memory_stats()
    for index, x in enumerate(hidden.split(1)):
        reference = reference_logits[index : index + 1].float()
        ref_top = reference.topk(2, dim=1)
        eager, graph, eager_times, graph_times = replay_sample(
            torch, project, x, args.warmup, args.repeats
        )
        timings["eager"].append(eager_times)
        timings["graph"].append(graph_times)
        result = graph.float()
        graph_agreement.append(bool(torch.equal(eager, graph)))
        agreement.append(bool((result.argmax(1) == reference.argmax(1)).all()))
        margins.append(float(ref_top.values[0, 0] - ref_top.values[0, 1]))
        error = (result - reference).abs().flatten()
        errors.append(
            dict(
                max=float(error.max()),
                mean=float(error.mean()),
                p99=float(error.quantile(0.99)),
            )
        )
    row = dict(
        format=args.format,
        format_version=2,
        backend=backend,
        input_sha256=input_sha256,
        weight_shape=list(weight.shape),
        samples=hidden.shape[0],
        output_dtype="bfloat16",
        activation_scale="dynamic" if args.format != "bf16" else None,
        device=torch.cuda.get_device_name(),
        sm=list(torch.cuda.get_device_capability()),
        torch_version=torch.__version__,
        construction_seconds=construction_seconds,
        construction_peak_bytes=construction_peak,
        replay_peak_bytes=torch.cuda.max_memory_allocated(),
        weight_bytes=quant.nbytes,
        scale_bytes=0 if scale is None else scale.nbytes,
        timings_ms_by_sample=timings,
        oracle=oracle,
        eager_graph_bitwise_by_sample=graph_agreement,
        measurement="prewarmed event pairs, batched enqueue, final event wait",
        projection_batch_size=1,
        top1_agreement=sum(agreement) / len(agreement),
        top1_by_sample=agreement,
        reference_margins=margins,
        errors=errors,
        limitations=[
            "Operator replay only; no serving acceptance/capacity result.",
            "BF16 export is per-sample full-vocabulary GEMV; captured serving logits preferred.",
            "Backend failure is an unsupported experiment; no silent fallback.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
