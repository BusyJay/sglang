# SPDX-License-Identifier: Apache-2.0
"""Compare native B12X loads on local checkpoint bytes; no CSF decoding."""

import argparse
import json
import mmap
import statistics
from pathlib import Path

import torch

from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--bits", type=int, nargs="+", default=[0, 64, 128])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 16, 32, 47])
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    assert not manifest.scale_encoding
    capacity, topk = 147, manifest.top_k
    layers = args.layers
    with manifest.payload_path.open("rb") as f:
        mp = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        weights = []
        for layer in layers:
            tensors = {}
            for name, record in manifest.tensors[layer].items():
                dtype = getattr(torch, record.dtype.replace("torch.", ""))
                shape = list(record.shape)
                size = record.nbytes
                # Keep real, distinct experts in slots 1..10; zero is masked.
                if shape and shape[0] == manifest.num_experts:
                    size //= manifest.num_experts
                    shape[0] = capacity
                    tensor = torch.zeros(shape, dtype=dtype, device="cuda")
                    active_shape = [topk] + shape[1:]
                    active = (
                        torch.frombuffer(
                            mp,
                            dtype=dtype,
                            count=topk * size // tensor.element_size(),
                            offset=record.offset,
                        )
                        .reshape(active_shape)
                        .clone()
                    )
                    tensor[1 : topk + 1].copy_(active)
                else:
                    tensor = (
                        torch.frombuffer(
                            mp,
                            dtype=dtype,
                            count=size // torch.empty([], dtype=dtype).element_size(),
                            offset=record.offset,
                        )
                        .reshape(shape)
                        .clone()
                        .cuda()
                    )
                tensors[name] = tensor
            weights.append(tensors)
    output = {"layers": layers, "capacity": capacity, "topk": topk, "cases": []}
    for m in args.tokens:
        torch.manual_seed(731 + m)
        x = (torch.randn(m, 2560, device="cuda") * 0.02).bfloat16()
        ids = (
            torch.arange(1, topk + 1, device="cuda", dtype=torch.int32)
            .expand(m, -1)
            .contiguous()
        )
        router = torch.rand(m, topk, device="cuda")
        router /= router.sum(-1, keepdim=True)
        graphs, results = {}, {}
        for bits in args.bits:
            adapter = CachedB12xW4A16(
                x.device, 2560, 640, capacity, native_load_bits=bits
            )
            # A plain event is needed to capture several independent scratch uses.
            adapter.done = torch.cuda.Event()
            ys = [adapter.run(x, ids, router, tensors).clone() for tensors in weights]
            torch.cuda.synchronize()
            assert all(torch.isfinite(y).all().item() for y in ys)
            if bits == 0:
                reference = ys
            else:
                for expected, actual in zip(reference, ys):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            adapter.used = False
            with torch.cuda.graph(graph):
                for _ in range(4):
                    for tensors in weights:
                        adapter.run(x, ids, router, tensors)
            graphs[bits] = (graph, adapter)
            results[bits] = []
        # Alternate execution order to reduce clock/thermal bias.
        for repeat in range(9):
            for bits in (args.bits if repeat % 2 == 0 else reversed(args.bits)):
                graph = graphs[bits][0]
                for _ in range(3):
                    graph.replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(30):
                    graph.replay()
                end.record()
                end.synchronize()
                results[bits].append(
                    start.elapsed_time(end) * 1000 / (30 * 4 * len(weights))
                )
        case = {
            "m": m,
            "bitwise_equal": True,
            "timing": {
                str(bits): {
                    "median_us": statistics.median(samples),
                    "samples_us": samples,
                }
                for bits, samples in results.items()
            },
        }
        output["cases"].append(case)
        print(json.dumps(case), flush=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
