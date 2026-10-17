#!/usr/bin/env python3
"""Mapped BF16 gather parity/graph/latency; checkpoint is temporary, no cache dirs."""

import argparse
import ast
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from tools.qwen38_cache_tests._load_source import REPO_ROOT
from tools.qwen38_cache_tests.test_cpu_host_embedding import checkpoint


def kernel():
    tree = ast.parse((REPO_ROOT / "python/sglang/srt/models/qwen4_exp.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_gather_ple_embedding_from_pinned_kernel"
    )
    # Triton inspect.getsource requires a real file for a JIT function.
    return ast.unparse(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=248320)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    report = {"cuda_available": torch.cuda.is_available(), "cases": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not report["cuda_available"]:
        report["status"] = "GPU_REQUIRED"
        path.write_text(json.dumps(report, indent=2))
        return
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        kernel_path = directory / "gather.py"
        kernel_path.write_text(
            "import triton\nimport triton.language as tl\n" + kernel()
        )
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "mapped_gather_bench", kernel_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gather = module._gather_ple_embedding_from_pinned_kernel
        # Values are deterministic and exactly representable in BF16.
        source = (
            torch.arange(args.rows * args.width, dtype=torch.int32)
            .remainder_(251)
            .reshape(args.rows, args.width)
            .to(torch.bfloat16)
        )
        weights = directory / "embedding.safetensors"
        checkpoint(weights, source)
        from sglang.srt.models.qwen38_host_embedding import (
            Qwen38MappedTokenEmbedding,
            find_token_embedding,
        )

        baseline = torch.cuda.memory_allocated()
        started = time.perf_counter()
        model = Qwen38MappedTokenEmbedding(
            args.rows, args.width, find_token_embedding([weights])
        )
        model.weight_loader(model.weight, source)
        report.update(
            register_seconds=time.perf_counter() - started,
            host_weight_bytes=model.weight.nbytes,
            mapped_gpu_bytes=torch.cuda.memory_allocated() - baseline,
        )
        device_weight = source.cuda()
        report["gpu_weight_bytes"] = device_weight.nbytes
        report["saved_gpu_bytes"] = device_weight.nbytes - report["mapped_gpu_bytes"]
        stub = SimpleNamespace(_gather_ple_embedding_from_pinned_kernel=gather)
        with mock.patch.dict("sys.modules", {"sglang.srt.models.qwen4_exp": stub}):
            for tokens in (1, 3, 128, 4096):
                ids = (
                    torch.arange(tokens, device="cuda", dtype=torch.int32) * 127
                ) % args.rows
                # Exercise flat and non-flat input shapes, edge rows and duplicates.
                ids[0] = args.rows - 1
                reference = nn.functional.embedding(ids.long(), device_weight)
                actual = model._embed_local_shard(ids)
                parity = torch.equal(reference, actual)
                graph = torch.cuda.CUDAGraph()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        model._embed_local_shard(ids)
                torch.cuda.current_stream().wait_stream(side)
                with torch.cuda.graph(graph):
                    graph_output = model._embed_local_shard(ids)
                graph.replay()
                graph_parity = torch.equal(reference, graph_output)

                def measure(call):
                    samples = []
                    for _ in range(args.rounds):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        begin.record()
                        for _ in range(args.iterations):
                            call()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end) * 1000 / args.iterations)
                    return statistics.median(samples)

                mapped_us = measure(lambda: model._embed_local_shard(ids))
                gpu_us = measure(
                    lambda: nn.functional.embedding(ids.long(), device_weight)
                )
                report["cases"].append(
                    dict(
                        tokens=tokens,
                        parity=parity,
                        graph_parity=graph_parity,
                        mapped_us=mapped_us,
                        gpu_us=gpu_us,
                        added_us=mapped_us - gpu_us,
                    )
                )
        model.close()
    report["status"] = (
        "PASS"
        if all(c["parity"] and c["graph_parity"] for c in report["cases"])
        else "FAIL"
    )
    path.write_text(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
