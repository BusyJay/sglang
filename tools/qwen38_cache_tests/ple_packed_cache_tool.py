#!/usr/bin/env python3
"""Build/benchmark persistent FP4 PLE rows without loading a model or service."""

import argparse
import ast
import importlib.util
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def copy_function():
    path = ROOT / "python/sglang/srt/models/qwen38_fp4_ple.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "copy_packed_ple_table"
    ]
    namespace = {"torch": torch, "os": os}
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 -- local source only
    return namespace["copy_packed_ple_table"]


def checkpoint_planes(checkpoint):
    index = load_source(
        "ple_startup_index", "python/sglang/srt/models/qwen4_exp_ple_lazy.py"
    )
    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = {
        name: shard for name, shard in mapping.items() if index.PLE_SHARD.search(name)
    }
    if not names:
        raise ValueError("No indexed PLE weight/scale shards in checkpoint")
    descriptors = {}
    for shard in sorted(set(names.values())):
        path = checkpoint / shard
        if Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise ValueError("Invalid checkpoint shard path")
        descriptors.update(index.read_ple_checkpoint_index(str(path)))
    prefixes = {name.split(".ngram_embedding.shard_")[0] for name in names}
    if len(prefixes) != 1:
        raise ValueError("This tool requires exactly one FP4 PLE table")
    paired = {"weight": {}, "weight_scale": {}}
    for name in names:
        match = index.PLE_SHARD.search(name)
        kind = name.rsplit(".", 1)[1]
        paired[kind][int(match.group(1))] = descriptors[name]
    if paired["weight"].keys() != paired["weight_scale"].keys():
        raise ValueError("PLE weight/scale shard sets differ")
    weights, scales, cursor = [], [], 0
    width = None
    for shard in sorted(paired["weight"]):
        if shard != len(weights):
            raise ValueError("Missing PLE shard")
        weight, scale = paired["weight"][shard], paired["weight_scale"][shard]
        if weight.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
            raise ValueError("Not an FP4 PLE checkpoint")
        if weight.shape[0] != scale.shape[0] or weight.shape[1] != scale.shape[1] * 8:
            raise ValueError("Invalid weight/scale geometry")
        if width is not None and width != weight.shape[1] * 2:
            raise ValueError("PLE shard widths differ")
        width = weight.shape[1] * 2
        end = cursor + weight.shape[0]
        weights.append((cursor, end, weight))
        scales.append((cursor, end, scale))
        cursor = end
    return weights, scales, cursor, width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Small pinned correctness/performance experiment",
    )
    parser.add_argument("--rows", type=int, default=1048576)
    parser.add_argument(
        "--pin-benchmark",
        action="store_true",
        help="Allocate a full pinned table and load it twice",
    )
    parser.add_argument(
        "--pageable",
        action="store_true",
        help="CPU-only byte parity; does not measure pin cost",
    )
    args = parser.parse_args()
    cache = load_source(
        "ple_startup_cache", "python/sglang/srt/models/qwen38_ple_packed_cache.py"
    )
    copy = copy_function()
    if args.synthetic:
        if args.rows <= 0:
            parser.error("rows must be positive")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        descriptors = []
        for kind, stride in [("weight", 80), ("scale", 10)]:
            path = args.cache_dir / ("synthetic-" + kind + ".bin")
            if not path.exists() or path.stat().st_size != args.rows * stride:
                with path.open("wb") as handle:
                    block = bytes(range(256)) * 4096
                    remaining = args.rows * stride
                    while remaining:
                        chunk = block[: min(remaining, len(block))]
                        handle.write(chunk)
                        remaining -= len(chunk)
            st = path.stat()
            descriptors.append(
                types.SimpleNamespace(
                    path=str(path),
                    shape=(args.rows, stride),
                    dtype=torch.uint8 if kind == "weight" else torch.float8_e4m3fn,
                    offset=0,
                    identity=(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns),
                )
            )
        weights, scales, rows, width = (
            [(0, args.rows, descriptors[0])],
            [(0, args.rows, descriptors[1])],
            args.rows,
            160,
        )
    else:
        if args.checkpoint is None:
            parser.error("Need --checkpoint or --synthetic")
        weights, scales, rows, width = checkpoint_planes(args.checkpoint.resolve())
    row_bytes = width // 2 + width // 16

    def writer(view):
        table = torch.frombuffer(view, dtype=torch.uint8).view(rows, row_bytes)
        try:
            copy(table, weights, scales, rows, width)
        finally:
            del table

    path, hit, seconds = cache.ensure_packed_ple_cache(
        args.cache_dir, weights, scales, rows, width, 0, writer
    )
    result = {
        "status": "CACHE_READY",
        "cache_path": str(path),
        "cache_hit": hit,
        "prepare_seconds": seconds,
        "rows": rows,
        "row_bytes": row_bytes,
        "bytes": rows * row_bytes,
        "torch": torch.__version__,
    }
    # Bound memory for a full checkpoint check: compare beginning/end of every shard.
    with path.open("rb") as handle:
        for start, end, _ in weights:
            for row in (start, end - 1):
                expected = torch.empty(1, row_bytes, dtype=torch.uint8)
                copy(expected, weights, scales, 1, width, row_start=row)
                handle.seek(row * row_bytes)
                if handle.read(row_bytes) != expected.numpy().tobytes():
                    raise AssertionError("Packed cache sample mismatch")
    result["boundary_samples_exact"] = True
    if args.synthetic or args.pin_benchmark:
        if not args.pageable and not torch.cuda.is_available():
            raise RuntimeError("Pinned benchmark requires CUDA; cache build completed")
        started = time.perf_counter()
        table = torch.empty(
            rows, row_bytes, dtype=torch.uint8, pin_memory=not args.pageable
        )
        result["allocate_seconds"] = time.perf_counter() - started
        result["pinned"] = not args.pageable
        started = time.perf_counter()
        copy(table, weights, scales, rows, width)
        result["legacy_repack_seconds"] = time.perf_counter() - started
        # Full byte comparison is bounded per chunk; no second full table allocation.
        with path.open("rb") as handle:
            for start in range(0, rows, 65536):
                part = table[start : start + 65536]
                if handle.read(part.numel()) != part.numpy().tobytes():
                    raise AssertionError("Full packed cache mismatch")
        result["full_byte_parity"] = True
        result["cached_load_seconds"] = [
            cache.read_packed_ple_cache(path, table.numpy()) for _ in range(2)
        ]
        if args.synthetic:
            # Exercise the real finalize branch twice, without constructing the model.
            source_path = ROOT / "python/sglang/srt/models/qwen38_fp4_ple.py"
            tree = ast.parse(source_path.read_text())
            cls = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == "Qwen38Fp4PleEmbedding"
            )
            method = next(
                node
                for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "finalize_lazy"
            )
            tree.body = [method]
            namespace = {
                "torch": torch,
                "nn": torch.nn,
                "os": os,
                "time": time,
                "logger": logging.getLogger("ple_finalize_check"),
                "allocate_ple_host_table": lambda **kw: torch.empty(
                    kw["shape"], dtype=kw["dtype"], pin_memory=not args.pageable
                ),
                "copy_packed_ple_table": copy,
                "make_ple_file_prefetcher": lambda table: None,
                "make_ple_file_rss_trimmer": lambda table: None,
            }
            exec(compile(tree, str(source_path), "exec"), namespace)  # noqa: S102
            previous = os.environ.get("SGLANG_QWEN38_PLE_PACKED_CACHE")
            os.environ.pop("SGLANG_QWEN38_PLE_PACKED_CACHE", None)
            try:
                # Warm hit must never invoke the checkpoint-plane repacker.
                namespace["copy_packed_ple_table"] = lambda *a, **kw: (
                    _ for _ in ()
                ).throw(AssertionError("Cache hit unexpectedly repacked checkpoint"))
                obj = types.SimpleNamespace(
                    fp4_global_scale=torch.ones(1),
                    _lazy_shards=weights,
                    _scale_shards=scales,
                    org_vocab_size=rows,
                    _checkpoint_rows=rows,
                    embedding_dim=width,
                    _lazy=False,
                    _packed_ready=False,
                    _backend="pinned",
                    _table_dir=str(args.cache_dir),
                    weight=torch.empty(
                        rows, width // 2, device="meta", dtype=torch.uint8
                    ),
                    shard_indices=types.SimpleNamespace(org_vocab_start_index=0),
                )
                namespace["finalize_lazy"](obj)
                torch.testing.assert_close(obj.weight, table, atol=0, rtol=0)
                result["finalize_cache_hit_exact"] = True
            finally:
                if previous is None:
                    os.environ.pop("SGLANG_QWEN38_PLE_PACKED_CACHE", None)
                else:
                    os.environ["SGLANG_QWEN38_PLE_PACKED_CACHE"] = previous
        result["status"] = (
            "PAGEABLE_CHECK_PASS" if args.pageable else "PINNED_CHECK_PASS"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
