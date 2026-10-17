#!/usr/bin/env python3
"""Controlled overlap sweep and valid-route-only diagnostic ablations."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from sglang.srt.layers.quantization import qwen38_cached_w4a16 as original

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w4a16_verify_gpu_check import swizzle
from graph_bench import timing


def variants(folder):
    source = Path(original.__file__).read_text()
    zero = """    if reuse:
        intermediate.zero_()
        partial2.zero_()
"""
    assert source.count(zero) == 1
    start = source.index("    # Preserve every route (including duplicates)")
    end = source.index("\n\n@triton.jit\ndef _splitk_swiglu", start)
    vector = """    route_tokens = rr // TOPK
    scattered = tl.sum(tl.where(
        tt[:, None, None] == route_tokens[None, :, None],
        values[:, None, :], 0.0), axis=0)
    tl.store(Y + rr[:, None] * N + n[None, :], scattered,
             mask=(rr[:, None] < ROUTES) & (n[None, :] < N) &
                  (all_ids[:, None] == expert) & valid)
"""
    vector_source = source[:start] + vector + source[end:]
    result = {
        "baseline": (original, "baseline"),
        "fused": (original, "fused"),
        "reuse": (original, "reuse"),
    }
    for name, text in (
        ("reuse_no_zero", source.replace(zero, "    if reuse:\n        pass\n")),
        ("reuse_vector", vector_source),
        (
            "reuse_vector_no_zero",
            vector_source.replace(zero, "    if reuse:\n        pass\n"),
        ),
    ):
        path = folder / (name + ".py")
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        result[name] = (module, "reuse")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    folder = args.output.parent / "ablations"
    folder.mkdir(parents=True, exist_ok=True)
    modes = variants(folder)
    metadata = {}
    current = {}
    # Capture compiler resource reports without changing generated code.
    for module in {id(m): m for m, _ in modes.values()}.values():
        for attr in ("_cache_project_splitk", "_cache_project_reuse"):
            jit = getattr(module, attr)
            run = jit.run

            def wrapped(*a, _run=run, _attr=attr, **kw):
                compiled = _run(*a, **kw)
                if compiled is not None:
                    key = (current["mode"], current["tokens"], kw.get("FIRST"))
                    metadata[key] = dict(
                        kernel=_attr,
                        first=kw.get("FIRST"),
                        n_regs=getattr(compiled, "n_regs", None),
                        n_spills=getattr(compiled, "n_spills", None),
                        shared_bytes=getattr(compiled.metadata, "shared", None),
                    )
                return compiled

            jit.run = wrapped
    torch.manual_seed(20261018)
    experts, hidden, width, topk = 32, 2560, 640, 10
    tensors = {}
    for prefix, rows, cols in (("w13", 2 * width, hidden), ("w2", hidden, width)):
        tensors[prefix + "_weight"] = torch.randint(
            0, 256, (experts, rows, cols // 2), dtype=torch.uint8, device="cuda"
        )
        scales = (
            torch.randint(1, 6, (experts, rows, cols // 16))
            .float()
            .mul(0.125)
            .to(torch.float8_e4m3fn)
        )
        tensors[prefix + "_blockscale_swizzled"] = swizzle(scales)
        tensors[prefix + "_input_scale_quant"] = torch.ones(1, device="cuda")
        tensors["g1_alphas" if prefix == "w13" else "g2_alphas"] = torch.full(
            (experts,), 0.025, device="cuda"
        )
    reports = []
    for tokens in (2, 3):
        for shared in (0, 2, 4, 6, 8, 10):
            ids = torch.tensor(
                [
                    list(range(shared))
                    + list(
                        range(
                            shared + t * (topk - shared),
                            shared + (t + 1) * (topk - shared),
                        )
                    )
                    for t in range(tokens)
                ],
                device="cuda",
                dtype=torch.int32,
            )
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.2
            router = torch.randn(tokens, topk, device="cuda").softmax(-1)
            current.update(mode="fused", tokens=tokens)
            expected = original.cached_w4a16_moe(
                x, ids, router, tensors, split_k=1, verify_mode="fused"
            )
            for name, (module, mode) in modes.items():
                current.update(mode=name, tokens=tokens)

                def run():
                    return module.cached_w4a16_moe(
                        x, ids, router, tensors, split_k=1, verify_mode=mode
                    )

                torch.testing.assert_close(run(), expected, atol=0.004, rtol=0.02)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(32):
                        out = run()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, atol=0.004, rtol=0.02)
                unique = shared + tokens * (topk - shared)
                row = dict(
                    tokens=tokens,
                    shared_experts=shared,
                    unique_experts=unique,
                    reuse_fraction=1 - unique / (tokens * topk),
                    mode=name,
                    compiler=[metadata[name, tokens, first] for first in (True, False)],
                    **timing(graph, 32, samples=20),
                )
                reports.append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "compiler"}),
                    flush=True,
                )
                args.output.write_text(
                    json.dumps(dict(status="passed_so_far", reports=reports), indent=2)
                    + "\n"
                )
                del out, graph
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                gpu=torch.cuda.get_device_name(),
                scope="CUDA graph, resident synthetic weights, all routes valid; no-zero variants are diagnostic and unsafe for invalid routes",
                reports=reports,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
