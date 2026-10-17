#!/usr/bin/env python3
"""Numerical checks for NVFP4 PLE/MTP, including graph replay (requires SM120)."""

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.layers.quantization.qwen38_fp4_mtp import fp4_mtp_moe
from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel


def dequant(weight, scale, global_scale):
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
    values = lut[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return (values * scale.float().repeat_interleave(16, -1) * global_scale).bfloat16()


def check():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required; this check must not silently skip")
    torch.manual_seed(42)
    e, h, n, topk = 4, 2560, 640, 3
    w13 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((e, 2 * n, h // 16), 0.5, dtype=torch.float8_e4m3fn, device="cuda")
    s2 = torch.full((e, h, n // 16), 0.5, dtype=torch.float8_e4m3fn, device="cuda")
    g13 = torch.tensor([[0.03, 0.05]] * e, device="cuda")
    g2 = torch.full((e,), 0.03, device="cuda")
    reports = []
    for tokens in (1, 2, 3, 4, 17):
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = torch.tensor([[1, 1, -1]] * tokens, device="cuda", dtype=torch.int64)
        router = torch.tensor([[0.2, 0.3, 0.0]] * tokens, device="cuda")

        def run():
            return fp4_mtp_moe(x, ids, router, w13, w2, s13, s2, g13, g2)

        expected = torch.zeros(tokens, h, device="cuda", dtype=torch.float32)
        for t in range(tokens):
            for k in range(topk):
                expert = int(ids[t, k])
                if expert < 0:
                    continue
                gate = (
                    dequant(w13[expert, :n], s13[expert, :n], g13[expert, 0]).float()
                    @ x[t].float()
                )
                up = (
                    dequant(w13[expert, n:], s13[expert, n:], g13[expert, 1]).float()
                    @ x[t].float()
                )
                inter = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
                down = (
                    dequant(w2[expert], s2[expert], g2[expert]).float() @ inter.float()
                )
                expected[t] += down.float() * router[t, k]
        got = run()
        torch.testing.assert_close(got.float(), expected, atol=0.004, rtol=0.01)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        x.mul_(0.75)
        graph.replay()
        torch.testing.assert_close(captured, run(), atol=0, rtol=0)
        reports.append(
            {
                "tokens": tokens,
                "max_abs_error": float((got.float() - expected).abs().max()),
            }
        )
    packed = torch.randint(0, 256, (7, 80), dtype=torch.uint8)
    scales = torch.full((7, 10), 0.5, dtype=torch.float8_e4m3fn)
    host = torch.cat((packed, scales.view(torch.uint8)), 1).pin_memory()
    ids = torch.tensor([6, 0, 6, -1, 7], device="cuda")
    out = torch.empty(5, 160, device="cuda", dtype=torch.bfloat16)
    _gather_ple_embedding_from_pinned_kernel[(5,)](
        host.data_ptr(),
        ids,
        out,
        embedding_dim=160,
        tp_vocab_start=0,
        tp_vocab_end=7,
        is_fp8=False,
        BLOCK_D=256,
        MASK_INVALID=True,
        IS_NVFP4=True,
    )
    expected = dequant(packed.cuda(), scales.cuda(), 1.0)
    torch.testing.assert_close(out[:3], expected[[6, 0, 6]], atol=0, rtol=0)
    assert not out[3:].count_nonzero()
    # Exercise the actual lazy subclass and inherited staging/event lifecycle.
    import tempfile
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index
    from sglang.srt.models.qwen38_fp4_ple import Qwen38Fp4PleEmbedding

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ple.safetensors"
        prefix = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0."
        save_file({prefix + "weight": packed, prefix + "weight_scale": scales}, path)
        desc = read_ple_checkpoint_index(str(path))
        embedding = Qwen38Fp4PleEmbedding.__new__(Qwen38Fp4PleEmbedding)
        torch.nn.Module.__init__(embedding)
        embedding.embedding_dim = 160
        embedding.org_vocab_size = embedding._checkpoint_rows = 7
        embedding._cache_tables = 1
        embedding._lazy_shards = [(0, 7, desc[prefix + "weight"])]
        embedding._scale_shards = [(0, 7, desc[prefix + "weight_scale"])]
        embedding._lazy = embedding._fp4_ple = True
        embedding._staging_event = None
        embedding._lazy_stats_at = float("inf")
        embedding._lazy_lookup_seconds = 0.0
        embedding._block_d = 256
        embedding.tp_size = 1
        embedding.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=7
        )
        embedding.weight = torch.nn.Parameter(
            torch.empty(7, 80, dtype=torch.uint8, device="meta"), requires_grad=False
        )
        global_scale = 0.001234567
        embedding.register_buffer(
            "fp4_global_scale", torch.tensor([global_scale], device="cuda")
        )
        expected = dequant(packed.cuda(), scales.cuda(), global_scale)
        embedding.finalize_lazy()
        # Force multiple chunks, repeated rows, eviction and a stream change.
        embedding._batch_rows = 2
        for lookup in ([6, 0, 6, -1, 7], [2, 1, 0, 5, 2]):
            lookup_gpu = torch.tensor(lookup, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                result = embedding.gather(lookup_gpu)
            torch.cuda.current_stream().wait_stream(stream)
            reference = torch.zeros_like(result)
            for i, row in enumerate(lookup):
                if 0 <= row < 7:
                    reference[i] = expected[row]
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
        torch.cuda.synchronize()
        if getattr(embedding, "_lazy_cache", None) is not None:
            embedding._lazy_cache.close()
        if getattr(embedding, "_lazy_huge_table", None) is not None:
            embedding._lazy_huge_table.close()
    torch.cuda.synchronize()
    return {"status": "passed", "mtp": reports, "ple": "exact parity"}


def check_checkpoint_ple(checkpoint):
    """Read real row bytes independently, including every shard boundary."""
    import re
    from types import SimpleNamespace
    from unittest import mock

    from safetensors import safe_open

    from sglang.srt.environ import envs
    from sglang.srt.model_loader.qwen38_nvfp4_common import (
        cached_checkpoint_weight_files,
    )
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index
    from sglang.srt.models.qwen38_fp4_ple import Qwen38Fp4PleEmbedding

    root = Path(checkpoint)
    descriptors = {}
    for file in cached_checkpoint_weight_files(root):
        descriptors.update(read_ple_checkpoint_index(file))
    names = sorted(
        (name for name in descriptors if name.endswith(".weight")),
        key=lambda name: int(re.search(r"shard_(\d+)", name).group(1)),
    )
    weights, scales, probes, cursor = [], [], [], 0
    for name in names:
        weight = descriptors[name]
        scale = descriptors[name.removesuffix("weight") + "weight_scale"]
        end = cursor + weight.shape[0]
        weights.append((cursor, end, weight))
        scales.append((cursor, end, scale))
        for local_row in (0, weight.shape[0] - 1):
            # Independent scalar-file reads bypass the row-cache implementation.
            with open(weight.path, "rb") as wfile, open(scale.path, "rb") as sfile:
                wfile.seek(weight.offset + local_row * 80)
                sfile.seek(scale.offset + local_row * 10)
                packed = torch.frombuffer(bytearray(wfile.read(80)), dtype=torch.uint8)
                block_scale = torch.frombuffer(
                    bytearray(sfile.read(10)), dtype=torch.uint8
                )
            probes.append((cursor + local_row, packed, block_scale))
        cursor = end
    embedding = Qwen38Fp4PleEmbedding.__new__(Qwen38Fp4PleEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.embedding_dim = 160
    embedding.org_vocab_size = embedding._checkpoint_rows = cursor
    embedding._cache_tables = 1
    embedding._lazy_shards, embedding._scale_shards = weights, scales
    embedding._lazy = embedding._fp4_ple = True
    embedding._staging_event = None
    embedding._lazy_stats_at = float("inf")
    embedding._lazy_lookup_seconds = 0.0
    embedding._block_d, embedding.tp_size = 256, 1
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=cursor
    )
    embedding.weight = torch.nn.Parameter(
        torch.empty(cursor, 80, device="meta", dtype=torch.uint8), requires_grad=False
    )
    embedding.register_buffer(
        "fp4_global_scale", torch.full((1,), float("nan"), device="cuda")
    )
    global_name = names[0].split(".shard_")[0] + ".weight_scale_2"
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(
        str(root / index[global_name]), framework="pt", device="cpu"
    ) as handle:
        global_scale = handle.get_tensor(global_name)
    target_name = global_name.replace("model.language_model.", "model.")
    loaded = set()
    assert Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer(
        None,
        target_name,
        global_scale,
        {
            target_name.removesuffix("weight_scale_2")
            + "fp4_global_scale": embedding.fp4_global_scale
        },
        loaded,
    )
    with (
        mock.patch.object(envs.SGLANG_QWEN4_PLE_LAZY_CACHE_MB, "get", return_value=1),
        mock.patch.object(envs.SGLANG_QWEN4_PLE_LAZY_BATCH_ROWS, "get", return_value=2),
    ):
        embedding.finalize_lazy()
    try:
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        reference = []
        for _, packed, block_scale in probes:
            codes = torch.stack((packed & 15, packed >> 4), -1).flatten().long()
            values = lut[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
            reference.append(
                (
                    values
                    * block_scale.view(torch.float8_e4m3fn)
                    .float()
                    .repeat_interleave(16)
                    * global_scale
                ).bfloat16()
            )
        expected = (
            torch.stack([*reference, torch.zeros(160), torch.zeros(160)])
            .bfloat16()
            .cuda()
        )
        ids = torch.tensor([*(row for row, _, _ in probes), -1, cursor], device="cuda")
        for reverse in (False, True):
            lookup = ids.flip(0) if reverse else ids
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                result = embedding.gather(lookup)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(
                result, expected.flip(0) if reverse else expected, atol=0, rtol=0
            )
        torch.cuda.synchronize()
        return {
            "status": "passed",
            "shards": len(names),
            "lookups_per_pass": ids.numel(),
            "global_scale": global_scale.item(),
            "parity": "exact",
            "cache_budget_mib": 1,
        }
    finally:
        torch.cuda.synchronize()
        embedding._lazy_cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint", help="also check real PLE rows and global scale"
    )
    args = parser.parse_args()
    report = check()
    if args.checkpoint:
        report["checkpoint_ple"] = check_checkpoint_ple(args.checkpoint)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
