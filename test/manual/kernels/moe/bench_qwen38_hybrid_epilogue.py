# SPDX-License-Identifier: Apache-2.0
"""Measure one full-GPU B12X layer with old merge and fused FC2 output.

Requires an SM120 GPU and the Qwen38 pack manifest. Preloaded resident
experts avoid weight transfers so the measured delta isolates output handling.
"""

import argparse
import mmap
import random
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--capacity", type=int, default=6388)
    args = p.parse_args()
    if args.samples < 1 or args.warmup < 1 or args.capacity < 32:
        p.error("invalid sample count or capacity")

    import torch
    from sglang.kernels.ops.moe.qwen38_hybrid_plan import hybrid_merge
    from sglang.srt.layers.moe.qwen38_hybrid_device import _MappedBatchPage
    from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
        Qwen38ExpertPack,
        load_manifest,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("SM120 GPU required")
    manifest = load_manifest(args.manifest)
    pack = Qwen38ExpertPack(manifest, source_mode="file_cow")
    from sglang.kernels.ops.moe.qwen38_expert_cache import _module as fill_module

    fill = fill_module()
    source = torch.frombuffer(pack.cuda_source_mapping(), dtype=torch.uint8)
    source_address = fill.mailbox_register(source)
    page = _MappedBatchPage(manifest.hidden_size, manifest.top_k, torch.device("cuda"))
    failed_map = mmap.mmap(-1, mmap.PAGESIZE, flags=mmap.MAP_SHARED)
    failed_bytes = torch.frombuffer(failed_map, dtype=torch.uint8)
    failed_address = fill.mailbox_register(failed_bytes)
    try:
        layer = 0
        names = (
            "w13_weight",
            "w2_weight",
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
            "g1_alphas",
            "g2_alphas",
        )
        dtypes = dict(
            w13_weight=torch.uint8,
            w2_weight=torch.uint8,
            w13_blockscale_swizzled=torch.float8_e4m3fn,
            w2_blockscale_swizzled=torch.float8_e4m3fn,
            g1_alphas=torch.float32,
            g2_alphas=torch.float32,
        )
        pool = {
            name: torch.zeros(
                (args.capacity, *manifest.tensors[layer][name].shape[1:]),
                dtype=dtypes[name],
                device="cuda",
            )
            for name in names
        }
        row = manifest.tensors[layer]
        layout = torch.tensor(
            [
                (
                    pool[name].data_ptr(),
                    row[name].nbytes // manifest.num_experts,
                    source_address + row[name].offset,
                )
                for name in names
            ],
            dtype=torch.int64,
            device="cuda",
        )
        scales = {
            name: torch.frombuffer(
                pack.cuda_source_mapping(),
                dtype=torch.float32,
                count=1,
                offset=row[name].offset,
            )
            .clone()
            .to("cuda")
            for name in ("w13_input_scale_quant", "w2_input_scale_quant")
        }
        plan = torch.zeros(4 + 2 * manifest.top_k, dtype=torch.int32, device="cuda")
        plan[1] = manifest.top_k
        for route in range(manifest.top_k):
            plan[4 + 2 * route] = route
            plan[5 + 2 * route] = route + 1
        fill.mailbox_fill_grid(plan, layout, manifest.top_k, failed_address, 8)
        torch.cuda.synchronize()
        plan[1] = 0
        plan[2] = 0
        ids = torch.arange(
            1, manifest.top_k + 1, dtype=torch.int32, device="cuda"
        ).view(1, -1)
        weights = torch.full(
            (1, manifest.top_k), 0.1, dtype=torch.float32, device="cuda"
        )
        x = torch.randn(1, manifest.hidden_size, dtype=torch.bfloat16, device="cuda")
        backend = CachedB12xW4A16(
            torch.device("cuda"),
            manifest.hidden_size,
            manifest.intermediate_size,
            args.capacity,
            enable_fp32=True,
        )
        completed = torch.zeros(1, dtype=torch.int32, device="cuda")
        cpu_ids = torch.full_like(ids, -1)
        rng = random.Random(617)
        timings = {"old": [], "fused": []}
        outputs = {}
        for step in range(args.warmup + args.samples):
            order = ["old", "fused"]
            rng.shuffle(order)
            for mode in order:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                if mode == "old":
                    gpu = backend.run(
                        x, ids, weights, {**pool, **scales}, output_dtype=torch.float32
                    )
                    output = torch.empty_like(x)
                    hybrid_merge(
                        cpu_ids,
                        gpu,
                        completed,
                        output,
                        page.batch.device_addresses[3],
                        1,
                        failed_address,
                    )
                else:
                    output = backend.run(
                        x,
                        ids,
                        weights,
                        {**pool, **scales},
                        hybrid=(
                            plan,
                            page.batch.device_addresses[5],
                            page.batch.device_addresses[3],
                            failed_address,
                            1,
                        ),
                    )
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
                if step >= args.warmup:
                    timings[mode].append(elapsed)
                if step == 0:
                    outputs[mode] = output.clone()
        torch.testing.assert_close(outputs["old"], outputs["fused"], rtol=0, atol=0)
        old, fused = (statistics.median(timings[mode]) for mode in ("old", "fused"))
        print(f"old_us={old:.3f} fused_us={fused:.3f} ratio={fused / old:.3f}")
    finally:
        torch.cuda.synchronize()
        fill.mailbox_unregister(failed_bytes)
        fill.mailbox_unregister(source)
        page.close(torch.device("cuda"))
        del failed_bytes
        failed_map.close()
        del source
        pack.close()


if __name__ == "__main__":
    main()
