# SPDX-License-Identifier: Apache-2.0
"""GPU parity gate for synthetic native pack; run on the target GPU manually.

This verifies CPU/GPU numerical partitioning, not serving lifecycle or speed.
"""

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cores", default="0,1,2,3")
    parser.add_argument("--backend", choices=("triton", "b12x"), default="b12x")
    args = parser.parse_args()
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["GOMP_SPINCOUNT"] = "0"
    os.nice(15)
    import torch
    from build_qwen38_cpu_hybrid import check_temperature
    from check_qwen38_cpu_hybrid import fixture

    from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4
    from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe

    cores = tuple(map(int, args.cores.split(",")))
    if not 1 <= len(cores) <= 4 or not set(cores) <= os.sched_getaffinity(0):
        raise ValueError("choose at most four available physical cores")
    if len(set(cores)) != len(cores):
        raise ValueError("duplicate cores")
    siblings = [
        Path(
            f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        ).read_text()
        for c in cores
    ]
    if len(set(siblings)) != len(cores):
        raise ValueError("cores share physical siblings")
    geometries = ((2560, 640),) if args.backend == "b12x" else ((64, 32), (160, 160))
    for h, width in geometries:
        payload, offsets, _, values = fixture(h, width)
        tensors = {}
        names = (
            "w13_weight",
            "w2_weight",
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
            "g1_alphas",
            "g2_alphas",
            "w13_input_scale_quant",
            "w2_input_scale_quant",
        )
        shapes = (
            (3, 2 * width, h // 2),
            (3, h, width // 2),
            (3, (2 * width + 127) // 128 * 128, ((h // 16 + 3) // 4) * 4),
            (3, (h + 127) // 128 * 128, ((width // 16 + 3) // 4) * 4),
            (3,),
            (3,),
            (),
            (),
        )
        for k, (name, shape) in enumerate(zip(names, shapes)):
            end = offsets[k + 1] if k + 1 < len(offsets) else len(payload)
            raw = torch.frombuffer(
                bytearray(payload[offsets[k] : end]), dtype=torch.uint8
            ).clone()
            dtype = (
                torch.uint8
                if k < 2
                else torch.float8_e4m3fn
                if k < 4
                else torch.float32
            )
            tensors[name] = raw.view(dtype).reshape(shape).cuda()
        pack = SimpleNamespace(
            manifest=SimpleNamespace(
                hidden_size=h,
                intermediate_size=width,
                num_experts=3,
                top_k=4,
                tensors={
                    0: {
                        name: SimpleNamespace(offset=offset)
                        for name, offset in zip(names, offsets)
                    }
                },
            ),
            cuda_source_mapping=lambda: payload,
        )
        cpu = CpuNvfp4(args.library, pack, cores, gpu_backend=args.backend)
        native = None
        if args.backend == "b12x":
            from sglang.srt.layers.quantization.qwen38_cached_b12x import (
                CachedB12xW4A16,
            )

            native = CachedB12xW4A16(
                torch.device("cuda"), h, width, 1025, enable_fp32=True
            )
            # Global noncontiguous slots, >512 capacity; masked slot 0 stays zero.
            slots = torch.tensor([1, 600, 1024], device="cuda")
            for name in names[:6]:
                source = tensors[name]
                pool = torch.zeros(
                    (1025, *source.shape[1:]), dtype=source.dtype, device="cuda"
                )
                pool.reshape(1025, -1).view(torch.uint8).index_copy_(
                    0, slots, source.reshape(3, -1).view(torch.uint8)
                )
                tensors[name] = pool

        def gpu_run(x, ids, weights):
            if native is not None:
                ids = torch.where(ids >= 0, slots[ids.clamp_min(0).long()], -1).int()
                return native.run(x, ids, weights, tensors, output_dtype=torch.float32)
            return cached_w4a16_moe(
                x,
                ids,
                weights,
                tensors,
                split_k=0,
                verify_mode="baseline",
                output_dtype=torch.float32,
            )

        for nt in (1, 2, 3):
            check_temperature()
            x = (
                torch.tensor(values[: nt * h], dtype=torch.uint16)
                .view(torch.bfloat16)
                .reshape(nt, h)
            )
            ids = torch.tensor(
                [[0, 1, 0, -1], [1, 2, -1, -1], [2, 0, 1, -1]][:nt], dtype=torch.int32
            )
            weights = torch.tensor(
                [[0.13, 0.29, 0.07, 0], [0.37, 0.19, 0, 0], [0.21, 0.31, 0.17, 0]][:nt],
                dtype=torch.float32,
            )
            full = gpu_run(x.cuda(), ids.cuda(), weights.cuda()).to(torch.bfloat16)
            for resident in ({0}, {0, 1}, set(), {0, 1, 2}):
                cpu_ids = ids.clone()
                gpu_ids = ids.clone()
                for e in range(3):
                    (cpu_ids if e in resident else gpu_ids)[ids == e] = -1
                out = torch.empty_like(x, dtype=torch.float32)
                cpu.run(0, x, cpu_ids, weights, out)
                partial = gpu_run(x.cuda(), gpu_ids.cuda(), weights.cuda())
                actual = (partial + out.cuda()).to(torch.bfloat16)
                torch.testing.assert_close(
                    actual.float(), full.float(), rtol=0.025, atol=0.003
                )
                torch.cuda.synchronize()
                print(
                    f"PASS h={h} M={nt} resident={sorted(resident)} max_abs_error={(actual.float() - full.float()).abs().max().item():.6g}",
                    flush=True,
                )
                time.sleep(0.2)
        cpu.release_worker()
        cpu.close()
    print(
        "GPU synthetic partition parity passed; run serving/graph/prefill transitions separately."
    )


if __name__ == "__main__":
    main()
