"""Temporary live operand parity probe; activated only by an explicit PYTHONPATH."""

import json
import struct
from pathlib import Path
import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import decode
from sglang.srt.model_loader.qwen38_nvfp4_pack.tiered import fixed_bytes

original = CachedB12xW4A4.run
count = 0


def probe(self, x, ids, router, tensors):
    global count
    result = original(self, x, ids, router, tensors)
    if x.shape[0] < 128:
        return result
    layer = count % 48
    count += 1
    if layer not in (0, 1, 47):
        return result
    active = torch.unique(ids[ids >= 0]).long()
    remap = torch.full(
        (tensors["w13_weight"].shape[0],), -1, device=x.device, dtype=torch.int32
    )
    remap[active] = torch.arange(len(active), device=x.device, dtype=torch.int32)
    local = torch.where(ids >= 0, remap[ids.clamp_min(0)], -1).contiguous()
    nn = {
        n: (
            t[active].contiguous()
            if n not in ("w13_input_scale_quant", "w2_input_scale_quant")
            else t
        )
        for n, t in tensors.items()
    }
    file = Path(
        "/root/.cache/sglang/qwen38-fp4-local-csf-tiered/b9ce18a6887e18c089fdea8509a33e63429a3300/experts.bin"
    ).open("rb")
    for name, s in self.specs.items():
        raw = []
        for row in nn[name].cpu().numpy():
            head = bytearray(row.tobytes())
            off = struct.unpack_from("<Q", head, 976)[0]
            n = struct.unpack_from("<I", head, 1008)[0]
            head[976:1008] = bytes(32)
            head[1016:1024] = bytes(8)
            if off:
                file.seek(off)
                head = (
                    head[: fixed_bytes(s["rows"], s["columns"])]
                    + file.read(n * 4)
                    + bytes((-n * 4) % 16)
                )
            raw.append(
                torch.frombuffer(
                    bytearray(decode(head, s["rows"], s["columns"])), dtype=torch.uint8
                ).clone()
            )
        nn[name] = (
            torch.stack(raw)
            .reshape(len(active), s["rows"], s["columns"])
            .cuda()
            .view(torch.float8_e4m3fn)
        )
    file.close()
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    e = len(active)
    ref = cutlass_fused_moe(
        output=torch.empty_like(x),
        input=x,
        token_selected_experts=local.int(),
        token_final_scales=router.float(),
        fc1_expert_weights=nn["w13_weight"].view(torch.long),
        fc2_expert_weights=nn["w2_weight"].view(torch.long),
        output_dtype=torch.bfloat16,
        input_sf=None,
        quant_scales=[
            nn["w13_input_scale_quant"].reshape(()),
            nn["w13_blockscale_swizzled"].view(torch.int32),
            nn["g1_alphas"],
            nn["w2_input_scale_quant"].reshape(()),
            nn["w2_blockscale_swizzled"].view(torch.int32),
            nn["g2_alphas"],
        ],
        swiglu_alpha=torch.ones(e, device="cuda"),
        swiglu_beta=torch.zeros(e, device="cuda"),
        swiglu_limit=torch.full((e,), float("inf"), device="cuda"),
        activation_type=ActivationType.Swiglu,
    )[0]
    from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16

    a16 = CachedB12xW4A16(x.device, self.hidden, self.width, e)
    ref16 = a16.run(x[:1], local[:1], router[:1], nn)
    native = CachedB12xW4A4(
        x.device,
        self.hidden,
        self.width,
        e,
        self.topk,
        scale_encoding={},
        deterministic_output=True,
    )
    other = original(native, x, local, router, nn)
    report = dict(
        layer=layer,
        shape=list(x.shape),
        pool=tensors["w13_weight"].shape[0],
        active=e,
        cutlass_max=ref.abs().max().item(),
        csf_max=result.abs().max().item(),
        native_b12x_max=other.abs().max().item(),
        cutlass_vs_csf_l2=((ref - result).float().norm() / ref.float().norm()).item(),
        native_vs_csf_l2=(
            (other - result).float().norm() / other.float().norm()
        ).item(),
        cutlass_vs_native_l2=((ref - other).float().norm() / ref.float().norm()).item(),
        a16_max=ref16.abs().max().item(),
    )
    print("LIVE_PARITY", json.dumps(report), flush=True)
    with Path("/results/live-parity.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    if layer == 47:
        raise RuntimeError("Intentional stop after live operand parity probe")
    return ref


CachedB12xW4A4.run = probe
