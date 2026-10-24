"""Compare compressed and native operators on actual checkpoint expert bytes."""

import mmap
from pathlib import Path
import torch
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import load_manifest
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe
from sglang.kernels.ops.moe.qwen38_expert_cache import csf_patch_headers

root = Path("/root/.cache/sglang")
rev = "b9ce18a6887e18c089fdea8509a33e63429a3300"
ns = load_manifest(root / "qwen38-fp4-local" / rev / "manifest.json")
cs = load_manifest(root / "qwen38-fp4-local-csf-tiered" / rev / "manifest.json")
files = [m.payload_path.open("rb") for m in [ns, cs]]
mm = [mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) for f in files]
H, I = ns.hidden_size, ns.intermediate_size
capacity = max(8, ns.top_k)
keep = []


def load(m, mp, layer, expert):
    out = {}
    for n, r in m.tensors[layer].items():
        dtype = getattr(torch, r.dtype.replace("torch.", ""))
        shape = list(r.shape)
        off = r.offset
        count = r.nbytes
        if shape and shape[0] == 512:
            count //= 512
            off += expert * count
            shape[0] = 1
        t = (
            torch.frombuffer(
                mp,
                dtype=dtype,
                count=count // torch.empty([], dtype=dtype).element_size(),
                offset=off,
            )
            .clone()
            .reshape(shape)
            .cuda()
        )
        if n in m.scale_encoding:
            s = m.scale_encoding[n]
            rg = s["overflow"][str(layer)]
            if rg["nbytes"]:
                bank = (
                    torch.frombuffer(
                        mp, dtype=torch.uint8, count=rg["nbytes"], offset=rg["offset"]
                    )
                    .clone()
                    .cuda()
                )
                keep.append(bank)
                csf_patch_headers(
                    t,
                    bank.data_ptr() - rg["offset"],
                    rg["offset"],
                    rg["offset"] + rg["nbytes"],
                )
        out[n] = (
            t.repeat((capacity,) + (1,) * (t.ndim - 1))
            if t.ndim
            and t.shape[0] == 1
            and n not in ("w13_input_scale_quant", "w2_input_scale_quant")
            else t
        )
    return out


A = CachedB12xW4A16(torch.device("cuda"), H, I, capacity, enable_fp32=False)
B = CachedB12xW4A16(
    torch.device("cuda"),
    H,
    I,
    capacity,
    enable_fp32=False,
    scale_encoding=cs.scale_encoding,
)
A4 = CachedB12xW4A4(
    torch.device("cuda"),
    H,
    I,
    capacity,
    ns.top_k,
    scale_encoding={},
    deterministic_output=False,
)
B4 = CachedB12xW4A4(
    torch.device("cuda"),
    H,
    I,
    capacity,
    ns.top_k,
    scale_encoding=cs.scale_encoding,
    deterministic_output=False,
)
for l, e in [(0, 0), (5, 0), (47, 0)]:
    nn = load(ns, mm[0], l, e)
    cc = load(cs, mm[1], l, e)
    print(
        "L/E",
        l,
        e,
        "scalars",
        [(n, t.cpu().tolist()) for n, t in nn.items() if t.numel() == 1],
        flush=True,
    )
    for nt in [1, 3, 129]:
        torch.manual_seed(51)
        x = torch.randn(nt, H, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.arange(ns.top_k, device="cuda", dtype=torch.int32)
            .expand(nt, -1)
            .contiguous()
        )
        assert ids.min().item() >= 0 and ids.max().item() < capacity
        assert nn["w13_weight"].shape[0] == cc["w13_weight"].shape[0] == capacity
        w = torch.full_like(ids, 1 / ns.top_k, dtype=torch.float32)
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        ref = cutlass_fused_moe(
            output=torch.empty_like(x),
            input=x,
            token_selected_experts=ids,
            token_final_scales=w,
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
            swiglu_alpha=torch.ones(capacity, device="cuda"),
            swiglu_beta=torch.zeros(capacity, device="cuda"),
            swiglu_limit=torch.full((capacity,), float("inf"), device="cuda"),
            activation_type=ActivationType.Swiglu,
        )[0]
        pairs = [
            ("cutlass-a4", ref, A4.run(x, ids, w, nn)),
            ("a4", A4.run(x, ids, w, nn), B4.run(x, ids, w, cc)),
        ]
        if nt <= 3:
            pairs.extend(
                [
                    ("a16", A.run(x, ids, w, nn), B.run(x, ids, w, cc)),
                    (
                        "triton",
                        cached_w4a16_moe(x, ids, w, nn),
                        cached_w4a16_moe(x, ids, w, cc),
                    ),
                ]
            )
            if nt == 3:
                split = torch.cat(
                    [
                        B.run(
                            x[i : i + 1],
                            ids[i : i + 1].clone(),
                            w[i : i + 1].clone(),
                            cc,
                        )
                        for i in range(nt)
                    ]
                )
                pairs.append(("csf-m3-vs-split", B.run(x, ids, w, cc), split))
        for name, a, b in pairs:
            assert torch.isfinite(a).all().item(), (l, nt, name, "native nonfinite")
            assert torch.isfinite(b).all().item(), (l, nt, name, "compressed nonfinite")
            print(
                name,
                nt,
                "native",
                a.abs().max().item(),
                "compressed",
                b.abs().max().item(),
                "diff",
                (a - b).abs().max().item(),
                "nan",
                b.isnan().sum().item(),
                flush=True,
            )
            print(
                "relative L2",
                ((a - b).float().norm() / a.float().norm()).item(),
                flush=True,
            )
            if name in ("a16", "triton"):
                assert torch.equal(a, b), (l, nt, name, "native/CSF mismatch")
    print("completed; exact assertions apply to a16/triton only", flush=True)
