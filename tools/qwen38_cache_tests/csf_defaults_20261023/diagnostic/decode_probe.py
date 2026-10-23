"""Explicit-PYTHONPATH probe: compare live decode slots to checkpoint experts."""

import json, mmap, struct
from pathlib import Path
import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import load_manifest
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import decode
from sglang.srt.model_loader.qwen38_nvfp4_pack.tiered import fixed_bytes

old4 = CachedB12xW4A4.run
old16 = CachedB12xW4A16.run
seen = False
count = 0


def prefill(self, x, *args, **kwargs):
    global seen
    if x.shape[0] >= 128:
        seen = True
    return old4(self, x, *args, **kwargs)


def probe(self, x, ids, router, tensors, **kwargs):
    global count
    result = old16(self, x, ids, router, tensors, **kwargs)
    if not seen:
        return result
    layer = count % 48
    count += 1
    root = Path("/root/.cache/sglang")
    rev = "b9ce18a6887e18c089fdea8509a33e63429a3300"
    m = load_manifest(root / "qwen38-fp4-local" / rev / "manifest.json")
    f = m.payload_path.open("rb")
    mp = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    active = torch.unique(ids[ids >= 0]).long()
    r = m.tensors[layer]["w13_weight"]
    stride = r.nbytes // 512
    lookup = {
        mp[r.offset + i * stride : r.offset + i * stride + 32]: i for i in range(512)
    }
    selected = [
        lookup.get(bytes(t.cpu().tolist()), -1)
        for t in tensors["w13_weight"][active].reshape(len(active), -1)[:, :32]
    ]
    if -1 in selected:
        raise RuntimeError(("No checkpoint weight match", layer, selected))
    remap = torch.full(
        (tensors["w13_weight"].shape[0],), -1, device=x.device, dtype=torch.int32
    )
    remap[active] = torch.arange(len(active), device=x.device, dtype=torch.int32)
    local = torch.where(ids >= 0, remap[ids.clamp_min(0)], -1).contiguous()
    nn = {}
    matches = {}
    for n, r in m.tensors[layer].items():
        dtype = getattr(torch, r.dtype)
        shape = list(r.shape)
        if shape[0] == 512:
            shape[0] = len(active)
            b = r.nbytes // 512
            data = b"".join(
                mp[r.offset + e * b : r.offset + (e + 1) * b] for e in selected
            )
        else:
            data = mp[r.offset : r.offset + r.nbytes]
        nn[n] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape).cuda()
        if n not in self.scale_encoding:
            live = tensors[n][active] if r.shape[0] == 512 else tensors[n]
            matches[n] = torch.equal(live.reshape(-1), nn[n].reshape(-1))
        else:
            spec = self.scale_encoding[n]
            raw = []
            for row in tensors[n][active].cpu().numpy():
                head = bytearray(row.tobytes())
                off = struct.unpack_from("<Q", head, 976)[0]
                count_rec = struct.unpack_from("<I", head, 1008)[0]
                head[976:1008] = bytes(32)
                head[1016:1024] = bytes(8)
                if off:
                    with (
                        root / "qwen38-fp4-local-csf-tiered" / rev / "experts.bin"
                    ).open("rb") as cf:
                        cf.seek(off)
                        head = (
                            head[: fixed_bytes(spec["rows"], spec["columns"])]
                            + cf.read(count_rec * 4)
                            + bytes((-count_rec * 4) % 16)
                        )
                raw.append(decode(head, spec["rows"], spec["columns"]))
            matches[n] = (
                b"".join(raw) == nn[n].view(torch.uint8).cpu().numpy().tobytes()
            )
    native = CachedB12xW4A16(
        x.device, self.hidden, self.width, len(active), enable_fp32=True
    )
    ref = old16(native, x, local, router, nn, **kwargs)
    report = dict(
        layer=layer,
        tokens=x.shape[0],
        experts=selected,
        slot_matches=matches,
        native_max=ref.abs().max().item(),
        csf_max=result.abs().max().item(),
        max_diff=(ref - result).abs().max().item(),
        relative_l2=((ref - result).float().norm() / ref.float().norm()).item(),
    )
    print("DECODE_PARITY", json.dumps(report), flush=True)
    with Path("/results/live-decode-all-layers.jsonl").open("a") as dest:
        dest.write(json.dumps(report) + "\n")
    if layer == 47:
        raise RuntimeError("Intentional stop after live decode parity")
    return result


CachedB12xW4A4.run = prefill
CachedB12xW4A16.run = probe
