"""Verify every compressed scale slab and every unchanged tensor, streaming."""

import hashlib
import json
import mmap
from pathlib import Path

from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import decode
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import load_manifest

root = Path("/root/.cache/sglang")
rev = "b9ce18a6887e18c089fdea8509a33e63429a3300"
native = load_manifest(root / "qwen38-fp4-local" / rev / "manifest.json")
csf = load_manifest(root / "qwen38-fp4-local-csf-tiered" / rev / "manifest.json")
report = {"status": "running", "scale_slabs": 0, "unchanged_bytes": 0, "layers": []}
with native.payload_path.open("rb") as nf, csf.payload_path.open("rb") as cf:
    with (
        mmap.mmap(nf.fileno(), 0, access=mmap.ACCESS_READ) as nm,
        mmap.mmap(cf.fileno(), 0, access=mmap.ACCESS_READ) as cm,
    ):
        for layer, records in native.tensors.items():
            hashes = {}
            for name, record in records.items():
                dest = csf.tensors[layer][name]
                nh, ch = hashlib.sha256(), hashlib.sha256()
                if name in csf.scale_encoding:
                    spec = csf.scale_encoding[name]
                    size = spec["rows"] * spec["columns"]
                    for expert in range(native.num_experts):
                        raw = nm[
                            record.offset + expert * size : record.offset
                            + (expert + 1) * size
                        ]
                        start = dest.offset + expert * spec["stride"]
                        if spec["encoding"] == "b12x_csf_tiered_v1":
                            from sglang.srt.model_loader.qwen38_nvfp4_pack.tiered import (
                                restore_slab,
                            )

                            encoded = restore_slab(
                                cm, start, spec, spec["overflow"][str(layer)]
                            )
                        else:
                            encoded = cm[start : start + spec["stride"]]
                        expanded = decode(
                            encoded,
                            spec["rows"],
                            spec["columns"],
                        )
                        assert raw == expanded, (layer, name, expert)
                        nh.update(raw)
                        ch.update(expanded)
                        report["scale_slabs"] += 1
                else:
                    assert record.nbytes == dest.nbytes, (layer, name)
                    for offset in range(0, record.nbytes, 8 << 20):
                        count = min(8 << 20, record.nbytes - offset)
                        a = nm[record.offset + offset : record.offset + offset + count]
                        b = cm[dest.offset + offset : dest.offset + offset + count]
                        assert a == b, (layer, name, offset)
                        nh.update(a)
                        ch.update(b)
                        report["unchanged_bytes"] += count
                assert nh.hexdigest() == ch.hexdigest()
                hashes[name] = nh.hexdigest()
            report["layers"].append({"layer": layer, "sha256_expanded": hashes})
            print("verified layer", layer, flush=True)
report["status"] = "passed"
Path("/results/pack-validation.json").write_text(json.dumps(report, indent=2) + "\n")
