"""Measure real slab lengths versus the fixed strides used by the runtime."""

import json
import mmap
import statistics
import struct
from pathlib import Path

from sglang.srt.model_loader.qwen38_nvfp4_pack.format import load_manifest

root = Path("/root/.cache/sglang")
rev = "b9ce18a6887e18c089fdea8509a33e63429a3300"
manifests = [
    load_manifest(root / name / rev / "manifest.json")
    for name in ("qwen38-fp4-local", "qwen38-fp4-local-csf")
]
native, csf = manifests
report = {
    "native_payload_bytes": native.payload_nbytes,
    "csf_payload_bytes": csf.payload_nbytes,
    "planes": {},
}
with (
    csf.payload_path.open("rb") as f,
    mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapping,
):
    for name, spec in csf.scale_encoding.items():
        rows, columns, stride = (spec[k] for k in ("rows", "columns", "stride"))
        fixed = 1024 + rows * (1 + columns // 2) + rows * columns // 16
        values, maxima = [], []
        for layer, records in csf.tensors.items():
            sizes = []
            for expert in range(csf.num_experts):
                count = struct.unpack_from(
                    "<I", mapping, records[name].offset + expert * stride + 1008
                )[0]
                sizes.append(fixed + (count + 3) // 4 * 16)
            values.extend(sizes)
            maxima.append({"layer": layer, "max_bytes": max(sizes)})
        report["planes"][name] = dict(
            native_bytes=rows * columns,
            fixed_stride=stride,
            min_bytes=min(values),
            median_bytes=statistics.median(values),
            mean_bytes=statistics.mean(values),
            p95_bytes=sorted(values)[int(0.95 * len(values))],
            max_bytes=max(values),
            padding_fraction=1 - statistics.mean(values) / stride,
            layer_maxima=maxima,
        )
Path("/results/pack-metrics.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
