"""Find the first nonfinite value across repeated integrated expert invocations."""

import json
from pathlib import Path
import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16

seen = False
calls = {}
instances = {}


def wrap(cls, label):
    original = cls.run

    def run(self, x, *args, **kwargs):
        global seen
        if torch.cuda.is_current_stream_capturing():
            return original(self, x, *args, **kwargs)
        if label == "prefill" and x.shape[0] >= 128:
            seen = True
        if not seen:
            return original(self, x, *args, **kwargs)
        key = (label, id(self))
        calls[key] = calls.get(key, 0) + 1
        layer = instances.setdefault(key, len([k for k in instances if k[0] == label]))
        in_finite = torch.isfinite(x).all().item()
        result = original(self, x, *args, **kwargs)
        out_finite = torch.isfinite(result).all().item()
        if not (in_finite and out_finite):
            report = dict(
                path=label,
                layer=layer,
                call=calls[key],
                tokens=x.shape[0],
                input_finite=in_finite,
                output_finite=out_finite,
                input_max=x.abs().max().item(),
                output_max=result.abs().max().item(),
            )
            print("NONFINITE", json.dumps(report), flush=True)
            Path("/results/first-nonfinite.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            raise RuntimeError("Stop at first nonfinite expert boundary")
        return result

    cls.run = run


wrap(CachedB12xW4A4, "prefill")
wrap(CachedB12xW4A16, "decode")
