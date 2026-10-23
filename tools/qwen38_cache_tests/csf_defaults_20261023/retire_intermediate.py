"""Remove only this experiment's superseded fixed-stride CSF payload."""

import json
from pathlib import Path

root = Path("/root/.cache/sglang")
rev = "b9ce18a6887e18c089fdea8509a33e63429a3300"
audit = Path("/results/intermediate-retirement.json")
assert (
    json.loads(Path("/results/pack-validation.json").read_text())["status"] == "passed"
)
assert (
    root / "qwen38-fp4-local-csf-tiered" / rev / "experts.bin"
).stat().st_size == 65235107840
old = root / "qwen38-fp4-local-csf" / rev
payload = old / "experts.bin"
if audit.exists():
    raise FileExistsError(audit)
manifest = json.loads((old / "manifest.json").read_text())
assert payload.stat().st_size == 67225387008
(root / "archives" / "csf-intermediate-manifest-20261023.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
report = dict(
    reason="Superseded experiment intermediate; tiered pack fully validated",
    payload=str(payload),
    reclaimed_bytes=payload.stat().st_blocks * 512,
    native_preserved=True,
    tiered_preserved=True,
)
payload.unlink()
(old / "manifest.json").unlink()
(old / "RETIRED.md").write_text(
    "Fixed-stride experiment retired after lossless tiered conversion. Use ../.. /qwen38-fp4-local-csf-tiered at the same checkpoint. Archived manifest: ../../archives/csf-intermediate-manifest-20261023.json\n"
)
audit.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
