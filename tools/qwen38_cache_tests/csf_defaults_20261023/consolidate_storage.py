"""One-time local archive consolidation; retain old paths and verify duplicates."""

import hashlib
import json
import os
from pathlib import Path

repo = Path("/home/jay/workspace/sglang")
cache = Path("/home/jay/workspace/prod/cache/sglang")
summary = repo / "tools/qwen38_cache_tests/csf_defaults_20261023/storage-summary.json"
if summary.exists():
    raise FileExistsError(f"consolidation already completed; audit record: {summary}")
archive = cache / "archives"
patches = archive / "patches"
patches.mkdir(parents=True, exist_ok=True)
report = {"moved": [], "deduplicated": [], "bytes_reclaimed": 0}
for path in sorted(repo.glob("*.tar.gz")):
    # This small historical source bundle is tracked by Git; preserve its bytes.
    if path.name == "qwen38-mtp-w4a4-patches-20261018.tar.gz":
        continue
    if path.is_symlink():
        report["moved"].append({"old": str(path), "canonical": str(path.resolve())})
        continue
    dest = patches / path.name
    if dest.exists():
        raise FileExistsError(dest)
    path.rename(dest)
    path.symlink_to(os.path.relpath(dest, path.parent))
    report["moved"].append({"old": str(path), "canonical": str(dest)})
packs = repo / "results/packs"
dest = archive / "sglang-results-packs"
if not packs.is_symlink():
    if dest.exists():
        raise FileExistsError(dest)
    packs.rename(dest)
    packs.symlink_to(os.path.relpath(dest, packs.parent))
    report["moved"].append({"old": str(packs), "canonical": str(dest)})
else:
    report["moved"].append({"old": str(packs), "canonical": str(packs.resolve())})
(archive / "consolidation-20261023.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
legacy = cache / "qwen38-flash-next/fc694b54fb0174e0913e6adf86691ef85a4ead47"
assembled = dest / "assembled"
assert json.loads((legacy / "manifest.json").read_text()) == json.loads(
    (assembled / "manifest.json").read_text()
)
for path in sorted(assembled.rglob("*.bin")):
    canonical = legacy / path.relative_to(assembled)
    if not canonical.is_file() or path.stat().st_size != canonical.stat().st_size:
        continue
    if path.samefile(canonical):
        continue
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as a, canonical.open("rb") as b:
        while True:
            left, right = a.read(8 << 20), b.read(8 << 20)
            if left != right:
                raise ValueError(f"different contents: {path}")
            if not left:
                break
            digest.update(left)
    # Hard links also remain valid inside containers with different mount roots.
    temporary = path.with_suffix(".dedup-link")
    os.link(canonical, temporary)
    os.replace(temporary, path)
    report["deduplicated"].append(
        {
            "alias": str(path),
            "canonical": str(canonical),
            "bytes": size,
            "sha256": digest.hexdigest(),
        }
    )
    report["bytes_reclaimed"] += size
    print("deduplicated", path.relative_to(assembled), "bytes", size, flush=True)
    (archive / "consolidation-20261023.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
(archive / "consolidation-20261023.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
summary.write_text(json.dumps(report, indent=2) + "\n")
print("reclaimed bytes", report["bytes_reclaimed"], flush=True)
