#!/usr/bin/env python3
"""Package recommended series, checking every step in an isolated Git index."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

BASE = 'd1d85bf29b855e511d2a96f9a7f6cec73fc2bb1f'
A2 = '55b13ab866d71b610599c5e92feb921003f737a1'
L2 = 'bb8a71dc35356f145cd3c6950d25da2e767e6a70'
PACKED = '1d71e13964ef29db46fc103a76b13b7257e20d21'
SUBMIT = 'f703b2f9e993e6a6420ce186cdc69e7a78a02597'
ARGV = '49d67ac22ef5dcb1ee07f06860588829f520ed95'
DEFAULTS = '12d8a01e97101edfc7d229b231409e5798fa8862'
FINALR3 = "af9aac635231f587268859ec3ceec0ab10b34050"
ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--logs", required=True)
    p.add_argument("--target", default="HEAD")
    args = p.parse_args()

    def git(*argv, env=None):
        return (
            subprocess.check_output(["git", *argv], cwd=ROOT, env=env).decode().strip()
        )

    target = git("rev-parse", args.target)
    out = Path(args.output).resolve()
    out.mkdir(exist_ok=False)
    stops = [BASE, A2, L2, PACKED, SUBMIT, ARGV, DEFAULTS, FINALR3, target]
    names = [
        "00-benchmark-a2.patch",
        "06-l2-r3.patch",
        "05-packed.patch",
        "01-submission.patch",
        "10-final-tools.patch",
        "11-experimental-defaults.patch",
        "12-final-delivery.patch",
        "13-startup-memory.patch",
    ]
    records = []
    for start, end, name in zip(stops, stops[1:], names):
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "--full-index", start, end], cwd=ROOT
        )
        (out / name).write_bytes(patch)
        records.append(
            {
                "patch": name,
                "base": start,
                "target": end,
                "base_tree": git("rev-parse", start + "^{tree}"),
                "target_tree": git("rev-parse", end + "^{tree}"),
            }
        )
    with tempfile.TemporaryDirectory(prefix="qwen38-final-index-") as td:
        env = {**os.environ, "GIT_INDEX_FILE": td + "/index"}
        for i, start in enumerate(stops[:-1]):
            git("read-tree", start, env=env)
            for record in records[i:]:
                git("apply", "--cached", "--check", str(out / record["patch"]), env=env)
                git("apply", "--cached", str(out / record["patch"]), env=env)
                if git("write-tree", env=env) != record["target_tree"]:
                    raise ValueError("series target tree mismatch")
    (out / "series").write_text("\n".join(names) + "\n")
    docs = out / "handoff"
    docs.mkdir()
    for name in [
        "FINAL_GPU_HANDOFF.md",
        "GPU_HANDOFF.md",
        "FINAL_VALIDATION.json",
        "FINAL_R1_VALIDATION.json",
        "FINAL_R1_RESPONSE.md",
        "FINAL_R2_VALIDATION.json",
        "FINAL_R2_RESPONSE.md",
        "EXPERIMENTAL_DEFAULTS.md",
        "PATCH06_HANDOFF.md",
        "PATCH05_HANDOFF.md",
        "PATCH01_HANDOFF.md",
        "RESULT_SCHEMA.md",
        "machine.example.json",
        "gates.example.json",
        "STARTUP_MEMORY_DIAGNOSTICS.md",
        "STARTUP_MEMORY_VALIDATION.json",
    ]:
        shutil.copyfile(ROOT / "tools/qwen38_cache_tests/benchmark" / name, docs / name)
    shutil.copytree(args.logs, out / "raw")
    (out / "package.json").write_text(
        json.dumps(
            {
                "commit": target,
                "parent": git("rev-parse", target + "^"),
                "base": BASE,
                "performance_base": "384c83da44a4334dcaa74c474195ab960126e029",
                "steps": records,
                "target_tree": git("rev-parse", target + "^{tree}"),
                "isolated_apply_every_start": "passed",
                "GPU": "not_run",
            },
            indent=2,
        )
        + "\n"
    )
    (out / "README.md").write_text(
        """Recommended Qwen38 series: benchmark a2 → consolidated 06 r3 → 05 → 01 → final tools/argv fix → experimental defaults → delivery r3 → startup memory diagnosis.
Historical defective 06 r1/r2 are not replayed. SHA256SUMS covers all files.
At implementation base apply every entry in series. At a2 skip 00; at 06 r3 skip 00/06;
at 05 skip 00/06/05; at 01 apply 10/11/12/13; at argv-fix apply 11/12/13; at defaults apply 12/13; at final r3 apply only 13. All earlier starts also apply 13. Do not mix paths.
package.json records exact base/target trees. Each start/step checked in isolated index.
Same-tree toggled ablations are not independent 04+single patch source trees.
See handoff/FINAL_GPU_HANDOFF.md for runnable staged commands and unexecuted tests.
User-authorized test defaults dynamic/packed/packed/group2 (L2 overlap-off).
Explicit old baseline off/direct/direct/group1/overlap-on retained.
For allocator diagnosis see handoff/STARTUP_MEMORY_DIAGNOSTICS.md (commands and investigation inside this tar).
All source/tools are delivered in patches, including the summarizer.
No GPU benchmark values or validated acceleration profile included.
"""
    )

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (out / "SHA256SUMS").write_text(
        "".join(
            f"{sha(f)}  {f.relative_to(out)}\n"
            for f in sorted(out.rglob("*"))
            if f.is_file()
        )
    )
    archive = Path(str(out) + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    with tarfile.open(archive, "r:gz") as tar:
        for f in out.rglob("*"):
            if (
                f.is_file()
                and tar.extractfile(out.name + "/" + str(f.relative_to(out))).read()
                != f.read_bytes()
            ):
                raise ValueError("tar content mismatch")
    Path(str(archive) + ".sha256").write_text(sha(archive) + "  " + archive.name + "\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": sha(archive),
                "commit": target,
                "target_tree": git("rev-parse", target + "^{tree}"),
            },
            indent=2,
        )
    )


if __name__ == '__main__':main()
