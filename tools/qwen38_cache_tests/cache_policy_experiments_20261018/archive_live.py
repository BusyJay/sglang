"""Archive complete experiments, including raw process repeats and route tapes."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    raw, dest = args.raw, args.destination
    experiment = json.loads((raw / "experiment.json").read_text())
    if not (raw / "completed.json").exists():
        raise ValueError("Experiment is incomplete")
    restored = json.loads((raw / "restored.json").read_text())
    if not restored["language_healthy"] or not restored["vit_unchanged"]:
        raise ValueError("Serving restoration has not been verified")
    if (raw / "restored.json").stat().st_mtime < (
        raw / "completed.json"
    ).stat().st_mtime:
        raise ValueError("Restoration record predates this completion")
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {}

    def copy(file):
        relative = file.relative_to(raw)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = file.read_bytes()
        # These remain transparently readable by read_sweeps and trace tools.
        if file.name.endswith((".jsonl", ".trace.json", ".patch")):
            target = target.with_name(target.name + ".gz")
            target.write_bytes(gzip.compress(data, mtime=0))
        else:
            shutil.copy2(file, target)
        manifest[str(target.relative_to(dest))] = dict(
            archived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            original_sha256=hashlib.sha256(data).hexdigest(),
        )

    for case in ["coding", "raft"]:
        for stage, expected in [("timing", 5 * experiment["rounds"]), ("parity", 5)]:
            folder = raw / f"{case}-{stage}"
            reports = list(folder.glob("[0-9]*.json"))
            if len(reports) != expected:
                raise ValueError(f"Incomplete matrix: {case}-{stage}")
            for file in folder.iterdir():
                copy(file)
        for suffix in [
            ".jsonl",
            ".jsonl.meta.json",
            "-request.json",
            "-response.json",
            "-cpu.json",
            "-cpu.log",
            "-server-info.json",
        ]:
            copy(raw / f"{case}{suffix}")
        copy(raw / f"record-{case}.sh")
    if (raw / "profiles").exists():
        for file in (raw / "profiles").iterdir():
            copy(file)
    for name in [
        "experiment.json",
        "completed.json",
        "restored.json",
        "hardware.txt",
        "launcher-before.sh",
        "gpu.sh",
        "tool-source-hashes.json",
        "timing-source-hashes.json",
        "tooling.patch",
        "runtime-source-provenance.json",
        "transfer-manifest.json",
        "raft-cpu-adaptive.json",
        "raft-cpu-adaptive.log",
        "cpu-validation.log",
        "cpu-new-cases.log",
        "cpu-checks.sh",
        "run.log",
        "restored-scheduler-env.json",
        "post-restore-health.json",
        "summary.json",
        "profile-prefix.jsonl",
    ]:
        if (raw / name).exists():
            copy(raw / name)
    pack = Path(
        "/home/jay/workspace/prod/cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json"
    )
    shutil.copy2(pack, dest / "pack-manifest.json")
    pack_sha = hashlib.sha256(pack.read_bytes()).hexdigest()
    for case in ["coding", "raft"]:
        for file in (raw / f"{case}-timing").glob("[0-9]*.json"):
            if json.loads(file.read_text())["manifest_sha256"] != pack_sha:
                raise ValueError("Archived manifest differs from timed pack manifest")
    manifest["pack-manifest.json"] = dict(archived_sha256=pack_sha)
    (dest / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Archived", len(manifest), "artifacts", dest)


if __name__ == "__main__":
    main()
