"""Verify the recorded archive; no GPU, serving, network or live source edits.

Run from any directory with Python's standard library. Temporary application
checks live inside this evidence directory and are removed on completion.
Historical scripts are parsed, not executed or reformatted.
"""

import ast
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[2]
    manifest = json.loads((evidence / "archive_manifest.json").read_text())
    paths = [root / name for name in manifest["files"]]
    failures = []
    for name, identity in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256(path) != identity["sha256"]:
            failures.append(f"Archive hash mismatch: {name}")
            continue
        if path.stat().st_size != identity["bytes"]:
            failures.append(f"Archive size mismatch: {name}")
        if path.suffix == ".json":
            json.loads(path.read_text())
        elif path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
    for name in ("README.md", "EXPERIMENTS.md", "ARCHIVE.md"):
        path = evidence / name
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if resolved not in paths and resolved != evidence / "archive_manifest.json":
                failures.append(f"Unarchived document link: {name} -> {target}")

    result_dir = evidence / "results"
    final = json.loads((result_dir / "final_source_manifest.json").read_text())
    meta = json.loads((result_dir / "selected_patch_manifest.json").read_text())
    assert sha256(result_dir / "selected_candidate.patch") == meta["patch_sha256"]
    assert sha256(result_dir / "selected_registry.patch") == meta["registry_sha256"]
    baseline = json.loads((evidence / "baseline/source_manifest.json").read_text())
    for name, identity in baseline["files"].items():
        assert sha256(evidence / "baseline" / name) == identity, name
    selected = result_dir / "selected_source"
    selected_files = [path for path in selected.rglob("*") if path.is_file()]
    for path in selected_files:
        name = str(path.relative_to(selected))
        assert sha256(path) == final["files"][name], name

    scratch = evidence / "validation"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-apply-", dir=scratch) as temporary:
        checkout = Path(temporary)
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        for name in meta["baseline_manifest"]:
            path = checkout / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((evidence / "baseline" / name).read_bytes())
        patch = str(result_dir / "selected_candidate.patch")
        subprocess.run(["git", "apply", "--check", patch], cwd=checkout, check=True)
        subprocess.run(["git", "apply", patch], cwd=checkout, check=True)
        recreated = [p for p in checkout.rglob("*") if p.is_file() and ".git" not in p.parts]
        for path in recreated:
            name = str(path.relative_to(checkout))
            assert sha256(path) == final["files"][name], name
        assert len(recreated) == 11, len(recreated)

    summary = json.loads((result_dir / "final_model_summary.json").read_text())
    for name, groups in summary.items():
        raw = json.loads((result_dir / (name + ".json")).read_text())
        assert raw["complete"], name
        for mode, arms in groups.items():
            for arm, expected in arms.items():
                rows = [r for r in raw["samples"] if r["mode"] == mode and r["arm"] == arm]
                actual = sum(r["tok_s"] for r in rows) / len(rows)
                assert abs(actual - expected) < 1e-10, (name, mode, arm)
                assert all(r["exact"] for r in rows), (name, mode, arm)
    http = json.loads((result_dir / "http_final_summary.json").read_text())
    assert http["complete"] and len(http["pairs"]) == 6
    assert all(row["tokens_exact"] for row in http["pairs"])
    for arm in ("ar", "dflash"):
        actual = sum(r[arm + "_tok_s"] for r in http["pairs"]) / 6
        assert abs(actual - http[arm + "_mean"]) < 1e-10, arm
    assert not failures, "\n".join(failures)
    print(json.dumps({
        "archive_files": len(paths),
        "archive_bytes": sum(p.stat().st_size for p in paths),
        "final_source_copies_hash_exact": len(selected_files),
        "final_candidate_patch_recreated_files": 11,
        "fullmodel_summaries_checked": len(summary),
        "http_pairs_exact": 6,
        "registry_application": "Remote archived result only; complete registry baselines not present locally",
        "device_tests_rerun": False,
        "complete": True,
    }, indent=2))


if __name__ == "__main__":
    main()
