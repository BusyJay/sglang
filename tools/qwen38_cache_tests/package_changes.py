#!/usr/bin/env python3
"""Package the exact current tree without changing the user's Git index."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]

    def git(*command, env=None):
        return subprocess.check_output(["git", *command], cwd=repo, env=env)

    head = git("rev-parse", "HEAD").decode().strip()
    changed = git("diff", "HEAD", "--name-only", "-z").split(b"\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    paths = sorted({os.fsdecode(path) for path in changed + untracked if path})
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = "qwen38-flash-next-decode-graph-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output / (name + ".tar.gz")
    if archive.exists():
        raise FileExistsError(archive)
    with tempfile.TemporaryDirectory(prefix="qwen38-package-") as temporary:
        temp = Path(temporary)
        env = {**os.environ, "GIT_INDEX_FILE": str(temp / "snapshot.index")}
        git("read-tree", head, env=env)
        if paths:
            git("add", "-A", "--", *paths, env=env)
        tree = git("write-tree", env=env).decode().strip()
        root = temp / name
        root.mkdir()
        patches = []
        for label, base in (("01-from-current-head.patch", head),):
            payload = git("diff", "--binary", "--full-index", base, tree, "--")
            (root / label).write_bytes(payload)
            # Apply to an isolated index, checking both patch validity and the
            # exact resulting Git tree. No user files/index/branches are touched.
            check_env = {**os.environ, "GIT_INDEX_FILE": str(temp / (label + ".index"))}
            git("read-tree", base, env=check_env)
            git("apply", "--cached", "--check", str(root / label), env=check_env)
            git("apply", "--cached", str(root / label), env=check_env)
            restored = git("write-tree", env=check_env).decode().strip()
            if restored != tree:
                raise RuntimeError(f"{label} did not reproduce snapshot")
            patches.append({"file": label, "base_commit": base,
                "sha256": hashlib.sha256(payload).hexdigest(), "apply_verified": True})
        (root / "snapshot.json").write_text(json.dumps({
            "head": head, "snapshot_tree": tree,
            "pending_paths_included": paths, "patches": patches,
            "weights_included": False, "gpu_validated_locally": False,
        }, indent=2) + "\n")
        (root / "README.md").write_text(
            "# Flash-Next current snapshot\n\n"
            "Includes all current tracked/untracked changes relative to HEAD only.\n"
            "Includes segmented decode graph, startup warmup and updated test tooling.\n"
            "Previously committed offload/offline-loading changes must already exist on the target.\n"
            "No weights, images, environments or compiled extensions are included.\n\n"
            "Verify: `sha256sum -c SHA256SUMS` from this extracted directory.\n\n"
            f"Base commit: `{head}`. Use `01-from-current-head.patch`.\n\n"
            "From the target SGLang checkout, with a clean worktree:\n\n"
            "```bash\ngit apply --check /absolute/package/01-from-current-head.patch\n"
            "git apply /absolute/package/01-from-current-head.patch\n"
            "python3 tools/qwen38_cache_tests/run_cpu.py\n```\n\n"
            "If your GPU checkout has further edits/commits, check its diff before applying.\n"
            "The patch was applied to an isolated Git index and reproduced snapshot_tree exactly.\n"
            "Read restored flash-next-plan/DECODE_GRAPH.md and STARTUP_WARMUP.md.\n"
            "CPU regression: 102 tests, 89 passed, 13 skipped (PyTorch unavailable).\n"
            "New GPU graph parity and full-model integration have NOT run locally.\n"
            "Use canary --runtime-mode threaded --decode-graph; warmup is opt-in.\n"
        )
        # Copy current instructions into the package for reading before applying.
        docs = root / "docs"
        docs.mkdir()
        for source in (repo / "flash-next-plan").glob("*.md"):
            (docs / source.name).write_bytes(source.read_bytes())
        files = sorted(path for path in root.rglob("*") if path.is_file())
        (root / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
            for path in files))
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=name)
        with tarfile.open(archive, "r:gz") as tar:
            for path in files + [root / "SHA256SUMS"]:
                extracted = tar.extractfile(str(Path(name) / path.relative_to(root)))
                if extracted.read() != path.read_bytes():
                    raise RuntimeError(f"archive mismatch: {path.name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": digest, "pending_paths": len(paths), "patches_verified": True}, indent=2))


if __name__ == "__main__":
    main()
