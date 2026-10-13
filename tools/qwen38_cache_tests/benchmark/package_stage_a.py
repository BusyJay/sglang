#!/usr/bin/env python3
"""Package committed benchmark-only patch and verify exact application in isolation."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "d1d85bf29b855e511d2a96f9a7f6cec73fc2bb1f"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-log", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError("output must be a new directory")
    target = git("rev-parse", "HEAD").decode().strip()
    changed = git("diff", "--name-only", BASE, target).decode().splitlines()
    allowed = ("tools/qwen38_cache_tests/benchmark/", "tools/qwen38_cache_tests/test_cpu_benchmark.py")
    if not changed or any(not any(p.startswith(a) for a in allowed) for p in changed):
        raise ValueError("commit diff includes changes outside stage A benchmark")
    patch = git("diff", "--binary", BASE, target)
    out.mkdir(parents=True)
    patch_path = out / "01-stage-a-benchmark.patch"
    patch_path.write_bytes(patch)
    base_tree = git("rev-parse", BASE + "^{tree}").decode().strip()
    target_tree = git("rev-parse", target + "^{tree}").decode().strip()
    with tempfile.TemporaryDirectory(prefix="qwen38-stage-a-apply-") as directory:
        checkout = Path(directory) / "checkout"
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(checkout)], check=True)
        git("checkout", "--quiet", "--detach", BASE, cwd=checkout)
        git("apply", "--check", str(patch_path), cwd=checkout)
        git("apply", "--index", str(patch_path), cwd=checkout)
        applied_tree = git("write-tree", cwd=checkout).decode().strip()
        if applied_tree != target_tree:
            raise ValueError("applied tree differs from target")
        subprocess.run(["python3", "-m", "unittest", "tools.qwen38_cache_tests.test_cpu_benchmark", "-v"],
                       cwd=checkout, stdout=(out / "isolated-cpu.log").open("w"), stderr=subprocess.STDOUT, check=True)
        subprocess.run(["python3", "-m", "compileall", "-q", "tools/qwen38_cache_tests/benchmark"], cwd=checkout, check=True)
        for action in ("prepare-inputs", "freeze-answer", "validate", "run", "replay", "seal-trace",
                       "synthetic-trace", "summarize", "candidate-profile"):
            subprocess.run(["python3", "-m", "tools.qwen38_cache_tests.benchmark", action, "--help"],
                           cwd=checkout, stdout=subprocess.DEVNULL, check=True)
    for name in ("GPU_HANDOFF.md", "RESULT_SCHEMA.md", "VALIDATION.json", "suite.example.json", "machine.example.json", "STAGE_A_REVIEW.md", "STAGE_A_REVIEW_RESPONSE.md"):
        (out / name).write_bytes(git("show", f"{target}:tools/qwen38_cache_tests/benchmark/{name}"))
    shutil.copyfile(args.cpu_log, out / "cpu-regression.log")
    write(out / "package.json", {"stage": "A", "base_sha": BASE, "base_tree": base_tree,
          "performance_base_sha": "384c83da44a4334dcaa74c474195ab960126e029",
          "target_sha": target, "target_tree": target_tree, "applied_tree": applied_tree,
          "patches": [{"file": patch_path.name, "sha256": hashlib.sha256(patch).hexdigest(), "dependencies": [BASE]}],
          "verification": {"git_apply_check": "passed", "isolated_apply_tree": "passed",
                           "isolated_cpu_protocol": "passed", "cli_help": "passed", "python_compile": "passed",
                           "CUDA_compile": "not_run", "GPU_correctness": "not_run", "performance": "not_run"}})
    (out / "README.md").write_text("# Qwen3.8 Stage A benchmark handoff\n\n"
        "Start with GPU_HANDOFF.md. One benchmark-only patch; no 06/05/01.\n"
        "package.json identifies exact base, target, verified applied tree and patch hash.\n"
        "VALIDATION.json and CPU logs describe CPU evidence and all unverified GPU items.\n"
        "Run sha256sum -c SHA256SUMS, apply the patch to its exact clean base, then smoke/correctness before baseline.\n"
        "Return the entire result directory for review. Stage B waits for reviewed GPU baseline.\n")
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (out / "SHA256SUMS").write_text("".join(sums))
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    print(json.dumps({"directory": str(out), "archive": str(archive), "target": target,
                      "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
