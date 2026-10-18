"""Restart only the language worker and measure alternating service candidates."""

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def configure_launcher(path, prepared, cta):
    text = path.read_text()
    for key, value in [
        ("SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH", prepared),
        ("SGLANG_QWEN38_FILL_CTA", cta),
    ]:
        text, count = re.subn(
            r"^export " + key + r"=.*$",
            lambda match: f'export {key}="{value}"',
            text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"Expected exactly one explicit launcher export for {key}")
    path.write_text(text)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", required=True, help="name:prepared:CTA")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context-repeat", type=int, default=20000)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    deployment = Path("/home/jay/workspace/prod/homelab-llm")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    variants = [s.split(":") for s in args.variants]
    assert args.rounds > 0
    assert all(
        len(v) == 3
        and v[1] in ("0", "1")
        and v[2] in [str(i) for i in range(1, 9)] + ["adaptive"]
        for v in variants
    )
    manifest = {
        "variants": variants,
        "rounds": args.rounds,
        "runs": [],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    launcher = deployment / "home-jay/remote-mapping/llm/sglang-fp4-local.sh"
    (output / "launcher-before.sh").write_text(launcher.read_text())
    for repeat in range(args.rounds):
        order = variants if repeat % 2 == 0 else list(reversed(variants))
        for name, prepared, cta in order:
            folder = output / f"{repeat}-{name}"
            folder.mkdir(exist_ok=True)
            # Keep settings in the mounted script, including when the existing
            # container still carries older Compose environment values.
            (folder / "launcher.sh").write_text(
                configure_launcher(launcher, prepared, cta)
            )
            print("START", repeat, name, "prepared", prepared, "CTA", cta, flush=True)
            with (folder / "restart.log").open("w") as log:
                subprocess.run(
                    ["docker", "restart", "--timeout", "45", "homelab-llm"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            start = time.monotonic()
            while True:
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8329/health", timeout=2
                    ) as r:
                        if r.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() - start > 300:
                    raise RuntimeError(f"worker readiness timeout: {folder}")
                time.sleep(2)
            ready_s = time.monotonic() - start
            print("READY", name, round(ready_s, 2), flush=True)
            bench = root / "tools/qwen38_cache_tests/bench_mtp_service.py"
            for stage, extra in [
                ("warmup", ["--case", "0", "--tokens", "192"]),
                ("short", ["--tokens", "1024"]),
                (
                    "long",
                    [
                        "--case",
                        "1",
                        "--tokens",
                        "512",
                        "--context-repeat",
                        str(args.context_repeat),
                    ],
                ),
            ]:
                with (folder / f"{stage}.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(bench),
                            "--stream",
                            "--output",
                            str(folder / f"{stage}.json"),
                            *extra,
                        ],
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                data = json.loads((folder / f"{stage}.json").read_text())
                print(
                    "DONE",
                    repeat,
                    name,
                    stage,
                    [
                        (
                            v["meta_info"]["prompt_tokens"],
                            round(v["decode_tokens_per_second"], 2),
                        )
                        for v in data["cases"]
                    ],
                    flush=True,
                )
            payload = {
                "text": "The first ten prime numbers are",
                "stream": False,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "ignore_eos": True,
                },
            }
            req = urllib.request.Request(
                "http://127.0.0.1:8329/generate",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                canary = json.load(response)
            (folder / "canary.json").write_text(json.dumps(canary, indent=2) + "\n")
            with (folder / "server.log").open("w") as log:
                subprocess.run(
                    ["docker", "logs", "homelab-llm"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            manifest["runs"].append(
                {
                    "repeat": repeat,
                    "name": name,
                    "folder": folder.name,
                    "ready_s": ready_s,
                }
            )
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
