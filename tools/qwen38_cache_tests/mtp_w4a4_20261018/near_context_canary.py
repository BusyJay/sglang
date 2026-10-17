#!/usr/bin/env python3
"""Check that restored KV capacity supports a real near-limit A4 request."""

import json
import subprocess
import time
from pathlib import Path

import run_ab as bench


def main():
    root = Path(__file__).resolve().parent / "capacity-fix"
    folder = root / "near-context"
    folder.mkdir(exist_ok=True)
    command = json.loads((root / "round-0-w4a4/command.json").read_text())
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (folder / "service.log").open("w") as log, (folder / "gpu-memory.csv").open(
        "w"
    ) as memlog:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
                "--format=csv",
                "-lms",
                "100",
            ],
            stdout=memlog,
        )
        try:
            start = time.monotonic()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Canary server exited")
                try:
                    bench.api("/health", timeout=2)
                    break
                except Exception:
                    if time.monotonic() - start > 180:
                        raise RuntimeError("Canary startup timed out")
                    time.sleep(2)
            info = bench.api("/get_server_info")
            assert info["max_total_num_tokens"] == 221248, info["max_total_num_tokens"]
            print(
                "READY: A4, KV=221248; submitting approximately 218K prompt", flush=True
            )
            result = {
                "server_info": info,
                "case": bench.generate(
                    "test context " * 109000 + bench.PROMPTS[0], 128
                ),
            }
            meta = result["case"]["meta_info"]
            assert 218000 <= meta["prompt_tokens"] < 220000, meta
            assert meta["cached_tokens"] == 0 and meta["num_retractions"] == 0, meta
            result["health_after"] = bench.api("/health")
            result["status"] = "passed"
            (folder / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "ttft_s": result["case"]["ttft_s"],
                        "meta_info": meta,
                    }
                ),
                flush=True,
            )
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
            subprocess.run(
                ["docker", "stop", "--time", "20", bench.NAME],
                stdout=subprocess.DEVNULL,
                check=False,
            )
            proc.wait(timeout=60)


if __name__ == "__main__":
    main()
