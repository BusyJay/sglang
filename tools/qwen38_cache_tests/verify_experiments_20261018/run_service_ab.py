#!/usr/bin/env python3
"""Fresh-worker comparisons using the previously validated production launcher."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREVIOUS = ROOT.parent / "mtp_w4a4_20261018"
spec = importlib.util.spec_from_file_location("mtp_ab", PREVIOUS / "run_ab.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
NAME = "qwen38-verify-service-ab"


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def generate(input_value, tokens):
    ab.api("/flush_cache?timeout=30", {})
    payload = {
        "text" if isinstance(input_value, str) else "input_ids": input_value,
        "stream": True,
        "sampling_params": dict(
            temperature=0, max_new_tokens=tokens, ignore_eos=True, sampling_seed=103
        ),
    }
    request = urllib.request.Request(
        ab.URL + "/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    points, final = [], None
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            final = json.loads(line[6:])
            count = final.get("meta_info", {}).get("completion_tokens", 0)
            if count and (not points or count > points[-1][1]):
                points.append([time.monotonic() - start, count])
    elapsed = time.monotonic() - start
    assert final and final["meta_info"]["completion_tokens"] == tokens, final
    assert final["meta_info"].get("cached_tokens", 0) == 0, final
    assert final["meta_info"].get("num_retractions", 0) == 0, final
    assert len(points) >= 2, points
    return dict(
        text=final["text"],
        meta_info=final["meta_info"],
        wall_s=elapsed,
        ttft_s=points[0][0],
        stream_points=points,
        decode_tokens_per_second=(points[-1][1] - points[0][1])
        / (points[-1][0] - points[0][0]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "validation/service")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--near-limit", action="store_true")
    parser.add_argument(
        "--long-input",
        type=Path,
        default=ROOT / "validation/code-context.json",
    )
    args = parser.parse_args()
    matrix = json.loads((ROOT / "experiment_matrix.json").read_text())
    assert all(v in matrix["variants"] or v == "defaults" for v in args.variants)
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    if any(results.glob("round-*")):
        parser.error("Use a fresh results directory; existing rounds are preserved")
    launch = results / "launch"
    launch.mkdir(exist_ok=True)
    for path in (PREVIOUS / "capacity-fix/launch").glob("*.sh"):
        shutil.copyfile(path, launch / path.name)
    long_input = json.loads(args.long_input.read_text())
    save(results / "long-input.json", long_input)
    save(
        results / "provenance.json",
        dict(
            head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            image=subprocess.check_output(
                ["docker", "image", "inspect", ab.IMAGE, "--format", "{{.Id}}"],
                text=True,
            ).strip(),
            order=args.variants,
            tokens=args.tokens,
            matrix=matrix,
            long_input_sha256=hashlib.sha256(args.long_input.read_bytes()).hexdigest(),
            scope="Fixed saved long input plus CPU-cache and Raft prompts; quality checks are smoke only",
            source_sha256={
                str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (REPO / "python/sglang").rglob("qwen38*.*")
                if p.is_file()
            },
        ),
    )
    for index, variant in enumerate(args.variants):
        folder = results / f"round-{index}-{variant}"
        folder.mkdir(exist_ok=False)
        command = json.loads(
            (PREVIOUS / "capacity-fix/round-0-w4a4/command.json").read_text()
        )
        command[command.index("--name") + 1] = NAME
        for i, value in enumerate(command):
            if value.endswith(":/workspace/results"):
                command[i] = f"{results}:/workspace/results"
        env = matrix["base_env"] | matrix["variants"].get(variant, {})
        if variant == "defaults":
            for key in (
                "SGLANG_QWEN38_W4A16_VERIFY_MODE",
                "SGLANG_QWEN38_PARALLEL_MISS_PLAN",
                "SGLANG_QWEN38_CACHE_STATS_FORMAT",
                "SGLANG_QWEN38_ADAPTIVE_CANDIDATE_MAX",
            ):
                env.pop(key, None)
        offset = command.index("--entrypoint")
        command[offset:offset] = [
            item for k, v in env.items() for item in ("-e", f"{k}={v}")
        ]
        save(folder / "command.json", command)
        with (
            (folder / "service.log").open("w") as log,
            (folder / "gpu-memory.csv").open("w") as memory_log,
        ):
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
                stdout=memory_log,
            )
            try:
                start = time.monotonic()
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited: {folder}")
                    try:
                        ab.api("/health", timeout=2)
                        break
                    except Exception:
                        if time.monotonic() - start > 600:
                            raise RuntimeError(f"Startup timeout: {folder}")
                        time.sleep(2)
                result = dict(
                    variant=variant,
                    startup_s=time.monotonic() - start,
                    server_info=ab.api("/get_server_info"),
                    cases=[],
                )
                assert result["server_info"]["max_total_num_tokens"] == 221248
                print(f"READY {index} {variant} {result['startup_s']:.1f}s", flush=True)
                result["warmup"] = generate(ab.PROMPTS[0], 128)
                cases = [
                    ("short", ab.PROMPTS[0]),
                    ("raft", "纯逻辑推理，不调用任何工具回答：Raft 算法的整体流程。"),
                    ("long", long_input),
                    ("long", long_input),
                ]
                for kind, prompt in cases:
                    row = generate(prompt, args.tokens)
                    row["kind"] = kind
                    result["cases"].append(row)
                    save(folder / "result.json", result)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "kind",
                                    "ttft_s",
                                    "decode_tokens_per_second",
                                    "meta_info",
                                )
                            }
                        ),
                        flush=True,
                    )
                result["quality"] = []
                for prompt, answer in (
                    ("What is 17 * 23? Respond with only the number.", "391"),
                    (
                        "Reverse the string abcdefg. Respond with only the reversed string.",
                        "gfedcba",
                    ),
                    (
                        "What does sum(i*i for i in range(10)) evaluate to in Python? Respond with only the number.",
                        "285",
                    ),
                    ("Solve 3*x + 7 = 28. Respond with only the value of x.", "7"),
                ):
                    ab.api("/flush_cache?timeout=30", {})
                    response = ab.api(
                        "/v1/chat/completions",
                        dict(
                            model=result["server_info"]["served_model_name"],
                            messages=[dict(role="user", content=prompt)],
                            temperature=0,
                            max_tokens=256,
                            chat_template_kwargs=dict(enable_thinking=False),
                        ),
                    )
                    content = response["choices"][0]["message"]["content"]
                    result["quality"].append(
                        dict(
                            prompt=prompt,
                            expected=answer,
                            response=response,
                            passed=content.strip() == answer,
                        )
                    )
                result["server_info_after"] = ab.api("/get_server_info")
                if args.near_limit:
                    result["near_limit"] = generate(
                        "test context " * 109000 + ab.PROMPTS[0], 128
                    )
                    assert (
                        218000
                        <= result["near_limit"]["meta_info"]["prompt_tokens"]
                        < 220000
                    )
                    result["health_after_near_limit"] = ab.api("/health")
                result["memory_after"] = ab.memory()
                save(folder / "result.json", result)
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
                subprocess.run(
                    ["docker", "stop", "--time", "20", NAME],
                    check=False,
                    stdout=subprocess.DEVNULL,
                )
                proc.wait(timeout=60)
        print(f"DONE {index} {variant}", flush=True)


if __name__ == "__main__":
    main()
