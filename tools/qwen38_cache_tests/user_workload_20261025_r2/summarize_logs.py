"""Summarize the first Hermes request without mixing later title/tool requests."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def summarize(path):
    text = path.read_text()
    prompt_tokens = 0
    chunks = []
    samples = []
    seen_decode = False
    last_cache_stats = None
    for line in text.splitlines():
        if "Qwen38ExpertCacheStats" in line:
            last_cache_stats = json.loads(line[line.index("{") :])
    for line in text.splitlines():
        if "Prefill batch," in line:
            if seen_decode:
                break
            match = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
            if match:
                chunks.append(int(match[1]))
                prompt_tokens += int(match[1])
        if "Decode batch," not in line:
            continue
        seen_decode = True
        accept = re.search(r"accept len: ([\d.]+)", line)
        rate = re.search(r"gen throughput \(token/s\): ([\d.]+)", line)
        if accept and rate:
            accept_length, tokens_s = float(accept[1]), float(rate[1])
            samples.append(
                {
                    "time": line[1:20],
                    "accept_length": accept_length,
                    "tokens_per_second": tokens_s,
                    "estimated_cycle_ms": 1000 * accept_length / tokens_s,
                }
            )
    # The first throughput interval includes prefill / time since previous work.
    steady = samples[1:]
    result = {
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "first_request_prompt_tokens": prompt_tokens,
        "first_request_prefill_chunks": chunks,
        "first_request_decode_samples": samples,
        "steady_sample_count": len(steady),
        "method": "First request only; discard first decode log interval; cycle ms uses rounded log accept length / throughput, not exact timestamps.",
    }
    if steady:
        for key in ("tokens_per_second", "accept_length", "estimated_cycle_ms"):
            result["steady_median_" + key] = statistics.median(s[key] for s in steady)
    if last_cache_stats:
        keys = (
            "decode_early_cache",
            "decode_early_cache_invocations",
            "decode_device_gate_invocations",
            "decode_device_gate_hits",
            "decode_device_gate_copy_bytes",
            "cache_hit_rate",
            "cache_slots_total",
            "kv_stage",
            "kv_backed_tokens",
        )
        result["last_process_counter_snapshot"] = {
            k: last_cache_stats.get(k) for k in keys
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"round_two": summarize(args.log)}
    if args.baseline_log:
        baseline = result["baseline_summary"] = summarize(args.baseline_log)
        current = result["round_two"]
        result["observed_median_throughput_change_percent"] = 100 * (
            current["steady_median_tokens_per_second"]
            / baseline["steady_median_tokens_per_second"]
            - 1
        )
        result["comparison_limits"] = [
            "Prompt token counts, output lengths and generated routes differ.",
            "Baseline included an uncalibrated py-spy sampler; round two omitted it.",
            "Adaptive expert layouts and random seeds differ after restart.",
            "This is an observed comparison, not a causal feature speedup measurement.",
        ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
