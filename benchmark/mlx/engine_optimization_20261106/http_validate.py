"""Real sequential HTTP/cache/sampling and concurrency contracts and timings."""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["ar", "mtp", "dflash"], required=True)
p.add_argument("--count", type=int, default=7)
p.add_argument("--output", required=True)
p.add_argument("--tokens", type=int, default=128)
a = p.parse_args()
root = Path("/Users/jay/workspace/sglang")
out = root / a.output
out.parent.mkdir(parents=True, exist_ok=True)
model = Path(
    "/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
)
draft = Path(
    "/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a"
)
port = 39007
base = f"http://127.0.0.1:{port}"


def req(route, body=None):
    r = urllib.request.Request(
        base + route,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=600) as f:
        raw = f.read()
    return (
        json.loads(raw)
        if raw and route not in ["/health", "/flush_cache", "/abort_request"]
        else raw.decode()
    )


def body(prompt, limit=32, **kw):
    return dict(
        rid="df-" + uuid.uuid4().hex,
        input_ids=prompt,
        sampling_params={"temperature": 0, "max_new_tokens": limit, "ignore_eos": True},
        **kw,
    )


r = {
    "mode": a.mode,
    "count": a.count,
    "precision_sources": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("python/sglang/kernels/ops/**/mlx*.py")
    },
    "cases": [],
    "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
}
r["production_sources"] = {
    f: hashlib.sha256((root / f).read_bytes()).hexdigest()
    for f in [
        "python/sglang/kernels/ops/attention/mlx_ragged.py",
        "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py",
    ]
}


def save():
    out.write_text(json.dumps(r, indent=2) + "\n")


started = time.perf_counter()
try:
    for _ in range(300):
        try:
            req("/health")
            break
        except OSError:
            time.sleep(1)
    else:
        raise TimeoutError("health")
    r["health_wait_seconds"] = time.perf_counter() - started
    r["server_info"] = req("/get_server_info")
    r["rss_at_health"] = subprocess.check_output(
        ["ps", "-axo", "pid,ppid,rss,command"], text=True
    )
    save()
    print("HEALTH", r["health_wait_seconds"], flush=True)

    def signature(event):
        return {
            "tokens": event["output_ids"],
            "logprobs": event["meta_info"].get("output_token_logprobs"),
            "top": event["meta_info"].get("output_top_logprobs"),
            "finish": event["meta_info"].get("finish_reason"),
        }

    def request_case(label, payload):
        start = time.perf_counter()
        event = req("/generate", payload)
        elapsed = time.perf_counter() - start
        if "error" in event:
            raise RuntimeError(event)
        assert event["meta_info"]["completion_tokens"] == len(event["output_ids"])
        return {
            "label": label,
            "elapsed_seconds": elapsed,
            "event": event,
            "signature": signature(event),
        }

    for length in (32, 2563):
        prompt = [10, 11, 12, 13] * (length // 4) + [10] * (length % 4)
        req("/flush_cache")
        cold = request_case(
            "cold" + str(length), body(prompt, return_logprob=True, top_logprobs_num=3)
        )
        warm = request_case(
            "warm" + str(length), body(prompt, return_logprob=True, top_logprobs_num=3)
        )
        assert cold["signature"] == warm["signature"], ("prefix", length)
        if length > 256:
            assert warm["event"]["meta_info"]["cached_tokens"] > 0
        r["cases"] += [cold, warm]
        save()
        stopped = body(prompt)
        stopped["sampling_params"].update(
            stop_token_ids=[cold["event"]["output_ids"][5]],
            ignore_eos=False,
            no_stop_trim=True,
        )
        stop = request_case("stop" + str(length), stopped)
        assert stop["event"]["meta_info"]["finish_reason"]["type"] == "stop"
        r["cases"].append(stop)
        save()
        for j in range(2):
            sampled = body(prompt, return_logprob=True, top_logprobs_num=3)
            sampled["sampling_params"].update(
                temperature=0.7, top_k=20, top_p=0.9, sampling_seed=123
            )
            value = request_case("sample" + str(length) + "-" + str(j), sampled)
            r["cases"].append(value)
            save()
            if j == 1:
                assert r["cases"][-1]["signature"] == r["cases"][-2]["signature"], (
                    "seed",
                    length,
                )
    # Natural prompts for single request speed and aggregate concurrent throughput.
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for text in [
            "Explain why the sky is blue.",
            "Write a Python function that computes the first n Fibonacci numbers, with an explanation.",
            "请介绍机器学习中的过拟合，并给出避免过拟合的方法。",
        ]
    ]
    for repeat in range(2):
        for text_id, prompt in enumerate(prompts):
            req("/flush_cache")
            payload = body(prompt, a.tokens, stream=True)
            start = time.perf_counter()
            arrivals = []
            request = urllib.request.Request(
                base + "/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    n = event["meta_info"]["completion_tokens"]
                    if n and (not arrivals or n != arrivals[-1][1]):
                        arrivals.append([time.perf_counter() - start, n])
            elapsed = time.perf_counter() - start
            assert event["meta_info"]["completion_tokens"] == a.tokens
            case = {
                "label": "perf",
                "text_id": text_id,
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "event": event,
                "signature": signature(event),
                "arrivals": arrivals,
                "ttft_seconds": arrivals[0][0],
                "decode_tok_s": (arrivals[-1][1] - arrivals[0][1])
                / (arrivals[-1][0] - arrivals[0][0]),
            }
            r["cases"].append(case)
            save()
            print(
                "PERF",
                repeat,
                text_id,
                a.tokens / elapsed,
                case["decode_tok_s"],
                flush=True,
            )
    req("/flush_cache")
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        events = list(
            pool.map(
                lambda prompt: request_case("concurrent", body(prompt, a.tokens)),
                prompts[:2],
            )
        )
    r["concurrent"] = {"elapsed_seconds": time.perf_counter() - start, "events": events}
    save()
    # Abort through the original SSE stream and then reuse its prefix and slot.
    payload = body([10, 11, 12, 13] * 80, 1024, stream=True)
    events = []
    sent = False
    request = urllib.request.Request(
        base + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            events.append(event)
            if not sent and event.get("meta_info", {}).get("completion_tokens", 0) >= 8:
                req("/abort_request", {"rid": payload["rid"], "abort_all": False})
                sent = True
    assert sent and events[-1]["meta_info"]["finish_reason"]["type"] == "abort"
    r["abort"] = events
    r["cases"].append(
        request_case(
            "post_abort",
            body(payload["input_ids"], return_logprob=True, top_logprobs_num=3),
        )
    )
    r["complete"] = True
    save()
finally:
    r["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    save()
