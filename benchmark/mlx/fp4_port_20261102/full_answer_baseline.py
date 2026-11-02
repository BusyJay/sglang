"""Record complete-answer quality and streaming throughput of a resident server."""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:30007")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
from mlx_lm.utils import load_tokenizer

tok = load_tokenizer(args.checkpoint)
repo = Path.cwd()


def request(route, body=None):
    req = urllib.request.Request(
        args.base_url + route,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode()


cases = [
    ("multiply", "What is 17 * 23? Give the answer concisely.", "391"),
    (
        "algebra",
        "Solve 3*x + 7 = 52. Explain briefly and give the final value of x.",
        "15",
    ),
    (
        "fibonacci",
        "Write a Python function fib(n) returning the first n Fibonacci numbers starting 0, 1. Use iteration. Give concise explanation and code.",
        None,
    ),
    ("overfitting", "请简洁说明过拟合，并列举三种应对方法。", None),
]
result = dict(
    schema=1,
    checkpoint=str(args.checkpoint),
    revision=args.checkpoint.name,
    configuration=request("/get_server_info"),
    model_info=request("/get_model_info"),
    code_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    platform=platform.platform(),
    environment={
        name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors.index.json")
    },
    samples=[],
    chat_checks=[],
    swap_before=subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
    scope="Four explicit tasks, short/2050-token contexts, first request plus two warm repetitions. Greedy native MTP2. Task-specific checks do not establish general accuracy. TTFT is observed first SSE token; decode throughput excludes time before that token. Cold means flushed prefix cache, not cold weight pages.",
)
result["source_sha256"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        Path("python/sglang/srt/hardware_backend/mlx/models/glm5_next.py"),
        Path("python/sglang/srt/hardware_backend/mlx/models/glm5_fp4.py"),
        Path("python/sglang/srt/hardware_backend/mlx/models/mlx_fp4_cache.py"),
        Path("python/sglang/srt/hardware_backend/mlx/mtp_serving.py"),
    ]
}


def save():
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


def code_check(answer):
    outcomes = []
    for block in answer.split("```")[1::2]:
        code = block.removeprefix("python").strip()
        try:
            tree = ast.parse(code)
            functions = [
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "fib"
            ]
            if len(functions) != 1:
                continue
            function = functions[0]
            if any(
                isinstance(n, (ast.Import, ast.ImportFrom))
                or isinstance(n, ast.Attribute)
                and n.attr.startswith("__")
                or isinstance(n, ast.Name)
                and n.id.startswith("__")
                for n in ast.walk(function)
            ):
                outcomes.append(
                    dict(passed=False, error="Unsupported generated constructs")
                )
                continue
            isolated = ast.unparse(ast.Module(body=[function], type_ignores=[]))
            harness = (
                "namespace={'__builtins__': {'range':range,'list':list,'int':int,'len':len,'ValueError':ValueError,'TypeError':TypeError,'isinstance':isinstance,'max':max,'min':min}}\nexec("
                + repr(isolated)
                + ",namespace)\nf=namespace['fib']\nchecks=[f(0)==[],f(1)==[0],f(2)==[0,1],f(10)==[0,1,1,2,3,5,8,13,21,34]]\nprint(checks)\nassert all(checks)\n"
            )
            process = subprocess.run(
                [sys.executable, "-I", "-c", harness],
                timeout=5,
                capture_output=True,
                text=True,
            )
            outcomes.append(
                dict(
                    passed=process.returncode == 0,
                    stdout=process.stdout,
                    error=process.stderr,
                )
            )
        except Exception as error:
            outcomes.append(dict(passed=False, error=str(error)))
    return dict(passed=any(v["passed"] for v in outcomes), executions=outcomes)


def quality(case, answer, finish):
    completed = finish.get("type") == "stop" and bool(answer.strip())
    if case in ("multiply", "algebra"):
        target = "391" if case == "multiply" else "15"
        return dict(
            completed=completed,
            passed=completed
            and bool(re.search(r"(?<!\d)" + target + r"(?!\d)", answer)),
            method="Expected numerical answer in final answer; reasoning saved for inspection",
        )
    if case == "fibonacci":
        check = code_check(answer)
        check.update(
            completed=completed,
            passed=completed and check["passed"],
            method="Execute only isolated fib function for n=0,1,2,10 in a bounded child process",
        )
        return check
    groups = [
        ["训练", "train"],
        ["泛化", "测试", "验证", "general"],
        ["正则", "regular"],
        ["dropout", "丢弃", "早停", "early", "交叉验证", "数据增强"],
    ]
    matches = [
        any(term.lower() in answer.lower() for term in group) for group in groups
    ]
    return dict(
        completed=completed,
        passed=completed and all(matches),
        method="Keyword rubric for train/generalization and mitigation coverage; requires human review, not a formal accuracy score",
        rubric=matches,
    )


def generate(case, prompt, label):
    payload = dict(
        rid="baseline-" + case + "-" + str(time.time_ns()),
        input_ids=prompt,
        sampling_params=dict(temperature=0, max_new_tokens=2048, ignore_eos=False),
        stream=True,
    )
    req = urllib.request.Request(
        args.base_url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    arrivals = []
    last = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event)
            last = event
            n = event.get("meta_info", {}).get("completion_tokens", 0)
            if n and (not arrivals or n != arrivals[-1][1]):
                arrivals.append([time.perf_counter() - start, n])
    elapsed = time.perf_counter() - start
    assert last is not None and arrivals
    ids = last["output_ids"]
    meta = last["meta_info"]
    assert len(ids) == meta["completion_tokens"]
    decoded = tok.decode(ids, skip_special_tokens=False)
    reasoning, sep, answer = decoded.partition("</think>")
    answer = re.sub(
        r"<\|(?:endoftext|user|assistant|observation)\|>.*$",
        "",
        answer if sep else decoded,
        flags=re.S,
    ).strip()
    row = dict(
        case=case,
        label=label,
        input_ids=prompt,
        output_ids=ids,
        output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        decoded=decoded,
        reasoning=reasoning if sep else "",
        answer=answer,
        has_think_end=bool(sep),
        event=last,
        arrivals_seconds_tokens=arrivals,
        elapsed_seconds=elapsed,
        ttft_seconds=arrivals[0][0],
        decode_tokens_per_second=(arrivals[-1][1] - arrivals[0][1])
        / (arrivals[-1][0] - arrivals[0][0])
        if len(arrivals) > 1
        else None,
        end_to_end_tokens_per_second=len(ids) / elapsed,
        quality=quality(case, answer, meta["finish_reason"]),
    )
    return row


save()
for context in ("short", "2050"):
    for case, text, _ in cases:
        prompt = tok.apply_chat_template(
            [dict(role="user", content=text)], tokenize=True, add_generation_prompt=True
        )
        if context == "2050":
            pad = tok.encode(
                "Background notes: "
                + "Machine learning uses data and algorithms. " * 2050
            )
            prompt = pad[: 2050 - len(prompt)] + prompt
            assert len(prompt) == 2050
        request("/flush_cache")
        for repeat in range(3):
            label = f"{context}-{repeat}"
            row = generate(case, prompt, label)
            result["samples"].append(row)
            save()
            print(
                "COMPLETE",
                case,
                label,
                "tokens",
                len(row["output_ids"]),
                "finish",
                row["event"]["meta_info"]["finish_reason"],
                "quality",
                row["quality"]["passed"],
                "TTFT",
                round(row["ttft_seconds"], 3),
                "decode_tps",
                round(row["decode_tokens_per_second"], 3),
                flush=True,
            )
# Verify public API emits separate complete reasoning and answer fields.
body = dict(
    model=result["configuration"]["model_path"],
    messages=[dict(role="user", content=cases[0][1])],
    temperature=0,
    max_tokens=2048,
    chat_template_kwargs=dict(enable_thinking=True),
)
start = time.perf_counter()
response = request("/v1/chat/completions", body)
message = response["choices"][0]["message"]
result["chat_checks"].append(
    dict(
        response=response,
        elapsed_seconds=time.perf_counter() - start,
        passed=bool(message.get("reasoning_content"))
        and "391" in (message.get("content") or "")
        and response["choices"][0]["finish_reason"] == "stop",
    )
)
save()
# Two simultaneous full answers, without another model process.
request("/flush_cache")
start = time.perf_counter()
prompts = [
    tok.apply_chat_template(
        [dict(role="user", content=text)], tokenize=True, add_generation_prompt=True
    )
    for _, text, _ in cases[:2]
]
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    rows = list(
        pool.map(
            lambda pair: generate(pair[0], pair[1], "concurrent"),
            [(cases[i][0], prompts[i]) for i in range(2)],
        )
    )
result["concurrent"] = dict(samples=rows, elapsed_seconds=time.perf_counter() - start)
result["concurrent"]["aggregate_end_to_end_tokens_per_second"] = (
    sum(len(r["output_ids"]) for r in rows) / result["concurrent"]["elapsed_seconds"]
)
result["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
result["summary"] = {}
for context in ("short", "2050"):
    rows = [r for r in result["samples"] if r["label"].startswith(context)]
    warm = [r for r in rows if not r["label"].endswith("-0")]
    result["summary"][context] = dict(
        requests=len(rows),
        finished=sum(r["quality"]["completed"] for r in rows),
        task_checks_passed=sum(r["quality"]["passed"] for r in rows),
        cold_ttft_seconds=[
            r["ttft_seconds"] for r in rows if r["label"].endswith("-0")
        ],
        warm_ttft_median_seconds=statistics.median(r["ttft_seconds"] for r in warm),
        warm_decode_median_tokens_per_second=statistics.median(
            r["decode_tokens_per_second"] for r in warm
        ),
        warm_decode_min_tokens_per_second=min(
            r["decode_tokens_per_second"] for r in warm
        ),
        warm_decode_max_tokens_per_second=max(
            r["decode_tokens_per_second"] for r in warm
        ),
        repetition_token_exact=all(
            len({r["output_sha256"] for r in rows if r["case"] == case}) == 1
            for case, _, _ in cases
        ),
    )
result["complete"] = True
save()
print("SUMMARY", json.dumps(result["summary"]), flush=True)
