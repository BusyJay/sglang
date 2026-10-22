"""Profile representative requests without Python stack collection.
Use actual streaming benchmark fixtures to verify that CPU work really occurs.
"""

import importlib.util, json, os, signal, subprocess, time, urllib.request
from pathlib import Path
from transformers import AutoTokenizer

r = Path("/results")
repo = Path("/workspace/sglang")
spec = importlib.util.spec_from_file_location(
    "bench",
    repo
    / "tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py",
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
# Reuse exactly the environment of the already-recorded LFU profiles.
env = dict(
    os.environ,
    SGLANG_ENABLE_QWEN38_CPU_HYBRID="0",
    SGLANG_QWEN38_CPU_CORES="2,3,4,5",
    SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH="1",
    SGLANG_QWEN38_FILL_CTA="adaptive",
    SGLANG_QWEN38_STARTUP_CACHE="auto",
    SGLANG_QWEN38_SOURCE_CACHE_DIR="/dev/shm/sglang-qwen38-startup",
    SGLANG_QWEN38_HOST_TOKEN_EMBEDDING="1",
    SGLANG_QWEN38_BORROW_PREFILL_ARENA="1",
    SGLANG_QWEN38_PLE_PACKED_CACHE="1",
    SGLANG_QWEN38_CACHE_W4A16="1",
    SGLANG_QWEN38_CACHE_W4A16_BACKEND="b12x",
    SGLANG_QWEN38_CACHE_PREFILL_W4A16="0",
    SGLANG_QWEN38_FP32_NORM_FUSION="1",
    SGLANG_QWEN38_CACHE_W4A16_SPLIT_K="1",
)
base = json.loads((r / "base-command.json").read_text())["command"]
base += ["--random-seed", "42"]
tok = AutoTokenizer.from_pretrained(
    base[base.index("--model-path") + 1], local_files_only=True
)
phrase = tok.encode(
    "Explain how an expert cache handles a miss. ", add_special_tokens=False
)


def fixture(length, index):
    return [
        1000 + ((index + 1) * 2654435761 + step * 40503) % 100000 for step in range(64)
    ] + (phrase * ((length + len(phrase) - 1) // len(phrase)))[:length]


chat = tok.apply_chat_template(
    [
        {
            "role": "user",
            "content": "Explain in detail how a CPU and GPU can cooperate to run a mixture of experts language model when GPU memory is limited. Discuss cache misses, data movement, latency, and dynamic cache resizing.",
        }
    ],
    tokenize=True,
    add_generation_prompt=True,
)
if hasattr(chat, "get"):
    chat = chat["input_ids"]
if chat and isinstance(chat[0], list):
    chat = chat[0]


def post(path, payload=None):
    req = urllib.request.Request(
        "http://127.0.0.1:30938" + path,
        json.dumps(payload or {}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.read().decode()


for mode in ["n0", "n1"]:
    out = r / ("active-" + mode)
    out.mkdir(exist_ok=True)
    cmd = list(base)
    i = cmd.index("--model-loader-extra-config") + 1
    cfg = json.loads(cmd[i])
    cfg.update(
        cpu_hybrid_mode="layered_lfu",
        layered_cpu_max_experts=int(mode[1:]),
        stats_log_interval=576,
    )
    cmd[i] = json.dumps(cfg)
    (out / "command.json").write_text(json.dumps({"command": cmd}, indent=2))
    process = None
    try:
        with (out / "server.log").open("w") as log:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("server exited")
                if time.monotonic() - start > 600:
                    raise TimeoutError("startup")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:30938/ready", timeout=3
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            warm = bench.generate(fixture(1024, 0), 128)
            (out / "warmup.json").write_text(json.dumps(warm, indent=2))
            for label, ids in [("short", fixture(1024, 1)), ("natural", chat)]:
                dest = out / label
                dest.mkdir(exist_ok=True)
                post(
                    "/start_profile",
                    {
                        "output_dir": str(dest),
                        "activities": ["CPU", "GPU"],
                        "with_stack": False,
                        "record_shapes": False,
                        "profile_id": mode + "-" + label,
                    },
                )
                result = bench.generate(ids, 512)
                (dest / "request.json").write_text(json.dumps(result, indent=2))
                post("/stop_profile")
                print(mode, label, "captured", flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
