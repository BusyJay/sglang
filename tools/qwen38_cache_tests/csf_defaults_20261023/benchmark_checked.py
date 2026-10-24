"""Keep the established workloads, rejecting degenerate outputs before timing passes."""

import importlib.util
import ast
import inspect
import re
import json
import os
import subprocess
import sys
from pathlib import Path

source = Path(
    "/workspace/sglang/tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py"
)
spec = importlib.util.spec_from_file_location("qwen38_existing_benchmark", source)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
generate = benchmark.generate
checked_cases = 0


def checked_generate(*args, **kwargs):
    global checked_cases
    case = generate(*args, **kwargs)
    text = case["response_text"] or ""
    if not text.strip() or re.search(r"([^\s])\1{63,}", text):
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        (output / "rejected-output.json").write_text(json.dumps(case, indent=2) + "\n")
        raise AssertionError("Degenerate output: performance result is invalid")
    checked_cases += 1
    if checked_cases == int(os.getenv("ROOT_CSF_CHAT_AFTER", "0")):
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        subprocess.run(
            [
                sys.executable,
                "/results/validate_mtp_chat.py",
                "--command",
                str(output / "command.json"),
                "--output",
                str(output / "chat-correctness.json"),
            ],
            check=True,
        )
    return case


benchmark.generate = checked_generate
if __name__ == "__main__":
    if "--warmup-only" in sys.argv:
        sys.argv.remove("--warmup-only")
        tree = ast.parse(inspect.getsource(benchmark.main))
        body = tree.body[0].body
        for i, node in enumerate(body):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "report" for t in node.targets
            ):
                workload = (
                    'workloads = [workloads[0], ("natural_chat", chat, 512)]'
                    if os.getenv("ROOT_CSF_NATURAL") == "full"
                    else (
                        'workloads = [workloads[0], ("natural_chat", chat, 128)]'
                        if os.getenv("ROOT_CSF_NATURAL") == "1"
                        else "workloads = workloads[:1]"
                    )
                )
                body.insert(i, ast.parse(workload).body[0])
                break
        else:
            raise RuntimeError("benchmark workload boundary changed")
        exec(
            compile(ast.fix_missing_locations(tree), str(source), "exec"),
            benchmark.__dict__,
        )
    benchmark.main()
