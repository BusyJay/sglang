"""Keep the established workloads, rejecting degenerate outputs before timing passes."""

import importlib.util
import ast
import inspect
import re
import json
import sys
from pathlib import Path

source = Path(
    "/workspace/sglang/tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py"
)
spec = importlib.util.spec_from_file_location("qwen38_existing_benchmark", source)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
generate = benchmark.generate


def checked_generate(*args, **kwargs):
    case = generate(*args, **kwargs)
    text = case["response_text"] or ""
    if not text.strip() or re.search(r"([^\s])\1{63,}", text):
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        (output / "rejected-output.json").write_text(json.dumps(case, indent=2) + "\n")
        raise AssertionError("Degenerate output: performance result is invalid")
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
                body.insert(i, ast.parse("workloads = workloads[:1]").body[0])
                break
        else:
            raise RuntimeError("benchmark workload boundary changed")
        exec(
            compile(ast.fix_missing_locations(tree), str(source), "exec"),
            benchmark.__dict__,
        )
    benchmark.main()
