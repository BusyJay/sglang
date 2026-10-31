# SPDX-License-Identifier: Apache-2.0
"""Evidence profiler for a local GLM MLX checkpoint or a bounded synthetic model.

No downloads or new dependencies. Run --help for capture and comparison modes.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

from mlx_profile import Recorder, source_manifest

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return dict(
        exit_code=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def environment(mx):
    return dict(
        python=sys.version,
        executable=sys.executable,
        macos=platform.mac_ver(),
        mlx=mx.__version__,
        mlx_lm=importlib.metadata.version("mlx-lm"),
        device=mx.device_info(),
        git=command(["git", "rev-parse", "HEAD"]),
        status=command(["git", "status", "--short"]),
        source_sha256=source_manifest(ROOT),
        swap=command(["sysctl", "vm.swapusage"]),
        xcode=command(["xcodebuild", "-version"]),
        capture_enabled=os.environ.get("MTL_CAPTURE_ENABLED"),
        tf32=os.environ.get("MLX_ENABLE_TF32"),
    )


def standalone_imports():
    # Package boundaries only: actual repository operators and model run intact.
    # The normal SGLang root imports Torch, unnecessary for this isolated replay.
    for name in (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.gemm",
        "sglang.kernels.ops.moe",
        "sglang.kernels.ops.layernorm",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.linear",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.mlx",
        "sglang.srt.hardware_backend.mlx.models",
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
            sys.modules[name] = module


def tiny_model(g, mx):
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    args = types.SimpleNamespace(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=512,
        n_routed_experts=32,
        num_experts_per_tok=8,
        n_shared_experts=1,
        swiglu_limit=10.0,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        linear_attn_config=dict(
            num_heads=2, head_dim=128, short_conv_kernel_size=4, gate_lower_bound=-5.0
        ),
        rms_norm_eps=1e-5,
        mhc=True,
        hc_mult=4,
        hc_eps=1e-6,
        hc_sinkhorn_iters=5,
        attention_bias=False,
        q_lora_rank=128,
        kv_lora_rank=128,
        num_attention_heads=2,
        qk_nope_head_dim=128,
        v_head_dim=128,
        index_n_heads=2,
        index_head_dim=128,
        index_kpool=4,
        index_topk=8,
        index_kpool_always_select_tail=True,
        vocab_size=512,
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"],
    )
    mx.random.seed(71)
    model = g.Model(args, enable_mtp=True)
    keep = ("A_log", "dt_bias", "hc_", "e_score_correction_bias")
    model.load_weights(
        [
            (k, v.astype(mx.bfloat16))
            for k, v in tree_flatten(model.parameters())
            if not any(s in k for s in keep)
        ],
        strict=False,
    )
    model.load_weights(
        [
            (k, mx.random.normal(v.shape) * 0.01)
            for k, v in tree_flatten(model.parameters())
            if "hc_" in k
        ],
        strict=False,
    )

    def quantization(path, module):
        if hasattr(module, "to_quantized") and ".mlp." in path:
            bits = 6 if "shared_experts" in path else 5 if "down_proj" in path else 4
            return dict(group_size=64, bits=bits)
        return False

    nn.quantize(model, class_predicate=quantization)
    model.load_weights(
        [
            (k, v.astype(mx.float16))
            for k, v in tree_flatten(model.parameters())
            if k.endswith(("scales", "biases"))
        ],
        strict=False,
    )
    model.configure_optimizations("affine_fused")
    model.prepare_for_inference()
    mx.eval(model.parameters())
    return model


def write_report(directory, report):
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# GLM MLX performance evidence",
        "",
        "Host spans measure Python build/enqueue/wait wall time, not GPU kernels. "
        "Sync attribution adds explicit drains and changes graph lifetime/overlap. "
        "Inclusive rows overlap; self rows subtract nested recorded spans.",
        "",
        f"Workload: `{report['workload']}`; synthetic: `{report['synthetic']}`; "
        f"steps: `{report['steps']}`; checkpoint revision: `{report['revision']}`.",
        "",
        "## Alternating measured trials",
        "",
        "| Mode | First ms | Median steady ms | Relative to baseline | Token hash equal |",
        "|---|---:|---:|---:|---|",
    ]
    baseline = report["modes"]["baseline"]["median_ms"]
    for mode, row in report["modes"].items():
        lines.append(
            f"| {mode} | {row['first_ms']:.3f} | {row['median_ms']:.3f} | "
            f"{row['median_ms'] / baseline:.3f}x | {row['matches_baseline']} |"
        )
    for mode in ("host", "sync"):
        if mode not in report["modes"]:
            continue
        lines += [
            "",
            f"## {mode} recorded host spans",
            "",
            "| Operation | Module | Calls | Inclusive ms | Self ms |",
            "|---|---|---:|---:|---:|",
        ]
        for row in report["modes"][mode]["summary"]["timings"][:40]:
            lines.append(
                f"| {row['name']} | {row['module']} | {row['calls']} | "
                f"{row['inclusive_ms']:.3f} | {row['self_ms']:.3f} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Use baseline decode_ms / generated_tokens for committed-token cost. "
        "Verify rows are speculative work, not committed throughput.",
        "- Inspect dispatch.json for shape/dtype changes and returned_none fusion misses. "
        "Descriptors identify operands; they do not prove full-array casts or DRAM traffic.",
        "- graph.dot contains only the still-lazy graph at export time. Evaluated or compiled "
        "regions can be opaque. Cast nodes prove graph conversion, not its measured cost.",
        "- Inspect window.gputrace in Xcode for dispatched kernel names, resources and casts. "
        "Use Metal System Trace for device execution and CPU/GPU gaps; host timeline alone "
        "cannot distinguish bandwidth from compute saturation.",
        "- Compare identical checkpoint, source hashes, prompt hash, steps and context. "
        "This replay excludes HTTP scheduling and network overhead.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_viewer(directory, report)


def write_viewer(directory, report):
    # Embedded local data only; no CDN, telemetry or dependencies.
    payload = json.dumps(report).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>MLX performance evidence</title>
<style>body{font:15px system-ui;margin:32px;background:#111827;color:#e5e7eb}
table{border-collapse:collapse;width:100%;margin:16px 0}td,th{text-align:left;padding:8px;border-bottom:1px solid #374151}
button,select,input{padding:8px;margin:8px;background:#1f2937;color:#fff;border:1px solid #6b7280}
th{cursor:pointer}small{color:#9ca3af}pre{white-space:pre-wrap}.bar{height:10px;background:#38bdf8}
</style><h1>MLX performance evidence</h1>
<p>CPU build/enqueue/wait spans. GPU intervals are in device.trace.json; shader timing is separate.</p>
<div id="baseline"></div><label>Mode <select id="mode"></select></label>
<label>Filter <input id="filter" placeholder="KDA / MoE / model.layers.0"></label>
<p>Click a column to sort. Inclusive durations overlap; self time subtracts recorded children.</p>
<table><thead><tr><th data-k="name">Operation</th><th data-k="module">Module</th><th data-k="calls">Calls</th>
<th data-k="inclusive_ms">Inclusive ms</th><th data-k="self_ms">Self ms</th></tr></thead><tbody id="rows"></tbody></table>
<details><summary>Shape / dtype / fusion dispatch observations</summary><pre id="dispatch"></pre></details>
<script id="data" type="application/json">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent),m=document.getElementById('mode');
const text=(tag,s)=>{const e=document.createElement(tag);e.textContent=s;return e};
document.getElementById('baseline').append(text('p',Object.entries(d.modes).map(([k,v])=>k+': '+v.median_ms.toFixed(3)+' ms; token parity '+v.matches_baseline).join(' | ')));
Object.keys(d.modes).filter(k=>d.modes[k].summary).forEach(k=>{const e=text('option',k);e.value=k;m.append(e)});
let sort='self_ms',descending=true;
function render(){const s=d.modes[m.value].summary,q=document.getElementById('filter').value.toLowerCase();
const rs=s.timings.filter(r=>(r.name+' '+r.module).toLowerCase().includes(q)).sort((a,b)=>{
const v=typeof a[sort]==='number'?a[sort]-b[sort]:String(a[sort]).localeCompare(String(b[sort]));return descending?-v:v});
const t=document.getElementById('rows');t.replaceChildren();rs.forEach(r=>{const tr=document.createElement('tr');
['name','module','calls','inclusive_ms','self_ms'].forEach(k=>tr.append(text('td',typeof r[k]==='number'&&k.endsWith('_ms')?r[k].toFixed(3):r[k])));t.append(tr)});
document.getElementById('dispatch').textContent=JSON.stringify(s.dispatch.filter(r=>r.op.toLowerCase().includes(q)),null,2)}
m.onchange=render;document.getElementById('filter').oninput=render;document.querySelectorAll('th').forEach(t=>t.onclick=()=>{descending=sort===t.dataset.k?!descending:true;sort=t.dataset.k;render()});render();
</script>""".replace("PAYLOAD", payload)
    (directory / "viewer.html").write_text(html)


def run(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    standalone_imports()
    g = importlib.import_module("sglang.srt.hardware_backend.mlx.models.glm5_next")
    from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing, clone_cache

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = environment(mx)
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    if args.tiny:
        model = tiny_model(g, mx)
        prompt = mx.array([list(range(1, args.context + 1))])
    else:
        directory = args.model_dir.resolve()
        if not (directory / "config.json").is_file():
            raise ValueError("--model-dir must be a local checkpoint; no downloads")
        weight_bytes = sum(p.stat().st_size for p in directory.glob("*.safetensors"))
        recommended = mx.device_info().get("max_recommended_working_set_size", 0)
        if not weight_bytes or (
            recommended and weight_bytes + 2 * 1024**3 > recommended
        ):
            raise ValueError(
                "Local checkpoint exceeds recommended working set plus 2 GiB reserve, "
                "or no safetensors exist; use --tiny on this machine"
            )
        # Match normal serving process defaults without changing any OS limit.
        if recommended:
            mx.set_wired_limit(recommended)
        env["process_wired_limit_bytes"] = recommended
        (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        from transformers import AutoTokenizer

        from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model

        model = load_mtp_model(directory, verify_mode=args.verify_mode)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        text = (
            args.prompt_file.read_text()
            if args.prompt_file
            else "Explain hash table collision handling."
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        prompt = mx.array([ids])
    model.configure_mtp_verification(args.verify_mode)
    inventory = [
        dict(name=k, shape=list(v.shape), dtype=str(v.dtype), bytes=v.nbytes)
        for k, v in tree_flatten(model.parameters())
    ]
    (output / "weights.json").write_text(json.dumps(inventory, indent=2) + "\n")
    base = model.make_cache()
    mx.eval(model(prompt, base), [c.cache for c in base])
    chain = mx.array([list(range(10, 10 + args.rows))])

    def replay(rec=None, graph=False):
        scope = rec.span if rec else lambda *a, **k: contextlib.nullcontext()
        started = time.perf_counter()
        if args.workload == "verify":
            cache = [clone_cache(c) for c in base]
            with scope("forced_verify", "workload"):
                logits, hidden, snapshots = model.verify_mtp_rows(chain, cache)
                if graph:
                    mx.export_to_dot(str(output / "graph.dot"), logits, hidden)
                mx.eval(logits, hidden, [[c.cache for c in row] for row in snapshots])
            tokens = mx.argmax(logits, -1).tolist()[0]
            stats = dict(verify_rows=args.rows)
            decode_ms = (time.perf_counter() - started) * 1000
            prefill_ms = 0.0
        else:
            engine = MlxMTPServing(model, args.steps)
            cache = model.make_cache()
            try:
                with scope("prefill", "workload"):
                    hidden = engine.prefill("profile", prompt, cache)
                    token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
                    tokens = [token.item()]
                prefill_ms = (time.perf_counter() - started) * 1000
                decode_start = time.perf_counter()
                with scope("decode_loop", "workload"):
                    while len(tokens) < args.tokens:
                        engine.set_remaining("profile", args.tokens - len(tokens))
                        logits = engine.decode("profile", token, cache)
                        if graph and len(tokens) == 1:
                            mx.export_to_dot(str(output / "graph.dot"), logits)
                        with scope("sample_and_materialize", "sampling"):
                            token = mx.argmax(logits, -1)[:, None]
                            tokens.append(token.item())
                    mx.eval(
                        engine.states["profile"].head.cache, [c.cache for c in cache]
                    )
                    mx.synchronize()
                decode_ms = (time.perf_counter() - decode_start) * 1000
                stats = engine.stats()
            finally:
                engine.remove("profile")
        return dict(
            elapsed_ms=(time.perf_counter() - started) * 1000,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            generated_tokens=len(tokens) - 1 if args.workload == "generate" else 0,
            tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            stats=stats,
        )

    modes = ["baseline", "host"] + (["sync"] if args.sync else [])
    recorders = {
        mode: Recorder(mx, mode, args.max_events)
        for mode in modes
        if mode != "baseline"
    }

    def trial(mode):
        if mode == "baseline":
            return replay()
        rec = recorders[mode]
        with rec.instrument(g, model, MlxMTPServing):
            return replay(rec)

    first = {mode: trial(mode) for mode in modes}
    for _ in range(args.warmup):
        replay()
    # Warm the instrumented/forced-sync graph variants too; exclude warm events.
    for mode in modes[1:]:
        for _ in range(args.warmup):
            trial(mode)
        recorders[mode] = Recorder(mx, mode, args.max_events)
    samples = {mode: [] for mode in modes}
    order = []
    for repeat in range(args.rounds):
        for mode in modes if repeat % 2 == 0 else modes[::-1]:
            row = trial(mode)
            samples[mode].append(row)
            order.append(dict(round=repeat, mode=mode, elapsed_ms=row["elapsed_ms"]))
            print(json.dumps(dict(mode=mode, round=repeat, **row)), flush=True)
    reference = samples["baseline"][0]["tokens_sha256"]
    report = dict(
        schema_version=1,
        synthetic=args.tiny,
        workload=args.workload,
        revision=args.revision,
        model_dir=str(args.model_dir) if args.model_dir else None,
        steps=args.steps,
        verify_mode=args.verify_mode,
        rows=args.rows,
        tokens=args.tokens,
        prompt_tokens=prompt.size,
        prompt_tokens_sha256=hashlib.sha256(
            json.dumps(prompt.tolist()).encode()
        ).hexdigest(),
        warmup=args.warmup,
        trial_order=order,
        modes={},
        first_trial_scope="after model preparation and base-cache prefill; not process-cold loading",
        capture_scope="one replay including prefill for generate; GPU captures may retain weight buffers",
    )
    for mode in modes:
        rows = samples[mode]
        report["modes"][mode] = dict(
            first_ms=first[mode]["elapsed_ms"],
            first_trial=first[mode],
            samples=rows,
            median_ms=statistics.median(r["elapsed_ms"] for r in rows),
            median_decode_ms=statistics.median(r["decode_ms"] for r in rows),
            matches_baseline=all(
                r["tokens_sha256"] == reference for r in [first[mode], *rows]
            ),
        )
        if mode != "baseline":
            rec = recorders[mode]
            rec.save(output / (mode + ".trace.json"))
            report["modes"][mode]["summary"] = rec.summary()
            report["modes"][mode]["dropped_events"] = rec.dropped
            (output / (mode + ".dispatch.json")).write_text(
                json.dumps(rec.summary()["dispatch"], indent=2) + "\n"
            )
    report["capture"] = "not requested"
    if args.capture:
        mx.synchronize()
        mx.metal.start_capture(str(output / "window.gputrace"))
        try:
            replay()
            mx.synchronize()
        finally:
            mx.metal.stop_capture()
        report["capture"] = (
            "one warmed uninstrumented replay; excluded from measurements"
        )
    if args.graph:
        graph_rec = Recorder(mx, graph_dir=output / "graphs")
        with graph_rec.instrument(g, model, MlxMTPServing):
            replay(graph_rec, graph=True)
        graph_rec.save(output / "graph-index.trace.json")
    report["memory"] = dict(
        active_bytes=mx.get_active_memory(),
        peak_bytes=mx.get_peak_memory(),
        cache_bytes=mx.get_cache_memory(),
        swap_after=command(["sysctl", "vm.swapusage"]),
    )
    write_report(output, report)
    if not all(r["matches_baseline"] for r in report["modes"].values()):
        raise RuntimeError("Instrumentation changed generated tokens; see results.json")
    print(str(output / "REPORT.md"))


def compare(args):
    documents = [
        json.loads((p / "results.json").read_text()) for p in (args.before, args.after)
    ]
    keys = (
        "synthetic",
        "workload",
        "revision",
        "steps",
        "verify_mode",
        "rows",
        "tokens",
        "prompt_tokens_sha256",
    )
    mismatch = {
        k: [d.get(k) for d in documents]
        for k in keys
        if documents[0].get(k) != documents[1].get(k)
    }
    reference = [d["modes"]["baseline"] for d in documents]
    print(
        json.dumps(
            dict(
                configuration_mismatch=mismatch,
                median_total_ms=[d["median_ms"] for d in reference],
                median_decode_ms=[d["median_decode_ms"] for d in reference],
                token_hash_equal=reference[0]["samples"][0]["tokens_sha256"]
                == reference[1]["samples"][0]["tokens_sha256"],
                after_over_before=reference[1]["median_decode_ms"]
                / reference[0]["median_decode_ms"],
                evidence="separate runs; repeat in alternating checkout order before claiming speedup",
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--output", type=Path)
    run_parser = sub.add_parser("run")
    model = run_parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--tiny", action="store_true")
    model.add_argument("--model-dir", type=Path)
    run_parser.add_argument(
        "--output", type=Path, required=True, help="new persistent directory"
    )
    run_parser.add_argument(
        "--revision",
        default="unrecorded",
        help="pin checkpoint revision for full-model reports",
    )
    run_parser.add_argument(
        "--workload", choices=("generate", "verify"), default="generate"
    )
    run_parser.add_argument("--steps", type=int, default=1)
    run_parser.add_argument(
        "--verify-mode", choices=("serial", "head", "ffn"), default="ffn"
    )
    run_parser.add_argument("--rows", type=int, default=2)
    run_parser.add_argument("--tokens", type=int, default=32)
    run_parser.add_argument(
        "--context", type=int, default=9, help="synthetic prompt length"
    )
    run_parser.add_argument("--prompt-file", type=Path)
    run_parser.add_argument("--warmup", type=int, default=2)
    run_parser.add_argument("--rounds", type=int, default=5)
    run_parser.add_argument("--max-events", type=int, default=200000)
    run_parser.add_argument(
        "--sync",
        action="store_true",
        help="separate intrusive synchronized attribution",
    )
    run_parser.add_argument("--graph", action="store_true")
    run_parser.add_argument(
        "--capture",
        action="store_true",
        help="requires MTL_CAPTURE_ENABLED=1 before launch",
    )
    diff = sub.add_parser("compare")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    metal = sub.add_parser(
        "metal-export", help="export actual device intervals from an Xcode .trace"
    )
    metal.add_argument("trace", type=Path)
    metal.add_argument("--output", type=Path, required=True)
    metal.add_argument(
        "--pid", type=int, help="defaults to the launched PID recorded in TOC"
    )
    args = parser.parse_args()
    if args.action == "compare":
        compare(args)
    elif args.action == "metal-export":
        import xml.etree.ElementTree as ET

        from mlx_metal_trace import analyze

        args.output.mkdir(parents=True, exist_ok=True)
        toc = args.output / "toc.xml"
        xml = args.output / "gpu.xml"
        for query, destination in (
            (["--toc"], toc),
            (
                [
                    "--xpath",
                    '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]',
                ],
                xml,
            ),
        ):
            subprocess.run(
                [
                    "xcrun",
                    "xctrace",
                    "export",
                    "--input",
                    str(args.trace),
                    *query,
                    "--output",
                    str(destination),
                ],
                check=True,
            )
        process = ET.parse(toc).find("./run/info/target/process")
        pid = args.pid if args.pid else int(process.attrib["pid"])
        print(json.dumps(analyze(xml, args.output, pid), indent=2))
    elif args.action == "doctor":
        import mlx.core as mx

        doc = environment(mx)
        doc["xctrace_templates"] = command(["xcrun", "xctrace", "list", "templates"])
        result = json.dumps(doc, indent=2) + "\n"
        if args.output:
            args.output.write_text(result)
        print(result)
    else:
        if (
            args.rounds < 1
            or args.warmup < 0
            or args.tokens < 2
            or not 1 <= args.rows <= 17
        ):
            parser.error("rounds>=1, warmup>=0, tokens>=2 and rows in [1,17] required")
        if args.tiny and not 1 <= args.context <= 128:
            parser.error("tiny context must be in [1,128]")
        if not -1 <= args.steps <= 16:
            parser.error("steps must be in [-1,16]")
        if args.capture and os.environ.get("MTL_CAPTURE_ENABLED") != "1":
            parser.error("launch with MTL_CAPTURE_ENABLED=1 for capture")
        run(args)


if __name__ == "__main__":
    main()
