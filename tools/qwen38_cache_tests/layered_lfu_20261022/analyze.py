import json, statistics
from pathlib import Path
root = Path(__file__).resolve().parent
results = {}
for directory in root.iterdir():
    path = directory / "performance.json"
    if not path.is_file(): continue
    data = json.loads(path.read_text())
    cases = data["modes"].get("default", {}).get("cases", [])
    stats = []
    log = directory / "default.log"
    for line in log.read_text().splitlines() if log.exists() else []:
        if "Qwen38ExpertCacheStats[" in line:
            try: stats.append(json.loads(line[line.index("{"):]))
            except (ValueError, json.JSONDecodeError): pass
    short = [c["decode_tokens_per_second"] for c in cases if c["name"].startswith("short_")]
    results[directory.name] = dict(status=data["status"], short_median_tok_s=statistics.median(short) if short else None,
        cases={c["name"]:dict(decode_tok_s=c["decode_tokens_per_second"], after64_tok_s=c["after_64_tokens_per_second"], ttft_s=c["ttft_seconds"], accept_length=c["meta_info"].get("spec_accept_length")) for c in cases},
        latest_stats=stats[-1] if stats else None)
(root / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
for name, result in sorted(results.items()):
    if name == "calibrate": continue
    print(name, result["status"], result["short_median_tok_s"], [(n, round(c["decode_tok_s"], 2)) for n,c in result["cases"].items() if n in ("natural_chat","post_resize_short")])
    stats=result["latest_stats"]
    if stats: print("  calls", stats["run_invocations"], "decode_copy_GiB", round(stats["decode_device_gate_copy_bytes"]/2**30,2), "CPU", stats.get("layered_cpu_hybrid"))
