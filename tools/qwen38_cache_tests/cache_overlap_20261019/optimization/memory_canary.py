"""Replay identical requests with allocator history saved at request boundaries."""

import json
import runpy
import urllib.request

original_dumps = json.dumps


def dumps(value, *args, **kwargs):
    if isinstance(value, dict) and "manifest_path" in value and "runtime_mode" in value:
        value = dict(
            value,
            prefill_direct_cache=True,
            decode_early_cache=True,
            preserve_cache_on_resize=True,
            kv_cache_stages=[80000, 160000, 220000],
            preallocate_gdn_proj=True,
            async_pack_prefetch=True,
        )
    return original_dumps(value, *args, **kwargs)


json.dumps = dumps
ns = runpy.run_path("tools/qwen38_cache_tests/full_model_canary.py", run_name="fixture")
main = ns["main"]
request = ns["request"]


def control(url, data):
    req = urllib.request.Request(
        url, json.dumps(data).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode()


def observed(url, data=None, timeout=30):
    if url.endswith("/generate") and data and len(data.get("input_ids", [])) >= 1024:
        length = len(data["input_ids"])
        base = url[: -len("/generate")]
        control(
            base + "/start_profile",
            {
                "activities": ["MEM"],
                "output_dir": f"/results/optimization/memory-{length}",
            },
        )
        try:
            return request(url, data, timeout)
        finally:
            control(base + "/stop_profile", {})
    return request(url, data, timeout)


main.__globals__["request"] = observed
raise SystemExit(main())
