"""Enable the three experimental loader flags for the existing model canary."""

import json
import runpy

original = json.dumps


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
    return original(value, *args, **kwargs)


json.dumps = dumps
runpy.run_path("tools/qwen38_cache_tests/full_model_canary.py", run_name="__main__")
