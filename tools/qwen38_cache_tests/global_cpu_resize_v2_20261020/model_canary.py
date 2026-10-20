"""Run the full-model canary as a long->short KV tail re-loan scenario.

Two things are needed to reach the re-loan in a live server and nothing else is
changed here:

* loader flags for the staged KV loan (``QWEN38_CANARY_FLAGS``, a JSON object);
* fixtures that the radix tree cannot share. The stock canary repeats one
  phrase for every ``--contexts`` entry, so a second 170K request is a pure
  prefix hit, the tree never evicts, and the physical KV tail the expert pool
  loaned out stays pinned by the first long request. Each context payload is
  therefore prefixed with a distinct salt run, which makes eviction pressure
  real and lets the working set actually drop two stages down.

Semantic probes keep their real prompts: only payloads that generate the
fixture's ``--context-max-new-tokens`` are salted.
"""

import json
import os
import runpy
import urllib.request
from itertools import count

_loads = json.loads
_dumps = json.dumps

_FLAGS = _loads(os.environ.get("QWEN38_CANARY_FLAGS", "{}"))
_SALT_MIN_TOKENS = int(os.environ.get("QWEN38_CANARY_SALT_MIN_TOKENS", "1024"))
_SALT_MAX_NEW_TOKENS = int(os.environ.get("QWEN38_CANARY_SALT_MAX_NEW_TOKENS", "8"))
_SALT_LENGTH = int(os.environ.get("QWEN38_CANARY_SALT_LENGTH", "64"))
_salt_ids = count(1)


def _dump(value, *args, **kwargs):
    if isinstance(value, dict) and "manifest_path" in value and "runtime_mode" in value:
        value = dict(value, **_FLAGS)
    return _dumps(value, *args, **kwargs)


def _salt_run(index, length):
    # Any id inside the embedding table works; the fixture text carries the
    # semantics, this only breaks prefix sharing between fixture requests.
    base = (index * 2654435761) % 100_000
    return [1000 + (base + step * 40503) % 100_000 for step in range(length)]


def _urlopen(request, *args, **kwargs):
    body = getattr(request, "data", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            payload = _loads(body)
        except ValueError:
            payload = None
        ids = payload.get("input_ids") if isinstance(payload, dict) else None
        sampling = payload.get("sampling_params") or {}
        if (
            isinstance(ids, list)
            and len(ids) >= _SALT_MIN_TOKENS
            and sampling.get("max_new_tokens") == _SALT_MAX_NEW_TOKENS
        ):
            payload["input_ids"] = _salt_run(next(_salt_ids), _SALT_LENGTH) + ids
            request = urllib.request.Request(
                request.full_url,
                _dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
    return _original_urlopen(request, *args, **kwargs)


json.dumps = _dump
_original_urlopen = urllib.request.urlopen
urllib.request.urlopen = _urlopen

print(f"canary loader flags: {_FLAGS}", flush=True)
print(
    f"salting fixtures >= {_SALT_MIN_TOKENS} tokens with {_SALT_LENGTH} distinct ids",
    flush=True,
)
runpy.run_path("tools/qwen38_cache_tests/full_model_canary.py", run_name="__main__")
