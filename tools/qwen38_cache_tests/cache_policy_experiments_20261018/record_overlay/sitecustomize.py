"""Recording-only worker overlay; activated by explicit PYTHONPATH."""

import os
import traceback

try:
    from tools.qwen38_cache_tests.cache_route_record import install

    install()
except Exception:
    traceback.print_exc()
    # Python otherwise ignores sitecustomize failures and serves without tape.
    os._exit(1)
