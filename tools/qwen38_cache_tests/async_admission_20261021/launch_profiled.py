"""Install the same probes in the launcher and multiprocessing spawn children."""

import runpy

from profile_hooks import install

install()

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
