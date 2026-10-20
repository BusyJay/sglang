# SPDX-License-Identifier: Apache-2.0
"""Build the CPU expert library once at startup, never during an import."""

import fcntl
import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
FLAGS = (
    "-O3",
    "-std=c++17",
    "-mavx512f",
    "-mavx512bw",
    "-mavx512vl",
    "-mavx512bf16",
    "-fPIC",
    "-shared",
    "-pthread",
)


def resolve_library(library, core):
    if library:
        return Path(library).expanduser().resolve()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(
            "CPU hybrid requires a C++ compiler or SGLANG_QWEN38_CPU_LIBRARY"
        )
    version = subprocess.check_output([*compiler, "--version"], timeout=10)
    source = Path(__file__).resolve().parent
    digest = hashlib.sha256(version + repr((compiler, FLAGS)).encode())
    for name in ("kernel.cpp", "base.hpp", "rows.hpp"):
        digest.update((source / name).read_bytes())
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = root / "sglang" / "cpu_nvfp4" / digest.hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "kernel.so"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.is_file():
            logger.info("Building Qwen38 CPU expert library on CPU %d", core)
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                candidate = Path(temporary) / "kernel.so"
                subprocess.run(
                    [
                        "nice",
                        "-n",
                        "15",
                        "taskset",
                        "-c",
                        str(core),
                        *compiler,
                        *FLAGS,
                        str(source / "kernel.cpp"),
                        "-o",
                        str(candidate),
                    ],
                    check=True,
                    timeout=120,
                )
                candidate.replace(output)
    return output
