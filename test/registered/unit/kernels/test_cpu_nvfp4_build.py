# SPDX-License-Identifier: Apache-2.0
"""CPU source changes invalidate the native cache; failed builds cannot publish."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.kernels.ops.moe.cpu_nvfp4 import build
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestCpuNvfp4Build(CustomTestCase):
    def test_source_invalidation_and_failure_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name in ("kernel.cpp", "base.hpp", "rows.hpp"):
                (source / name).write_text(name)

            def compile_library(command, **kwargs):
                Path(command[-1]).write_bytes(b"compiled library")

            with (
                patch.object(build, "__file__", str(source / "build.py")),
                patch.dict(os.environ, CXX="c++", XDG_CACHE_HOME=str(root)),
                patch.object(build.shutil, "which", return_value="/compiler/c++"),
                patch.object(
                    build.subprocess, "check_output", return_value=b"compiler v1"
                ),
                patch.object(
                    build.subprocess, "run", side_effect=compile_library
                ) as compiler,
            ):
                first = build.resolve_library("", 0)
                self.assertEqual(first.read_bytes(), b"compiled library")
                self.assertEqual(build.resolve_library("", 0), first)
                self.assertEqual(compiler.call_count, 1)
                (source / "rows.hpp").write_text("new decode math")
                compiler.side_effect = subprocess.CalledProcessError(1, "compiler")
                with self.assertRaises(subprocess.CalledProcessError):
                    build.resolve_library("", 0)
                self.assertEqual(
                    list(root.glob("sglang/cpu_nvfp4/*/kernel.so")), [first]
                )
                compiler.side_effect = compile_library
                second = build.resolve_library("", 0)
                self.assertNotEqual(first, second)
                self.assertEqual(second.read_bytes(), b"compiled library")
                compiler.side_effect = AssertionError(
                    "explicit library must not compile"
                )
                self.assertEqual(build.resolve_library(str(first), 0), first)


if __name__ == "__main__":
    unittest.main()
