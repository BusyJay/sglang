# SPDX-License-Identifier: Apache-2.0
"""Profiler boundaries must drain GPU work driven by Python workers."""

import ast
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT


class TestProfilerBoundaries(unittest.TestCase):
    def test_cudart_start_and_stop_follow_worker_completion(self):
        source = (
            REPO_ROOT
            / "python/sglang/srt/managers/scheduler_components/profiler_manager.py"
        )
        tree = ast.parse(source.read_text())
        methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_start_profile", "_stop_profile")
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *methods,
            ],
            type_ignores=[],
        )
        for name, cudart_name in (
            ("_start_profile", "cudaProfilerStart"),
            ("_stop_profile", "cudaProfilerStop"),
        ):
            with (
                self.subTest(boundary=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                allow_worker, completed = threading.Event(), threading.Event()

                def finish_staging():
                    if allow_worker.wait(5):
                        completed.set()

                worker = threading.Thread(target=finish_staging)
                worker.start()

                def synchronize():
                    allow_worker.set()
                    self.assertTrue(
                        completed.wait(1), "Python staging worker cannot progress"
                    )

                def profiler_boundary():
                    self.assertTrue(
                        completed.is_set(),
                        "cudart boundary entered with a pending Python worker",
                    )

                cudart = SimpleNamespace(**{cudart_name: profiler_boundary})
                cuda = SimpleNamespace(synchronize=synchronize, cudart=lambda: cudart)
                namespace = dict(
                    torch=SimpleNamespace(
                        cuda=cuda,
                        profiler=SimpleNamespace(
                            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA")
                        ),
                    ),
                    envs=SimpleNamespace(
                        SGLANG_PROFILE_V2=SimpleNamespace(get=lambda: False)
                    ),
                    current_platform=SimpleNamespace(is_out_of_tree=lambda: False),
                    get_device=lambda: SimpleNamespace(gpu_id=0, base_gpu_id=0),
                    ProfileReqOutput=SimpleNamespace,
                    logger=mock.Mock(),
                )
                exec(
                    compile(ast.fix_missing_locations(module), str(source), "exec"),
                    namespace,
                )
                manager = SimpleNamespace(
                    profile_in_progress=True,
                    profiler_activities=["CUDA_PROFILER"],
                    torch_profiler=None,
                    rpd_profiler=None,
                    torch_profiler_with_stack=None,
                    torch_profiler_record_shapes=None,
                    profile_prefix="",
                    detailed_annotations=False,
                    profile_id="test",
                    torch_profiler_output_dir=Path(directory),
                    _merge_profile_traces=lambda: "",
                    _apply_detailed_annotations=lambda enabled: None,
                )
                try:
                    self.assertTrue(namespace[name](manager).success)
                finally:
                    allow_worker.set()
                    worker.join(2)
