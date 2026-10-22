# SPDX-License-Identifier: Apache-2.0
"""A timed-out native service keeps its mapped resources for retry."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


def load_close(relative, class_name):
    source = Path(__file__).resolve().parents[5] / relative
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "close")
    namespace = {"envs": SimpleNamespace(
        SGLANG_QWEN38_CPU_READY_TIMEOUT=SimpleNamespace(get=lambda: 0))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["close"]


class TestLayeredClose(CustomTestCase):
    def test_timeout_then_retry(self):
        close = load_close("python/sglang/srt/layers/moe/qwen38_layered_hybrid.py",
                           "LayeredCpuHybrid")
        thread = Mock()
        thread.is_alive.side_effect = [True, False]
        cpu = Mock()
        page = Mock()
        subject = SimpleNamespace(closed=False, thread=thread, cpu=cpu,
                                  page=page, stop=SimpleNamespace(data_ptr=lambda: 123),
                                  runtime=SimpleNamespace(device="cuda"))
        cpu.library.qwen38_cpu_layered_stop = Mock()
        with self.assertRaises(TimeoutError):
            close(subject)
        self.assertFalse(subject.closed)
        page.close.assert_not_called()
        cpu.close.assert_not_called()
        close(subject)
        self.assertTrue(subject.closed)
        page.close.assert_called_once_with("cuda")
        cpu.release_worker.assert_called_once()
        cpu.close.assert_called_once()
        close(subject)
        cpu.close.assert_called_once()

    def test_runtime_keeps_mailbox_until_layered_service_stops(self):
        close = load_close("python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
                           "Qwen38ThreadedExpertRuntime")
        layered = Mock()
        layered.close.side_effect = [TimeoutError("still running"), None]
        mailbox = Mock()
        subject = SimpleNamespace(
            closed=False, synchronize=Mock(), hybrid=None,
            layered_hybrid=layered, miss_mailbox=mailbox, worker=None,
            adaptive_worker=None, adaptive_process=None,
        )
        with self.assertRaises(TimeoutError):
            close(subject)
        self.assertFalse(subject.closed)
        mailbox.close.assert_not_called()
        close(subject)
        self.assertTrue(subject.closed)
        subject.synchronize.assert_called_once()
        mailbox.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
