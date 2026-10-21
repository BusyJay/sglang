# SPDX-License-Identifier: Apache-2.0
"""Guard the source-derived B12X hybrid epilogue and direct-launch ABI."""

import ast
import inspect
import unittest

from sglang.srt.layers.quantization import qwen38_b12x_hybrid as subject
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestB12xHybrid(CustomTestCase):
    def test_installed_b12x_fc2_writers_are_guarded(self):
        from b12x.moe._shared.kernels import micro
        from b12x.moe._shared.kernels.w4a16 import kernel

        for name in subject._WRITERS:
            tree, count = subject.hybrid_method(
                inspect.getsource(getattr(micro.MoEMicroKernelBackend, name)), name
            )
            self.assertGreater(count, 0, name)
            method = tree.body[0]
            self.assertEqual(
                [argument.arg for argument in method.args.args[-5:]],
                list(subject._ARGUMENTS),
            )
            additions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_hybrid_add"
            ]
            self.assertEqual(len(additions), count)
        for name in subject._CALLERS:
            subject.hybrid_method(
                inspect.getsource(getattr(micro.MoEMicroKernelBackend, name)), name
            )
        tree = subject.hybrid_compile_function(
            inspect.getsource(kernel._compile_w4a16_small_m_direct)
        )
        direct = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "b12x_compile"
        )
        self.assertEqual(
            [node.args[0].attr for node in direct.args[-6:-2]],
            ["Int32", "Int32", "Float32", "Int32"],
        )

        self.assertEqual(direct.args[-2].func.attr, "Int32")
        self.assertEqual(direct.args[-2].args[0].value, 0)

    def test_unknown_writer_rejected(self):
        with self.assertRaises(RuntimeError):
            subject.hybrid_method(
                "def _m1_fc2_rowpair_narrow(self, scatter_output):\n"
                "    scatter_output[0] = value\n",
                "_m1_fc2_rowpair_narrow",
            )


if __name__ == "__main__":
    unittest.main()
