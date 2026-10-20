# SPDX-License-Identifier: Apache-2.0
"""Guarded upstream source conversion must preserve BF16 hidden activation."""

import ast
import unittest

from sglang.srt.layers.quantization import qwen38_b12x_fp32 as subject
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestB12xFp32(CustomTestCase):
    def test_only_final_scatter_cast_changes(self):
        source = """def projection(scatter_output, intermediate, acc):
    intermediate[0] = BFloat16(acc)
    scatter_output[0] = BFloat16(acc)
    scatter_output[1] = BFloat16(acc + 1)
"""
        tree, count = subject.fp32_scatter_method(source)
        self.assertEqual(count, 2)
        namespace = {"BFloat16": lambda v: round(v), "Float32": float}
        exec(compile(tree, "test", "exec"), namespace)
        out, hidden = [0, 0], [0]
        namespace["projection"](out, hidden, 1.125)
        self.assertEqual(hidden, [1])
        self.assertEqual(out, [1.125, 2.125])

    def test_unknown_scatter_epilogue_rejected(self):
        with self.assertRaises(RuntimeError):
            subject.fp32_scatter_method(
                "def p(scatter_output):\n    scatter_output[0] = acc\n"
            )

    def test_compile_changes_only_output_pointer(self):
        args = ["kernel"] + ["dummy(cutlass.Uint8)"] * 12 + ["dummy(cutlass.BFloat16)"]
        source = (
            "def compile_direct():\n    return b12x_compile(" + ",".join(args) + ")\n"
        )
        tree = subject.fp32_compile_function(source)
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "b12x_compile"
        )
        self.assertEqual(call.args[13].args[0].attr, "Float32")
        self.assertTrue(all(arg.args[0].attr == "Uint8" for arg in call.args[1:13]))

    def test_unknown_compile_abi_rejected(self):
        with self.assertRaises(RuntimeError):
            subject.fp32_compile_function(
                "def c():\n    return b12x_compile(kernel, output)\n"
            )


if __name__ == "__main__":
    unittest.main()
