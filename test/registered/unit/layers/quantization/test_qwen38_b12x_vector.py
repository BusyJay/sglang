# SPDX-License-Identifier: Apache-2.0
"""Guard native load fusion against upstream address changes."""

import unittest

from sglang.srt.layers.quantization.qwen38_b12x_vector import vector_method
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")

SOURCE = """def read(gate_byte_addr, up_byte_addr):
    result = []
    for seg in range(5):
        seg_byte_off = seg * 8
        gw0 = ld_global_nc_u32(gate_byte_addr + seg_byte_off)
        gw1 = ld_global_nc_u32(gate_byte_addr + seg_byte_off + Int64(4))
        uw0 = ld_global_nc_u32(up_byte_addr + seg_byte_off)
        uw1 = ld_global_nc_u32(up_byte_addr + seg_byte_off + Int64(4))
        result.append((gw0, gw1, uw0, uw1))
    return result
"""


class TestNativeVectorTransform(CustomTestCase):
    def test_ordered_word_consumption(self):
        for bits in (64, 128):
            ns = dict(
                Int64=int,
                _qwen_load64=lambda a: (a, a + 4),
                _qwen_load40=lambda a: tuple(range(a, a + 40, 4)),
            )
            exec(compile(vector_method(SOURCE, bits), "generated", "exec"), ns)
            for offset in (0, 40, 1200, 1240):
                self.assertEqual(
                    ns["read"](offset, offset + 1280),
                    [
                        (
                            offset + i,
                            offset + i + 4,
                            offset + 1280 + i,
                            offset + 1280 + i + 4,
                        )
                        for i in range(0, 40, 8)
                    ],
                )

    def test_changed_address_rejected(self):
        for bits in (64, 128):
            with self.assertRaisesRegex(RuntimeError, "adjacent"):
                vector_method(SOURCE.replace("Int64(4)", "Int64(8)"), bits)


if __name__ == "__main__":
    unittest.main()
