# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(
    __import__("sys").platform == "darwin", "Requires MLX Metal device"
)
class TestMlxCSF(CustomTestCase):
    def test_page_relative_exception_and_tail_row_tile(self):
        import mlx.core as mx

        from sglang.kernels.ops.quantization.mlx_csf import mlx_csf32_restore

        r, c = 160, 128
        blocks = r // 32 * c // 2
        tiles = ((r + 127) // 128) * (c // 4)
        pages = (tiles + 31) // 32
        offsets = mx.zeros((blocks + pages,), mx.uint16)
        # First block's first scale is overridden; remainder of page has one record.
        offsets[1:257] = 1
        origins = mx.array([0, 2], mx.uint32)
        out = mlx_csf32_restore(
            mx.full((r,), 20, mx.uint8),
            mx.full((r * c // 2,), 0x32, mx.uint8),
            origins,
            offsets,
            mx.array([99 << 8], mx.uint16),
            rows=r,
            columns=c,
        )
        self.assertEqual(out[0, 0].item(), 99)
        self.assertEqual(out[-1, -2].item(), 22)
        self.assertEqual(out[-1, -1].item(), 23)


if __name__ == "__main__":
    unittest.main()
