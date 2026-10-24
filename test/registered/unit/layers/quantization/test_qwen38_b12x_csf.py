# SPDX-License-Identifier: Apache-2.0
"""Native address substitution reads the same scale words without touching W4."""

import struct
import unittest

import numpy as np

from sglang.srt.layers.quantization.qwen38_b12x_csf import csf_method
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import decode, encode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestB12xCsf(CustomTestCase):
    def test_word_addresses_and_weight_loads(self):
        rows, cols = 128, 16
        raw = (
            np.random.default_rng(53)
            .integers(0, 256, rows * cols, dtype=np.uint8)
            .tobytes()
        )
        slab = encode(raw, rows, cols)
        spec = dict(rows=rows, columns=cols, stride=(len(slab) + 15) // 16 * 16)
        source = """def read(w1s_base_addr, w_base, eid, position):
    address = w1s_base_addr + eid * 2048 + position
    scale = ld_global_nc_u32(address)
    weight = ld_global_nc_u32(w_base + position)
    prefetch_global_l2(address)
    return scale, weight
"""
        decoded = decode(slab, rows, cols)
        observed = []

        def word(base, offset, r, c, stride):
            self.assertEqual((base, r, c, stride), (10000, rows, cols, spec["stride"]))
            self.assertEqual(offset // (r * c), 2)
            return struct.unpack_from("<I", decoded, offset % (r * c))[0]

        def word_at(base, expert, offset, r, c, stride):
            self.assertEqual(expert, 2)
            self.assertGreaterEqual(offset, 0)
            self.assertLess(offset, r * c)
            return word(base, expert * r * c + offset, r, c, stride)

        for direct in (False, True):
            tree, count = csf_method(
                source, {"w13_blockscale_swizzled": spec}, direct_expert=direct
            )
            self.assertEqual(count, 1)
            namespace = dict(
                _csf_word=word,
                _csf_word_at=word_at,
                Int64=int,
                ld_global_nc_u32=lambda addr: observed.append(addr) or 77,
            )
            exec(compile(tree, "generated", "exec"), namespace)
            for pos in (0, 4, 508, 512, 1020, 2044):
                value, weight = namespace["read"](10000, 50000, 2, pos)
                self.assertEqual(value, struct.unpack_from("<I", raw, pos)[0])
                self.assertEqual(weight, 77)
                self.assertEqual(observed[-1], 50000 + pos)

    def test_unknown_scale_load_abi_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "scale load"):
            csf_method(
                "def read(w1s_base_addr):\n    return ld_global_nc_u16(w1s_base_addr)\n",
                {"w13_blockscale_swizzled": dict(rows=128, columns=16, stride=1536)},
            )


if __name__ == "__main__":
    unittest.main()
