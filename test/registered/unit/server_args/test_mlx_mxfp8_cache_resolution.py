"""MXFP8 selection follows the actual runtime, preserving the CUDA guard."""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.kv_cache_hook import handle_mxfp8_kv_cache_compatibility
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMlxMXFP8CacheResolution(CustomTestCase):
    def test_default_precision_stays_unquantized(self):
        from sglang.srt.runtime_context import (
            get_context,
            get_device,
            get_model,
        )

        with get_context().override_server_args(model_path="dummy", device="mps"):
            self.assertEqual(get_model().kv_cache_dtype, "auto")
            self.assertEqual(get_device().mlx_prefill_activation_dtype, "bf16")

    def test_mxfp8_requires_either_mlx_or_blackwell(self):
        # Stub only the optional MLX installation/runtime predicate. Config and
        # platform selection remain real, so accepting mps alone fails here.
        for device, mlx, blackwell in (
            ("mps", True, False),
            ("mps", False, False),
            ("cpu", True, False),
            ("cuda", False, False),
            ("cuda", False, True),
        ):
            with self.subTest(device=device, mlx=mlx, blackwell=blackwell):
                args = ServerArgs(
                    model_path="dummy", device=device, kv_cache_dtype="mxfp8"
                )
                with (
                    override_platform(is_blackwell=blackwell),
                    patch(
                        "sglang.srt.arg_groups.kv_cache_hook.use_mlx", return_value=mlx
                    ),
                ):
                    if blackwell or device == "mps" and mlx:
                        handle_mxfp8_kv_cache_compatibility(args)
                    else:
                        with self.assertRaisesRegex(ValueError, "SM100"):
                            handle_mxfp8_kv_cache_compatibility(args)
                self.assertEqual(args.kv_cache_dtype, "mxfp8")


if __name__ == "__main__":
    unittest.main()
