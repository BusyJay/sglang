# SPDX-License-Identifier: Apache-2.0
"""Loader boundary for the opt-in cache overlap options."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_loader.qwen38_nvfp4_cache_loader import (
    Qwen38NvFp4CacheModelLoader,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


def build(**options):
    config = {"manifest_path": "/tmp/qwen38-manifest.json"}
    config.update(options)
    manifest = SimpleNamespace(
        num_layers=2,
        num_experts=64,
        validate_cutlass_layout=lambda: None,
        assert_runtime_compatible=lambda **kwargs: None,
    )
    with mock.patch(
        "sglang.srt.model_loader.qwen38_nvfp4_cache_loader.load_manifest",
        return_value=manifest,
    ):
        return Qwen38NvFp4CacheModelLoader(
            SimpleNamespace(model_loader_extra_config=config)
        )


class TestCacheLoaderConfig(CustomTestCase):
    def test_defaults_keep_every_overlap_option_off(self):
        build()  # no new option means the historical single-dispatch path

    def test_threaded_combo_is_accepted(self):
        build(
            runtime_mode="threaded",
            prefill_reuse_cache=True,
            prefill_direct_cache=True,
            decode_early_cache=True,
            preserve_cache_on_resize=True,
            cache_overlap_policy="auto",
        )

    def test_unknown_policy_and_missing_threaded_mode_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "auto or always"):
            build(runtime_mode="threaded", cache_overlap_policy="sometimes")
        with self.assertRaisesRegex(ValueError, "threaded"):
            build(cache_overlap_policy="always")

    def test_direct_cache_requires_reuse_and_threaded(self):
        with self.assertRaisesRegex(ValueError, "threaded"):
            build(prefill_direct_cache=True)
        with self.assertRaisesRegex(ValueError, "prefill_reuse_cache"):
            build(
                runtime_mode="threaded",
                prefill_reuse_cache=False,
                prefill_direct_cache=True,
            )

    def test_non_boolean_overlap_options_are_rejected(self):
        for option, value in (
            ("prefill_direct_cache", "true"),
            ("decode_early_cache", 1),
            ("preserve_cache_on_resize", None),
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "JSON boolean"):
                    build(runtime_mode="threaded", **{option: value})

    def test_decode_and_resize_options_require_threaded(self):
        for option in ("decode_early_cache", "preserve_cache_on_resize"):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "threaded"):
                    build(**{option: True})


if __name__ == "__main__":
    unittest.main()
