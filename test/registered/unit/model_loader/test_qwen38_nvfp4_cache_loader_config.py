# SPDX-License-Identifier: Apache-2.0
"""Loader defaults and validation for CPU hybrid, overlap and resize retention."""

import ast
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_loader.qwen38_nvfp4_cache_loader import (
    Qwen38NvFp4CacheModelLoader,
    default_cpu_hybrid,
    default_prefill_direct_cache,
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
    def test_cpu_hybrid_defaults_to_gpu_without_enable(self):
        from sglang.srt.environ import envs

        config = dict(runtime_mode="threaded", adaptive_cache=True)
        methods = [SimpleNamespace(use_w4a16=True)]
        with envs.SGLANG_ENABLE_QWEN38_CPU_HYBRID.override(False):
            envs.SGLANG_ENABLE_QWEN38_CPU_HYBRID.clear()
            for calibration in (None, "{}"):
                with (
                    self.subTest(calibration=calibration),
                    envs.SGLANG_QWEN38_CPU_COST_CALIBRATION.override(calibration),
                ):
                    self.assertFalse(default_cpu_hybrid(config, methods))

    def test_cpu_hybrid_enabled_without_calibration_stays_on_gpu(self):
        from sglang.srt.environ import envs

        config = dict(runtime_mode="threaded", adaptive_cache=True)
        methods = [SimpleNamespace(use_w4a16=True)]
        with (
            envs.SGLANG_ENABLE_QWEN38_CPU_HYBRID.override(True),
            envs.SGLANG_QWEN38_CPU_COST_CALIBRATION.override(None),
        ):
            self.assertFalse(default_cpu_hybrid(config, methods))

    def test_cpu_hybrid_requires_enable_calibration_and_supported_runtime(self):
        from sglang.srt.environ import envs

        config = dict(runtime_mode="threaded", adaptive_cache=True)
        methods = [SimpleNamespace(use_w4a16=True)]
        with (
            envs.SGLANG_ENABLE_QWEN38_CPU_HYBRID.override(True),
            envs.SGLANG_QWEN38_CPU_COST_CALIBRATION.override("{}"),
        ):
            self.assertTrue(default_cpu_hybrid(config, methods))
            for change in (
                dict(runtime_mode="sync"),
                dict(adaptive_cache=False),
                dict(cache_policy="slru"),
                dict(prefill_arena=False),
                dict(adaptive_cache_mode="observe"),
            ):
                with self.subTest(change=change):
                    self.assertFalse(
                        default_cpu_hybrid(dict(config, **change), methods)
                    )
            self.assertFalse(default_cpu_hybrid(config, []))
            self.assertFalse(
                default_cpu_hybrid(config, [SimpleNamespace(use_w4a16=False)])
            )
        with (
            envs.SGLANG_ENABLE_QWEN38_CPU_HYBRID.override(False),
            envs.SGLANG_QWEN38_CPU_COST_CALIBRATION.override("{}"),
        ):
            self.assertFalse(default_cpu_hybrid(config, methods))

    def test_resize_retention_defaults_on_and_accepts_explicit_false(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        build()
        build(runtime_mode="threaded")
        build(runtime_mode="threaded", preserve_cache_on_resize=False)
        self.assertIs(
            inspect.signature(Qwen38ThreadedExpertRuntime)
            .parameters["preserve_cache_on_resize"]
            .default,
            True,
        )
        # The loader passes an explicit default, overriding the constructor's.
        # Check both entry points so serving cannot silently disable retention.
        tree = ast.parse(inspect.getsource(Qwen38NvFp4CacheModelLoader))
        defaults = [
            node.elts[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and node.elts[0].value == "preserve_cache_on_resize"
            and isinstance(node.elts[1], ast.Constant)
        ]
        self.assertEqual(defaults, [True])

    def test_direct_cache_follows_its_two_preconditions(self):
        # Sync runtime cannot use it; threaded without cache reuse has no
        # cache rows to read. Only threaded+reuse defaults it on.
        self.assertFalse(default_prefill_direct_cache({}))
        self.assertFalse(default_prefill_direct_cache({"runtime_mode": "device"}))
        self.assertTrue(default_prefill_direct_cache({"runtime_mode": "threaded"}))
        self.assertFalse(
            default_prefill_direct_cache(
                {"runtime_mode": "threaded", "prefill_reuse_cache": False}
            )
        )

    def test_explicit_false_still_disables_the_default(self):
        build(runtime_mode="threaded", prefill_direct_cache=False)

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
