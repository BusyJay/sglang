# SPDX-License-Identifier: Apache-2.0
"""Single-request target/draft scratch sharing preserves concurrency isolation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as module
from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestQsaWorkspace(CustomTestCase):
    def test_single_request_shares_only_within_device(self):
        buffers = {}

        def get_buffer(name, allocate):
            if name not in buffers:
                buffers[name] = allocate()
            return buffers[name]

        with (
            patch.object(
                module,
                "get_schedule",
                return_value=SimpleNamespace(max_running_requests=1),
            ),
            patch.object(
                module,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm="EAGLE"),
            ),
            patch.object(module, "get_buffer", side_effect=get_buffer),
            patch.object(
                module.torch, "zeros", side_effect=lambda *a, **kw: object()
            ) as zeros,
        ):
            target, draft = SimpleNamespace(runner=object()), SimpleNamespace(
                runner=object()
            )
            allocate = module.QwenSparseAttnBackend._allocate_trtllm_workspace
            self.assertIs(allocate(target, "cuda:0"), allocate(draft, "cuda:0"))
            self.assertIsNot(allocate(target, "cuda:0"), allocate(draft, "cuda:1"))
            self.assertEqual(zeros.call_count, 2)

    def test_concurrent_or_runner_free_callers_stay_private(self):
        allocate = module.QwenSparseAttnBackend._allocate_trtllm_workspace
        for runner, limit, algorithm in (
            (object(), 2, "EAGLE"),
            (object(), None, None),
            (None, 1, "EAGLE"),
            (object(), 1, "DFLASH"),
        ):
            with (
                patch.object(
                    module,
                    "get_schedule",
                    return_value=SimpleNamespace(max_running_requests=limit),
                ),
                patch.object(
                    module,
                    "get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=algorithm),
                ),
                patch.object(module, "get_buffer") as shared,
                patch.object(
                    module.torch, "zeros", side_effect=lambda *a, **kw: object()
                ),
            ):
                owner = SimpleNamespace(runner=runner)
                self.assertIsNot(allocate(owner, "cuda:0"), allocate(owner, "cuda:0"))
                shared.assert_not_called()


class TestQsaPagedCompatibility(CustomTestCase):
    def test_unsupported_hardware_or_assembler_uses_packing(self):
        device = "cuda:0"
        for capability, assembler, supported in (
            ((8, 0), "13.3", False),
            ((9, 0), "11.7", False),
            ((9, 0), "11.8", True),
            ((12, 0), "13.0", True),
            ((12, 0), "13.3", True),
        ):
            with (
                self.subTest(capability=capability, assembler=assembler),
                patch.object(sparse_attn.torch.version, "hip", None),
                patch.object(
                    sparse_attn.torch.cuda,
                    "get_device_capability",
                    return_value=capability,
                ),
                patch(
                    "triton.backends.nvidia.compiler.get_ptxas",
                    return_value=SimpleNamespace(version=assembler),
                ),
            ):
                sparse_attn._qsa_native_fp8_pair_supported.cache_clear()
                self.assertEqual(
                    sparse_attn._qsa_native_fp8_pair_supported(device), supported
                )
        sparse_attn._qsa_native_fp8_pair_supported.cache_clear()


if __name__ == "__main__":
    unittest.main()
