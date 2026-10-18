import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True


def test_split_full_attention_applies_model_wrapper_once():
    # The hybrid backend takes the speculative attention mode from the
    # published configuration.
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(speculative_attention_mode="prefill")
    override.install()
    try:
        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode="prefill"),
            model_config=SimpleNamespace(context_len=2048),
            kv_cache_dtype=None,
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            kv_index_translator=None,
            init_new_workspace=None,
        )
        wrapper_inputs = []
        wrapped_backend = object()

        def wrap_once(model_runner, backend):
            assert model_runner is runner
            wrapper_inputs.append(backend)
            return wrapped_backend

        constructors = {
            "decode-test": lambda model_runner: _FakeBackend("decode"),
            "prefill-test": lambda model_runner: _FakeBackend("prefill"),
        }
        resolved = ResolvedAttentionBackendStr(
            decode="decode-test", prefill="prefill-test"
        )

        with (
            patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
            patch.object(
                attention_backend_setup,
                "attn_backend_wrapper",
                side_effect=wrap_once,
            ),
        ):
            result = attention_backend_setup._build_resolved_backend(
                model_runner=runner,
                resolved=resolved,
                init_new_workspace=True,
            )

        assert result is wrapped_backend
        assert len(wrapper_inputs) == 1
        split_backend = wrapper_inputs[0]
        assert isinstance(split_backend, HybridAttnBackend)
        assert split_backend.decode_backend.name == "decode"
        assert split_backend.prefill_backend.name == "prefill"
        assert runner.init_new_workspace is True
    finally:
        override.restore()


def test_equal_resolved_backends_ignore_stale_global_backend():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            attention_backend="global-test",
            speculative_attention_mode="prefill",
        ),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    constructors = {
        "global-test": lambda _runner: _FakeBackend("global"),
        "resolved-test": lambda _runner: _FakeBackend("resolved"),
    }
    resolved = ResolvedAttentionBackendStr(
        decode="resolved-test",
        prefill="resolved-test",
    )

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=lambda _runner, backend: backend,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=False,
        )

    assert result.name == "resolved"


class TestQsaConstruction(CustomTestCase):
    def test_sparse_hybrid_does_not_leave_discarded_dense_workspace(self):
        import torch

        from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig

        config = Qwen4ExpConfig(text_config=dict(
            indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128,
            indexer_budget=2048, indexer_compress_ratio=4))
        runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=config))
        persistent = {}

        def dense_backend(_runner):
            persistent["unused_workspace"] = torch.empty(256)
            return _FakeBackend("dense")

        def sparse_wrapper(_runner, dense):
            return SimpleNamespace(name="qsa", discarded_dense=dense)

        for resolved in (
            ResolvedAttentionBackendStr(prefill="flashinfer", decode="flashinfer"),
            ResolvedAttentionBackendStr(prefill="flashinfer", decode="triton"),
        ):
            with (self.subTest(resolved=resolved),
                  patch.dict(attention_backend_setup.ATTENTION_BACKENDS,
                             {"flashinfer": dense_backend, "triton": dense_backend}),
                  patch.object(attention_backend_setup, "attn_backend_wrapper",
                               side_effect=sparse_wrapper)):
                result = attention_backend_setup._build_resolved_backend(
                    model_runner=runner, resolved=resolved, init_new_workspace=False)
                self.assertEqual(result.name, "qsa")
                self.assertIsNone(result.discarded_dense)
                self.assertEqual(persistent, {})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
