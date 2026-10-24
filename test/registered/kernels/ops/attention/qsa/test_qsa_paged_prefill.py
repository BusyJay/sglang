# SPDX-License-Identifier: Apache-2.0
"""Paged prefill matches historical KV packing without its context-sized copies."""

import unittest

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.qsa.sparse_attn import (
    _qsa_fp8_pair_to_bf16,
    can_use_qsa_paged_prefill,
    pack_qsa_prefill_kv,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@triton.jit
def _convert_pairs(words, output, BLOCK: tl.constexpr):
    offset = tl.arange(0, BLOCK)
    low, high = _qsa_fp8_pair_to_bf16(tl.load(words + offset))
    tl.store(output + 2 * offset, low)
    tl.store(output + 2 * offset + 1, high)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQsaPagedPrefill(CustomTestCase):
    def require_native_fp8(self):
        q = torch.empty(1, 24, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.empty(1, 2, 256, device="cuda", dtype=torch.float8_e4m3fn)
        if not can_use_qsa_paged_prefill(q, k, k):
            self.skipTest("native FP8 pair conversion requires SM90+ / PTX 7.8+")

    def test_all_fp8_byte_encodings(self):
        self.require_native_fp8()
        raw = torch.arange(256, device="cuda").to(torch.uint8)
        expected = raw.view(torch.float8_e4m3fn).bfloat16()
        actual = torch.empty_like(expected)
        _convert_pairs[(1,)](raw.view(torch.uint16), actual, 128)
        finite = expected.isfinite()
        torch.testing.assert_close(
            actual.view(torch.int16)[finite],
            expected.view(torch.int16)[finite],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual.isnan(), expected.isnan())

    def compare(self, lengths, query_lengths, dtype, seed=173, topk=2048):
        torch.manual_seed(seed)
        total = sum(lengths)
        slots = total + 17
        # Exercise noncontiguous cache storage and unrelated physical order.
        k, v = [
            (torch.randn(slots, 2, 264, device="cuda") * 0.05).to(dtype)[..., :256]
            for _ in range(2)
        ]
        table = torch.zeros(4, max(lengths) + 13, device="cuda", dtype=torch.int32)
        requests = torch.tensor(
            [3, 1][: len(lengths)], device="cuda", dtype=torch.int32
        )
        physical = torch.randperm(slots, device="cuda", dtype=torch.int32)
        start = 0
        for request, length in zip([3, 1], lengths):
            table[request, :length] = physical[start : start + length]
            start += length
        cu_k = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
        cu_q = torch.tensor(
            [0] + list(torch.tensor(query_lengths).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
        kv_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        rows = sum(query_lengths)
        q = (torch.randn(rows, 24, 256, device="cuda") * 0.05).bfloat16()
        visible = torch.cat(
            [
                torch.arange(length - queries + 1, length + 1, device="cuda")
                for length, queries in zip(lengths, query_lengths)
            ]
        )
        indices = (torch.rand(rows, topk, device="cuda") * visible[:, None]).int()
        if topk == 2048:
            indices[:, -7:] = -1
        else:
            indices[::2, -3:] = -1

        def run(paged):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            if paged:
                keys, values = k, v
                extra = dict(req_to_token=table, req_indices=requests)
            else:
                keys, values = pack_qsa_prefill_kv(
                    k=k,
                    v=v,
                    req_to_token=table,
                    req_indices=requests,
                    cu_k=cu_k,
                    total_k=total,
                    max_k=max(lengths),
                    output_dtype=q.dtype,
                )
                extra = {}
            output = sparse_gqa_fwd_interface_triton_ck(
                q, keys, values, indices, cu_q, cu_k, kv_lens, 256**-0.5, **extra
            )
            torch.cuda.synchronize()
            return output, torch.cuda.max_memory_allocated() - before

        run(False)
        run(True)
        expected, old_peak = run(False)
        actual, new_peak = run(True)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if total > 200000:
            self.assertGreater(old_peak - new_peak, 400 << 20)
            print(
                "QSA_PREFILL_PEAK",
                dict(packed=old_peak, paged=new_peak, saved=old_peak - new_peak),
                flush=True,
            )

    def test_bf16_219k_context(self):
        self.compare([219007], [257], torch.bfloat16)

    def test_bf16_varlen_cache(self):
        self.compare([1031, 517], [17, 9], torch.bfloat16)

    def test_fp8_varlen_cache(self):
        self.require_native_fp8()
        for seed in (173, 174, 175, 176):
            with self.subTest(seed=seed):
                self.compare([1031, 517], [17, 9], torch.float8_e4m3fn, seed)

    def test_fp8_production_chunk_219k_context(self):
        self.require_native_fp8()
        self.compare([219007], [8192], torch.float8_e4m3fn, topk=2051)


if __name__ == "__main__":
    unittest.main()
