"""Changing the base cache must retain VMM addresses and old base bytes."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.qwen38_auto_capacity import forecast_loan_budgets
from sglang.srt.layers.moe.qwen38_staged_pool import StagedExpertPool
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestAutoCapacityCuda(CustomTestCase):
    def test_rebase_retains_base_data_and_staged_returns(self):
        manifest = SimpleNamespace(
            num_experts=128,
            tensors={
                0: {
                    name: SimpleNamespace(shape=(128, 32768), dtype="uint8")
                    for name in REQUIRED_TENSORS
                }
            },
        )
        pool = StagedExpertPool(manifest, torch.device("cuda"), [10, 10], 10)
        try:
            pool.configure_loans([24 << 20, 12 << 20, 0])
            pointers = {
                name: tensor.data_ptr() for name, tensor in pool.tensors.items()
            }
            for tensor in pool.tensors.values():
                tensor[:20].fill_(37)
            expected = forecast_loan_budgets(
                100, 256, pool.loan_bytes, pool.physical_bytes
            )
            pool.rebase_before_execution(100)
            torch.cuda.synchronize()
            self.assertEqual(pool.budgets, expected)
            for name, tensor in pool.tensors.items():
                self.assertEqual(tensor.data_ptr(), pointers[name])
                self.assertTrue(torch.all(tensor[:20] == 37).item())
            pool.release_stage(2)
            for tensor in pool.tensors.values():
                self.assertTrue(torch.all(tensor[:20] == 37).item())
            pool.grow_stage(0)
            self.assertEqual(pool.stage_index, 0)
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
