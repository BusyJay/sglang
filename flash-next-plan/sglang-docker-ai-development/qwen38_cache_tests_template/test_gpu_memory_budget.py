import unittest

import torch

from .gpu_test_utils import (
    assert_peak_within,
    require_cuda_free,
)


class TestGpuMemoryBudget(unittest.TestCase):
    REQUIRED_FREE_GIB = 0.5
    EXPECTED_PEAK_GIB = 0.25

    def test_small_allocation_stays_in_budget(self):
        require_cuda_free(self.REQUIRED_FREE_GIB)
        tensor = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device="cuda")
        tensor.zero_()
        torch.cuda.synchronize()
        assert_peak_within(self, self.EXPECTED_PEAK_GIB)


if __name__ == "__main__":
    unittest.main()
