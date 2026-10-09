from __future__ import annotations

import os
import unittest

import torch


GIB = 1 << 30


def configured_budget_gib() -> float:
    return float(os.environ.get("QWEN38_TEST_GPU_BUDGET_GIB", "8"))


def require_cuda_free(required_free_gib: float) -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable")
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < required_free_gib * GIB:
        raise unittest.SkipTest(
            f"Need {required_free_gib:.2f} GiB free, "
            f"have {free_bytes / GIB:.2f} GiB"
        )
    fraction = min(0.95, configured_budget_gib() / (total_bytes / GIB))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.reset_peak_memory_stats(0)
    return free_bytes, total_bytes


def assert_peak_within(testcase: unittest.TestCase, expected_peak_gib: float) -> None:
    allocated = torch.cuda.max_memory_allocated(0) / GIB
    reserved = torch.cuda.max_memory_reserved(0) / GIB
    budget = configured_budget_gib()
    testcase.assertLessEqual(
        reserved,
        budget,
        f"Reserved peak {reserved:.2f} GiB exceeds configured budget {budget:.2f} GiB",
    )
    testcase.assertLessEqual(
        allocated,
        expected_peak_gib,
        f"Allocated peak {allocated:.2f} GiB exceeds expected {expected_peak_gib:.2f} GiB",
    )

