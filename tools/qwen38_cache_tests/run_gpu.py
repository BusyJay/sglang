#!/usr/bin/env python3
"""Small GPU tests; never launches a server or downloads model weights."""

import argparse
import json
import pathlib
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/workspace/results/gpu-small.json")
    parser.add_argument("--require-sm120", action="store_true")
    args = parser.parse_args()
    import torch

    if args.require_sm120 and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
    ):
        parser.error("SM120 CUDA is required")
    root = pathlib.Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        str(root), pattern="test_gpu_*.py", top_level_dir=str(root.parent.parent)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    skips = [{"test": str(test), "reason": reason} for test, reason in result.skipped]
    passed = result.wasSuccessful() and result.testsRun > len(skips)
    if args.require_sm120 and skips:
        passed = False
    report = {
        "tests": result.testsRun,
        "skipped": len(result.skipped),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "status": ("passed" if passed else "failed_or_all_skipped"),
        "skip_details": skips,
        "failure_details": [
            {"test": str(test), "traceback": detail} for test, detail in result.failures
        ],
        "error_details": [
            {"test": str(test), "traceback": detail} for test, detail in result.errors
        ],
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "weights_downloaded": False,
    }
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
