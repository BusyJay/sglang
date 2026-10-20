# Global cache / CPU resize v2 integration review

Follow-up: [default CPU hybrid and serving resize validation](DEFAULT_ON_VALIDATION.md) supersedes the initial opt-in default and text-serving limitations below.

Reviewed 2026-10-20 against local head `a9e846fba0` on
`qwen3.8-flash-next-offload`. Archive:
`qwen38-global-cache-cpu-resize-v2-20261020.tar.gz`, SHA256
`58a6f960e71254e31f8da35e3beb5bd05c0f30e9d805d0f4ee818171b021492a`.
All archive SHA256SUMS entries verified. The full patch was applied with
`git apply --3way` in an isolated worktree; the reborrow-only upgrade did not
apply directly to the newer local head. Existing cache-stat, tail-reloan and
preallocated GDN fixes were retained.

## Review fixes

- CPU and B12X originally computed different functions. The CPU applied global
  scales to each weight before BF16 rounding; installed B12X applies scales to
  the projection results. Real H=2560/I=640 GPU partition tests failed the
  package's tolerance (first run: 174/2560 elements mismatched, max absolute
  difference 136). Keep the existing Triton CPU entry point and add an explicit
  B12X entry point with projection scaling. The serving adapter selects it by
  GPU backend and rejects libraries missing the new symbol. No tolerance was
  relaxed. The corrected B12X tests match final BF16 output exactly for all
  tested hit/miss partitions.
- Pad alpha rows to 16-byte alignment. A 1025-slot cache previously failed the
  native B12X pointer alignment assertion before launch.
- Repair GPU smoke setup: CUDA index_copy does not support float8 tensors;
  copy row-preserving byte views instead. Smoke now executes the production
  CpuNvfp4 adapter, including backend selection and source binding.
- Register new unit tests at module scope so CI's AST collector sees them;
  move them under layers/moe and layers/quantization and use CustomTestCase.
- Update the older fake B12X fixture for the explicit launch attributes; apply
  repository Python/C++ formatting. Add an independent CPU oracle for B12X
  scaling and a real expert VMM release/growth smoke.

## Validation actually rerun

Hardware: Ryzen 9 9950X, RTX 5090 D. GPU/dependency checks used disposable
containers from `swallow-sglang:nightly-dev-20261018-b234b5c3`, with the review
source mounted at `/review` and `PYTHONPATH=/review/python`. Torch
`2.14.1+cu130`, FlashInfer `0.7.2`. CPU jobs were limited to four physical cores.
No running service or deployment configuration was changed.

| Check | Result |
| --- | --- |
| tools/qwen38_cache_tests/test_cpu*.py | 351 passed |
| New policy, resize protocol, B12X transform, KV stages, tail reloan, loader tests | 55 passed; 18 subtests |
| Kernel namespace inventory tests | 70 passed |
| Rust extension rebuilt; integration tests selected by reclaim_device_tail | 3 passed, 465 deselected |
| CPU scalar oracle, Triton math, 1/2/4 cores | 30 passed; max abs error 0.00012207031 |
| CPU scalar oracle, B12X math, 1/2/4 cores | 30 passed; max abs error 0.00024414062 |
| FP32 split cancellation regression | Passed |
| B12X GPU + CPU partition, H2560/I640, M1..3, slots 1/600/1024 | 12 passed, final BF16 max abs difference 0 |
| Triton GPU + CPU partition, H64/I32 and H160/I160, M1..3 | 24 passed |
| GPU byte migration, budgets 11/20/6/16/4 | 5 transitions passed |
| Real StagedExpertPool VMM mappings, budgets 32/24/16 | 6 release/growth transitions passed; retained bytes checked for all six tensors |
| Clippy, changed radix-tree crate, all targets, tch/doc-only, -D warnings | Passed |
| Pre-commit Python/C++/Rust formatting, registrations and other applicable checks | Passed except existing F821 below |

The first full pre-commit run stalled in rustup attempting the repository's
uninstalled Rust 1.92 toolchain. The final run used the installed stable
compiler, skipped the broad clippy-rust-workspace hook, and ran the changed
standalone radix-tree crate's clippy directly with offline dependencies.
Ruff still reports the pre-existing `conv_input` F821 in qwen4_exp.py's
_gather_at closure (original head line 1486). It was verified in the baseline
and was not suppressed or changed by this integration.

A later combined GPU rerun hit the existing 75 C temperature guard between
B12X and Triton; B12X had passed all 12 checks. Triton was rerun separately.

## Reproduction

Build the CPU library on the target host (the B12X symbol is new even though
legacy ABI 2 remains compatible with the Triton entry point):

```bash
python test/manual/kernels/moe/build_qwen38_cpu_hybrid.py --core 0 --output /tmp/qwen38_cpu_hybrid.so
python test/manual/kernels/moe/check_qwen38_cpu_hybrid.py --library /tmp/qwen38_cpu_hybrid.so
python test/manual/kernels/moe/check_qwen38_cpu_hybrid.py --library /tmp/qwen38_cpu_hybrid.so --backend b12x
PYTHONPATH=python python test/manual/kernels/moe/smoke_qwen38_cpu_hybrid_gpu.py --library /tmp/qwen38_cpu_hybrid.so --backend b12x
PYTHONPATH=python python test/manual/kernels/moe/smoke_qwen38_cpu_hybrid_gpu.py --library /tmp/qwen38_cpu_hybrid.so --backend triton
PYTHONPATH=python python test/manual/kernels/moe/check_qwen38_global_vmm_gpu.py
```

The CPU hybrid switch remains off by default. The archive intentionally changes
`preserve_cache_on_resize` to default true for the legacy path as well; an
explicit false remains supported there. Rust deployments must rebuild their
extension to expose reclaim_device_tail.

## Limits

These checks establish synthetic numerical partitioning and expert physical
mapping safety. They do not establish full-model accuracy or throughput,
CUDA graph serving replay, prefill/vision transitions, borrowed-arena recovery
under live traffic, or the entire real KV allocator's three-stage lifecycle.
The VMM test uses real expert mappings, but no model and no real KV allocator.
The miss threshold 33 is uncalibrated. No 80/110 tok/s claim is made.
