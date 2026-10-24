# QSA FP8 paged prefill: exactness and memory (2026-10-24)

The optimized prefix-prefill path reads historical KV through the request's
logical-to-physical token table. It eliminates the full-history BF16 K/V copies;
all query rows, selected tokens, masking and attention arithmetic remain present.
MTP verification is unchanged and is never split into single-token calls.

## Origin of the one-ULP discrepancy

The original direct-FP8 experiment disagreed with the production packed-BF16
reference at two of 159,744 output elements (seed 173). The largest difference
was 1.862645149230957e-9. Investigation established:

1. Packing FP8 and then converting to BF16 produces exactly the same K/V values
   as the original packing/conversion kernel.
2. Paged BF16 matches packed BF16 bit for bit.
3. Both packed FP8 and paged FP8 produce the *same two* mismatches. Address
   translation therefore does not explain them.
4. The compiled Triton GPU IR uses MMA dot-operand `kWidth=4` when its backward
   slice contains an 8-bit source load, versus `kWidth=2` for BF16 source loads.
   This changes the FP32 accumulation order.

The compiler heuristic is visible in Triton 3.8.0's
[computeOrigBitWidth and AccelerateMatmul](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp).
The installed runtime reports Triton 3.8.0. The locally retained compiler IR
confirms the layouts actually used, independently of the upstream source.

The fix loads aligned pairs of FP8 bytes as `uint16`, converts each pair with native `cvt.rn.f16x2.e4m3x2`, then converts FP16
to BF16. Every finite FP8 E4M3 value is exactly representable in both formats. The load's source width is now 16 bits, so
Triton selects the reference's `kWidth=2` for both QK and PV dots. The bits and
precision of K/V are unchanged. All original mismatches disappear.

This paired-load change applies to the new paged path. Unsupported dtypes or
odd byte addresses/strides retain the established packing path. Existing direct
FP8 operator behavior outside the paged path is not changed.

Evidence: `qsa-fp8-diagnostic/summary.json` and its `packed_bf16.ttgir`,
`packed_fp8.ttgir`; `qsa-fp8-half-final/summary.json` and corresponding IR.
The final comparison retains the two mismatches in the unchanged legacy
direct-FP8 path while paged FP8 has zero mismatches against packed BF16.
`diagnose_qsa_fp8.py` reproduces the four-way comparison and retains IR/PTX/SASS.

`diagnose_qsa_stages.py` additionally captures intermediates from diagnostic-only
kernel clones. Its reference and old-path outputs exactly match their
uninstrumented counterparts. The observed final ULP changes specifically trace
to the PV FP32 accumulation:

| Paired loads restored | Different QK score elements | Different PV accumulator elements | Different final BF16 elements |
| --- | ---: | ---: | ---: |
| Neither (old direct FP8) | 15,891 | 10,558 | 2 |
| Keys only | 0 | 10,558 | 2 |
| Values only | 15,891 | 0 | 0 |
| Keys and values | 0 | 0 | 0 |

Scaled Q and the final softmax normalization sums are identical in all four
cases. With both fixes, every captured intermediate is identical. The QK
maximum absolute FP32 difference before correction was 9.313225746e-10; the
PV accumulator difference was 4.768371582e-7 before normalization. Both are
zero after correction. Full evidence is `qsa-fp8-stages-half-final/stages.json` and the
per-variant Triton GPU IR.

## Verification

`test/registered/kernels/ops/attention/qsa/test_qsa_paged_prefill.py` compares
against the actual packing kernel followed by the existing attention kernel,
using strict `rtol=0, atol=0`:

- All 256 FP8 byte encodings: finite values, including signed zero, match
  BF16 bits; both NaN codes remain NaN.
- BF16 history length 219,007, 257 query rows.
- BF16 variable-length requests with shuffled physical slots and padded strides.
- FP8 variable-length requests with four random seeds, including the original
  failing seed, negative-index masking and shuffled physical slots.
- FP8 history length 219,007, **8,192 query rows, 24 query heads, 2 KV heads,
  head dimension 256, top-k 2051 with valid and invalid tail entries**: all 50,331,648 output elements are identical.

Compute Sanitizer memcheck: **0 errors**. See
`capacity-qsa-native-half-memcheck-final.log` (5 tests executed, no skips).
API error reporting was disabled for this memcheck run: the separate raw log
`capacity-qsa-native-half-memcheck.log` records seven import-time
`cuGetProcAddress_v2` invalid-argument probes from the CUDA Python bindings;
all five tests also passed there, and no device memory violation was reported. This is operator-level exactness evidence;
full serving capacity results are recorded separately in `CAPACITY_REPORT.md`.

## Memory measurement

At 219K history and an 8,192-token query chunk, the measured additional PyTorch
allocation peak falls from **549,321,728** to **100,663,296** bytes: a saving of
**448,658,432 bytes, approximately 428 MiB**. Existing cache/input storage is
excluded from both sides. The new 96 MiB allocation is the attention output.

These are allocation peaks for this operator, not an assertion that whole-model
peaks decrease by the sum of savings from unrelated phases. Full-model runs
retain the 80K/160K/220K KV stages, 8,192-token prefill, original weight/KV/state
precisions and full batched MTP.


## Conversion compatibility and performance

A direct native `cvt.rn.bf16x2.e4m3x2` experiment was exact and fast with
Triton's bundled CUDA 13.3 assembler. It requires PTX 9.2 / CUDA 13.2+. However,
loading the serving/test dependencies selects `/usr/local/cuda/bin/ptxas` through
`TRITON_PTXAS_BLACKWELL_PATH`, which is CUDA 13.0 in this container. Therefore
that experiment is **not the final implementation**. The final FP16 intermediate
uses the older native pair conversion supported by this serving environment;
unsupported hardware/assemblers retain the established packing path. CPU tests
cover the hardware and assembler fallback decisions.

Final operator benchmark (`bench_qsa_paged_prefill.py`,
`qsa-paged-production-toolchain-performance.log`, `qsa-paged-performance.json`):

| API | Median time |
| --- | ---: |
| Original full-history packing + attention | 12.161 ms |
| Attention with history already packed | 11.605 ms |
| Final direct paged FP8 attention | 12.206 ms |

These are seven rotating CUDA-event samples per mode, 8192 queries, 219007
history tokens, top-k 2051, random physical slot order and uniform historical
selections. Every mode's output was checked bitwise before timing. This uses the same `/usr/local/cuda/bin/ptxas` **CUDA 13.0** assembler selected
by the serving process, explicitly set with
`TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas`. The measured difference
is **0.37%** versus packing + attention, within the limits of this small timing
series; no universal speedup is claimed. An earlier run of this same compatible
conversion using Triton's bundled CUDA 13.3 assembler measured 13.905 ms
against 12.216 ms (`qsa-paged-half-performance-final.log`). Those earlier
13.8% figures do not represent the production toolchain. It replaces an earlier correct
but slow software conversion experiment (~21.47 ms). Whole-model TTFT and decode
measurements use the final source and are reported in `CAPACITY_REPORT.md`.
