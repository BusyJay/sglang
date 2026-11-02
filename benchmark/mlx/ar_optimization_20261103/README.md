# GLM FP4 non-speculative AR optimization — 2026-11-03

Target: Apple M5 Ultra, 256 GiB, applegpu_g17d, MLX 0.32.4.
Checkpoint nvidia/GLM-5.3-Flash-FP4-Local revision
76348ddb2be2e35e39ecfc00e75dc07703f4f3d5. Baseline commit
269c57fcd60a6d1575f909c0c8eab43c3bc0f9d8. TF32 disabled; no MTP.
Full AR weights occupy 169.1578 GiB. Same process memory/wired budget 188 GiB,
cache 1 GiB; decode/prefill groups 2/4; buffer thresholds 200 ops/1024 MB.
These threshold settings are experimental runner choices, not independently proven optimizations.

## Adopted mechanism

MXFP8 projects real rows without allocating/processing zero-padded four-row
activations. It keeps native wide QMV's KL16 lane grouping, FP32 dot order,
per-group UE8M0 scale and reduction tree. Actual rows share decoded weights
in tiles of up to four. K/N/count are runtime arguments. The original packed
checkpoint bytes, activation/output dtype and equations are unchanged.
The production kernel lives under gemm with a lazy Metal registry entry.

MXFP8-only 128-token forced greedy short-context ABBA, sky/code/Chinese prompts:

| arm | sky | code | Chinese | sky repeat | code repeat | Chinese repeat |
|---|---:|---:|---:|---:|---:|---:|
| baseline tok/s |27.9349|27.9430|27.9294|27.8770|27.8717|27.9089|
| optimized tok/s |29.5438|29.5366|29.5461|29.5288|29.5482|29.5389|

Means 27.9108 → 29.5404 tok/s (+5.84%). 42 tok/s is not reached.
Raw per-token times, prefill, memory, generated IDs: final_ab.json.
All six paired generated token sequences match. Forty real-checkpoint
shape/dtype/count (1/3/6/9) projection checks are exact. Device tests cover
native rounding, tails, strided views and scale extremes. Seventy relevant
model/kernel tests plus 72 subtests pass. Namespace suite:117 pass, one
pre-existing AMX lazy-probe failure in unchanged srt/utils/common.py.
Black/pre-commit unavailable; existing Ruff formatter used, no installs.

Block-FP8 additionally hoists weight scales across each K128 block, retaining
the original lane-strided four-element accumulation order and input/weight
rounding. Real-checkpoint36 comparisons (BF16/F32, count1/3/6) are exact;
all six64-token sequences match. Full ABBA29.34–29.54→30.50–30.57 tok/s.
Final production also guards output tails before scale loads. Combined
128-token final qualification is recorded separately in combined_ab.json.

## Profile and NA selection

Metal System Trace decode_steady.trace contains a steady decode window;
initial tracing attach pauses are excluded from the GPU union. Compute active
3588.208 ms /3759.146 ms =95.45%; this measures scheduling duty, NOT ALU,
NA occupancy or DRAM utilization. CPU sampled stacks include driver submission,
MLX array evaluation and allocator recycling. Submission durations overlap and
must not be summed as CPU blocked time. Shader lists prove dense thin/fused
NAX already executes. Quantized AR paths remain mostly MXFP8 wide QMV and
NVFP4 gather_qmv_fast. Shader timing/counter tables in this System Trace are
empty: kernel-name presence does not give per-kernel time attribution.
The trace's shader inventory also includes previously compiled prefill shaders;
that inventory is not a per-decode call count.

MLX 0.32.4 source: applegpu_g17d native quantized matrix threshold is M>=32
for K,N<=2048; M>=18 for K,N<=4096; otherwise M>=12. Matrix-path NAX also
requires K%64==0, supported transposition/mode and non-FP32 input unless
TF32 enabled. Nonbatched split-K may select an older non-NAX matrix kernel
before qmm_nax. A singleton cannot invent real independent token rows.
Padding 16/32 rows was tested; full AR fell to about24.4 tok/s. Changing
dtype or TF32 is not adopted as a way to satisfy the selector.
Direct MPP narrow NAX candidates were also tested: block-FP8 F32 with
relaxed precision off; MXFP8 BF16 rows1/4/12, split1/2/4/8/16. They preserve
packed inputs but change reduction order. Their target microbenchmarks do
not beat the exact wide path. They remain diagnostic; no platform limitation
is claimed. Next mechanism would reduce register zero-row MMA/dequant work
and keep block scales outside the group dot while validating routing/quality.

## Other candidates

NVFP4 narrower output tiles1/2/4/8: a broken diagnostic grid was corrected
before use. BF16 micro outputs match; FP32 max difference <=9.54e-7 needs
further arithmetic localization. Compiled micros can favor narrow tiles, but
actual full AR ABBA regressed29.55→28.77 tok/s. The current manual candidate
adds wrapper/scalar operations and is not an MLX native tile selector change.
An integrated native selector/kernel remains unfinished.

Dynamic E4M3 direct bit-rounding replaced per-element binary search in a
diagnostic, retaining ties-to-even. Twelve dtype/boundary/random checks match
all bytes/scales. Full-model29.5 tok/s differences fall inside baseline drift;
not adopted, mechanism cost below measurable whole-model savings here.

Block-FP8 per-shape RPS micro-selected plan regressed full AR28.04→27.1.
RPS tuning must use actual dtype, dispatch graph and weight locality, not
standalone minima alone.

ICB + argument buffers: icb_probe.mm/json uses128 dependent dispatches with
explicit barriers and all-path correctness. Synthetic FP8 chain ICB reduces
CPU submit .0528→.0147 ms, completion2.4668→2.4131 ms; vector case GPU worsens.
This is NOT full-model replay. Native MLX ICB lifetime/resource integration
and MPSGraph projection integration remain unfinished. SDK APIs exist; a
safe implementation must retain graph-owned buffers and MLX stream dependencies.
No large full-model GPU debugger capture is needed; its resource duplication
was excessive and the incomplete capture was removed. System Traces retained.

## Reproduction / scope

Run final_ab.py with launch_ar.sh environment; remote uses the persistent
/Users/jay/workspace/prod/common/sglang-metal/bin/python. No installs, downloads,
system wired-memory changes or checkpoint requantization. Experiments, raw
JSON/logs and source candidates remain in this directory. Do not package
weights, environments, traces or cache data. Short AR is qualified; long-context,
MTP full-model throughput after this mechanism still needs separate validation.

Existing swap usage was nonzero before experiments. MXFP8-only final run
reported1226.88→1271.75 MiB system swap usage; this is system-wide, not
proof of which process paged. No memory limits were increased.

## Combined final result

combined_ab.json: baseline mean28.04046, optimized30.69145 tok/s (+9.45%).
All six paired128-token sequences match. Optimized steady range30.6636–30.7058.
First code/Chinese prefill0.4626/0.4546s includes new compiled shapes; their
second calls0.2428/0.2349s. Baseline0.18–0.25s. The mechanism does not remove
all first-shape compilation. Existing swap reported1273.19 MiB before combined
run; see raw system-wide after value in combined_ab.json.
Precomputing NVFP4 calibrated global scales had no meaningful benefit:
30.707/30.696 baseline versus30.718/30.700 candidate (scale_cache_ab.json).
It is not adopted; avoid added mutable weight-cache ownership for no benefit.

## HTTP and deployment

AR server restored on127.0.0.1:30007, launch_ar.sh, MTP steps0.
Six128-token HTTP requests pass. Whole-request throughput (includes prefill,
sampling/scheduler and HTTP) first calls27.52–27.67, repeated28.94–29.26tok/s.
Server steady decode logs29.8–29.9tok/s. These are optimized-only qualification,
not an HTTP baseline A/B; raw http_ar.json and ar_server.log.

Remote mechanism commits415e14a561 (unpadded MXFP8),4ac0c74624 (block-scale
hoist). No push. Inherited pyproject edits/deletion are not staged.
Ruff check on new kernel passes. Existing glm5_fp4.py/test file have six
pre-existing Ruff import/style findings; no unrelated rewrite.
