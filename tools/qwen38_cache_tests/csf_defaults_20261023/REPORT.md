# Tiered CSF evaluation on RTX 5090D (2026-10-23)

This experiment compares the current checkpoint at an equal expert-cache byte
budget, with layered LFU routing disabled (`off`), enabled with no CPU experts
(`n0`), and enabled with at most one eligible CPU expert (`n1`). `n0` retains the
hybrid FP32/merge framework and CPU-inclusive LFU heat; it is not the original
GPU-only path.

## Correctness comes before performance

The initial CSF matrix completed requests but generated long strings of `!`.
Those timings are **invalid** and are excluded from `analysis.json` groups. The
apparent 30–50% speedup and near-perfect MTP acceptance were symptoms of broken
outputs, not optimization results.

The W4A4 prepared adapter allocated its scatter output with `torch.empty`, while
B12X's non-deterministic kernel atomically adds expert contributions. B12X's
public launcher supplies a zeroed output. The adapter now does the same; the
deterministic reduction still uses an uninitialized allocation because it
writes every output element. Earlier parity tests used deterministic reduction
and missed the production path. The new regression exercises the production
scatter path, fully masked routes, 128-token chunk boundaries, and poisoned
uninitialized BF16 allocations. Commit: `464a35cdf1`.

The benchmark wrapper now rejects degenerate repeated-character output instead
of declaring a request-completion-only run valid. This is a smoke check, not an
accuracy benchmark. Raw invalid runs remain locally available for diagnosis.

## Storage and format

Only the scale planes are compressed; FP4 weight nibbles remain unchanged.

| Property | Native | Initial fixed-stride CSF | Tiered CSF |
| --- | ---: | ---: | ---: |
| Payload GiB | 63.282 | 62.609 | 60.755 |
| Expert slot bytes | 2,764,808 | 2,735,384 | 2,654,216 |
| GPU exception-bank bytes | 0 | 0 | 4,505,600 |
| Slots at 17,661,593,504-byte budget | 6,388 | — | 6,652 |

Tiered CSF saves 3.99% of the whole payload and holds 264 more expert slots at
the fixed byte budget, after deducting the banks. Base scale slabs are 128 KiB
for w13 and 64 KiB for w2. Exceptional replacement arrays use resident 16 KiB
bucket multiples: 160 w13 experts and 24 w2 experts. Rare outliers no longer
force every expert slot to the largest slab size. The format implementation
was committed as `9da04a50d0`.

At the 80k KV stage, borrowing unused KV capacity gives 7,006 native slots or
7,296 tiered slots; at the final stage it gives 6,388 or 6,652 respectively.
Source mode is shared THP for both encodings, in the same startup-cache
namespace. The cache manager may replace an unleased source mirror when the
encoding changes. These are ephemeral RAM mirrors, not extra disk weight packs.

All payloads live beneath `/home/user/workspace/prod/cache/sglang`, mounted at
`/root/.cache/sglang` in the diagnostic container. Old archives and pack copies
were moved beneath `archives/`; byte-verified identical historical files now
share inodes. This reclaimed 126.563 GiB. The superseded fixed-stride CSF
intermediate reclaimed another 62.609 GiB; its manifest is archived. Total:
**189.172 GiB reclaimed**. Native and tiered payloads are preserved independently.
See `storage-summary.json` and `intermediate-retirement.json`.

## Validation

- Every one of the 49,152 compressed scale slabs decoded byte-for-byte to the
  native checkpoint, and 60,398,174,592 unchanged bytes were compared and hashed
  (`pack-validation.json`).
- Prior format/cache/CPU/GPU/lifecycle checks: 203 tests / 372 subtests; final
  targeted checks after prefill binding changes: 95 tests / 20 subtests.
- Compute Sanitizer memcheck: zero errors on tiered decode, prefill, relocation
  and bank binding.
- After the scatter fix: all five CSF GPU tests passed, including the new
  production-path regression. End-to-end output validation still failed, so
  this fix alone is insufficient; the corrected performance matrix stopped
  immediately rather than timing invalid generation.
- Real checkpoint operator probes (layers 0, 5, 47, expert 0; M=1/3/129) compared
  native and tiered B12X W4A4, B12X W4A16 and Triton paths successfully. These
  isolate scale decoding; they do not replace end-to-end generation checks.

## Benchmark method

Torch 2.14.1+cu130, CUDA 13.0, FlashInfer 0.7.2; RTX 5090D; CPU service restricted
to cores 2–5. Command and environment snapshots are stored per run. Single
request, BF16 activations, FP8 KV, 80k/160k/220k adaptive KV stages, B12X W4A16
decode/verify and W4A4 prefill. MTP: NEXTN, 2 steps, draft length 3, seed 42.

Each short run has a 128-token warmup, three 1,088→512-token salted prompts and
a 512-token natural-language response. Decode rate excludes the first stream
event and counts token deltas, including multiple tokens returned by MTP.
Inspect acceptance length and time per verify round before attributing changes
to routing. Request text is retained locally and excerpts/hashes are summarized.

After the user requested fewer tests, the corrected matrix runs **one pass**
over the three CSF routing modes, reusing valid native baselines. No reverse
repeat. Long-context verification is reserved for a viable default candidate.

## Current decision and remaining issue

**Provisional default: native weights, layered GPU cache, CPU co-compute off,
MTP enabled.** `recommended-loader-config.json` and `recommended-environment.json`
are configuration artifacts only; no production service was deployed or restarted.
The valid native baseline has median short decode 103.742 tok/s and natural
response 107.737 tok/s across its two already-completed rounds. Native n0 has
104.694 / 103.885 tok/s; native n1 has 102.659 / 101.906 tok/s. There is no
consistent co-compute gain in these workloads. The 170,064-token native request
has TTFT 30.024 s, followed by 95.666 tok/s on the short post-resize request.

The compressed dynamic-cache matrix remains invalid and must be rerun only
after correctness is fixed. Do not treat its high decode rates as a result.
The provisional default does not imply compression is intrinsically slower.

Diagnostic evidence recorded before this checkpoint commit:

| Single 1,088→128-token generation | Result |
| --- | --- |
| Tiered CSF, dynamic cache, MTP | Degenerate output, rejected |
| Same, no MTP | Normal response, 63.46 tok/s (smoke only) |
| Same, forced W4A16 prefill, MTP | Degenerate output |
| Same, Triton compressed decode/verify | Degenerate output |
| Same, CUDA Graph disabled | Degenerate output |
| Same, prefill completion fence only | Degenerate output |
| Same, decode completion fence only | Degenerate output |
| Same, decode entry fence only | Degenerate output |
| Same, per-call finite-input/output checks | Normal response; intrusive diagnostic |
| Tiered CSF, fixed cache, dedicated prefill arena, MTP | Normal response, 71.39 tok/s (smoke only) |

The final positive control disables adaptive cache, removes KV stage borrowing,
and uses a dedicated prefill arena. It changes multiple related allocation/
lifecycle choices at once; further isolation is necessary. It points toward the
combination of compressed layout and dynamic/shared-pool integration. It is
**not proof of a particular race**, and no blanket synchronization workaround
has been merged. Changing CSF reads to coherent L2 loads did not resolve the
issue and was reverted. Already-validated native serving paths were not changed.

`diagnostic-summary.json` records all 48 layers of the first live M=3 verify:
GPU cache weights, decoded scales, alphas and scalar values matched the native
checkpoint, and compressed/native outputs were bitwise equal. This probe
synchronizes and cannot establish correctness of subsequent uninstrumented
rounds. The same summary records selected real prefill inputs against expanded
scales/CUTLASS and native B12X. Its small numerical differences are distinct from
the degenerate end-to-end failure. The earlier selected-layer decode probe used
shape-sensitive scalar equality (scalar versus length-one vector); the all-layer
probe corrects that comparison and checks all scalar values successfully.

Two additional targeted regressions passed: the strengthened nonzero W4A4
scatter check, and 24 GPU slot refills/decodes chained without per-call host
synchronization (M=1/2/3, ten routes), with native/CSF outputs bitwise equal.
These supplement, rather than replace, end-to-end validation.

Reproduction: `run_suite.py` is a **single pass**, with no reverse repeats.
`run_matrix.py` supports `--smoke`, `--no-spec`, `--no-graph`, and `--static-cache`.
Smoke results are diagnostic only, not full performance comparisons. The
`diagnostic/` modules activate only when explicitly added to `PYTHONPATH`;
they are not imported by the production runtime. `probe_live.py` records the
last scoped-fence diagnostic setup; the saved individual probes preserve the
other instrumentation bodies. Raw logs, response streams and rejected outputs
are excluded from Git. Condensed measurements and exact commands are retained.
