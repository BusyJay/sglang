# Tiered CSF evaluation on RTX 5090D (2026-10-23)

The subsequent lossless memory/capacity work and FP8 1ULP investigation are recorded in [CAPACITY_REPORT.md](CAPACITY_REPORT.md) and [QSA_FP8_REPORT.md](QSA_FP8_REPORT.md). Those reports distinguish final-source serving checks from the earlier diagnostics below.

Latest update: the single-row workaround has been removed. The actual failure
was missing GPU bank binding after legacy threaded cache fills; the original
batched MTP expert path is restored and validated. Inline decoder optimizations
and capacity-bucket compilation reduce compute overhead and resize-time JIT
stalls. See the final section and `root-final-summary.json` for measured results.

Native-only 64/128-bit load experiments (2026-10-24) are documented separately
in [NATIVE_VECTOR_REPORT.md](NATIVE_VECTOR_REPORT.md). Their gains are small and
workload-dependent; native default dispatch remains unchanged.

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
- The original real-checkpoint probe is invalid: it allocated eight expert
  slots while routing to IDs 0..9. Its previous success claim is withdrawn.
  The corrected probe reserves at least top_k slots, asserts route bounds and
  finite outputs, and uses enable_fp32=False for A16 production-off parity.
  W4A4 results are measurements; exact assertions apply only to A16/Triton.

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

## Follow-up: arena restoration and control reproducibility

A second compressed-format integration defect was found in borrowed-arena
return. CPU source headers carry file offsets and a zero GPU bank pointer.
`_return_arena` reloaded the source bytes while retaining cache hit metadata,
but did not rebind external bank addresses. Restored hits could therefore skip
the decode miss-fill operation that normally performs that binding. The return
path now binds restored layers on the transfer stream before publishing completion.
No extra decode launch or synchronization was added to ordinary hits.

The new GPU runtime regression executes the real borrow/drain/return and bank
binding logic with substituted source transport. It verifies exact restored
header/payload bytes, inline headers, preserved hit metadata, immutable CPU
source headers and bank contents. It fails against the old return body and
passes against the fix. This proves a lifecycle bug; it does **not** establish
the cause of the current MTP generation failure.

Small end-to-end controls after the fix:

| Control | Result |
| --- | --- |
| Original dynamic/shared setup | Repeated-character output, rejected |
| Adaptive pool + shared arena, KV stages removed | Repeated-character output, rejected |
| Adaptive pool + dedicated arena, KV stages removed | Repeated-character output, rejected |
| Clone both compressed scale planes before each eager expert call, no host fence | Repeated-character output, rejected |
| Repeat fixed-cache/dedicated-arena control | Repeated-character output, rejected |
| Dedicated arena with original KV stages | Startup fails: profiled KV budget 129,728 < required 220,032; no generation |

The previously successful fixed-cache smoke is **not reproducible**. It cannot
be used to assign the remaining problem to adaptive cache, KV borrowing, or
shared arena storage. Copying scale operands also did not establish a slice
addressing explanation. No new performance matrix was run on invalid outputs.
`restore-diagnostics.json` retains the control settings and outcomes;
`run_matrix.py` adds independent `--dedicated-arena` and `--no-kv-stages` controls.
The clone module requires an explicit diagnostic PYTHONPATH and is not used by
production or ordinary matrix runs.

## Historical decision before the bank-binding correction

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
lifecycle choices at once; further isolation is necessary. Its apparent indication toward compressed layout and dynamic/shared-pool
integration was not reproduced in the follow-up above. It is
**not proof of a particular race**, and no blanket synchronization workaround
has been merged. Changing CSF reads to coherent L2 loads did not resolve the
issue and was reverted. Already-validated native serving paths were not changed.

`diagnostic-summary.json` records all 48 layers of the first live M=3 verify:
GPU cache weights, decoded scales, alphas and scalar values matched the native
checkpoint, and compressed/native outputs were bitwise equal. This probe
synchronizes and deliberately raises before the last routed-expert call returns;
it does not complete even the first verify forward, logits, acceptance or commit.
Its compact native reference also enables route-local alpha refresh via
`enable_fp32=True`, unlike production off, which refreshes the full capacity.
It cannot establish production full-round parity or subsequent-round correctness. The same summary records selected real prefill inputs against expanded
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

## Audit follow-up: legal routes and complete MTP generation

The coverage corrections above withdraw the original out-of-bounds real-pack
probe and the implication that 48 synchronized expert comparisons completed a
verify. `audit-real-pack-summary.json` records the corrected run: capacity 10,
IDs 0..9, production-off alpha refresh (`enable_fp32=False`), layers 0/5/47,
expert 0 replicated across slots. Native/CSF A16 and Triton are bitwise equal at
M=1/3; compressed M=3 and three M=1 calls also match for these static operands.
All measured outputs are finite. This does not cover diverse expert routing or
concurrent cache replacement. W4A4 scatter measurements are not exact-equality
assertions.

`root_diagnostic/sitecustomize.py` now has explicit diagnostic modes to change
only alpha refresh, split only the expert calls, or combine the two. Splitting
retains the original MTP draft, target attention, verify, acceptance and commit;
three expert rows are evaluated separately and concatenated in their original
order. Route rows must be cloned: top_k=10 makes subsequent row-view addresses
incompatible with B12X's 16-byte pointer alignment. The initial unaligned attempt
failed at startup and is excluded from generation conclusions.

New full-generation evidence (`audit-mtp-summary.json`):

| Diagnostic | Outcome |
| --- | --- |
| Unmodified CSF, CPU hybrid off | 128 tokens, 45 verifies, repeated `!`, rejected |
| Route-local alpha refresh alone, M=3 retained | 128 tokens, 44 verifies, repeated `!`, rejected |
| Split expert M=3 into 3 x M=1 | 128-token smoke completed; first run misinterpreted noisy input, so no semantic pass claim |
| Same split, another startup | 128-token warmup: 54 verifies; natural chat: 128 tokens / 55 verifies, relevant reasoning, no degeneration |
| Same split, extended natural chat | 128-token warmup: 56 verifies; natural chat: 512 tokens / 257 verifies, relevant continuous reasoning, no degeneration |

All new runs keep dynamic cache, borrowed prefill arena and MTP enabled, with
CUDA graphs disabled for isolation. The 512-token response is still reasoning,
not a completed final answer or an accuracy benchmark. These completed requests
exercise full verification rounds, unlike the deliberately aborted parity
probe; they do not establish native/CSF per-round bitwise equality.

Alpha refresh alone is insufficient. Splitting is a promising diagnostic
workaround, not an established root-cause fix: it also changes launch timing,
scratch reuse, allocation and alpha refresh frequency. Historical unmodified
runs sometimes avoided degeneration, so one successful run is weak evidence.
The current evidence narrows attention to the execution differences between
batched M=3 expert calls and sequential M=1 calls during live MTP serving; it
does not prove a scale codec defect or an M=3 arithmetic defect. Production
runtime code was not changed. Graph replay, long-context/resize behavior and
accuracy remain unvalidated for the split workaround. A useful next diagnostic
is to preserve the first divergent verify's live operands and completion events,
then compare batched and split execution on those same operands.

## Correction: single-row dispatch is diagnostic only

The single-row runtime dispatch described above has been withdrawn. It was a
diagnostic workaround, not an acceptable performance fix. Its successful runs
remain diagnostic evidence only. Production CSF execution again uses batched
M=3, while investigation continues toward a root-cause fix with batched
correctness and controlled performance validation.


## Root cause fixed with batched MTP retained

A capture taken after the original launch, without checks in earlier expert
calls, preserved the failure at target verify invocation 2, layer 5. Its inputs
were finite. The intermediate buffer had nonfinite values exactly on routes
whose w13 CSF heads had an external file offset but a zero GPU bank pointer:

| Cache slot | External file offset | GPU pointer before fix |
| --- | ---: | ---: |
| 34 | 58,506,813,440 | 0 |
| 47 | 58,506,928,128 | 0 |

Those routes poisoned two complete output rows. Selected alphas exactly matched
their expected products. Replaying with rebound pointers produced finite
native/CSF outputs. `root-cause-proof.json` machine-checks the correspondence
between bad intermediate routes and the missing pointers; captures are retained
locally under `batch-root-capture-l5-call2/` and `root-fix-capture-call2/`.

The missing path was the non-arena branch of the threaded transfer worker.
Short prefills can fill the ordinary cache through `_fill`, which copies raw
source headers. Arena uploads and device miss fills already bind their banks;
this worker branch did not. Subsequent device-planned hits skip fill and retain
the invalid zero pointer. The decoder then treats external records as inline
and can read neighboring allocated data. Thus successful pack roundtrips and
memcheck runs could not establish serving correctness.

The fix binds uploaded cache headers on the transfer stream before publishing
ready. It runs only for legacy jobs with misses and CSF banks. Ordinary device
hits/misses retain their existing launch sequence. Native transport is unchanged.
There is no single-row dispatch, blanket host fence, or expanded scale buffer.
The complete MTP protocol and original batched expert calls remain in use.

The new real-worker regression substitutes only source transport, then verifies
the published external/inline header bytes across a miss followed by a hit and
checks source immutability. It fails on the old code at the bank-pointer bytes
and passes after binding. The existing borrowed-arena regression also passes.

## Inline decoder and compilation improvements

The CSF load substitution now passes the already-resolved native expert ID and
local byte offset instead of recovering them with 64-bit division for each
scale word. Two aligned packed-code bytes use one 16-bit load; packed integer
operations reconstruct the four scale bytes together. The codec and FP4 weight
nibbles are unchanged. Native/CSF BF16 and FP32 parity, external banks, relocation,
masked routes and asynchronous refills pass the existing GPU tests.

An isolated benchmark uses the same real captured operands, repaired bank
bindings, 50 expert calls per graph and five timed batches of 100 graph replays.
Alpha refresh is retained; completion dependencies are internal graph events
for this operator measurement. Median GPU time per expert invocation:

| Implementation | Microseconds |
| --- | ---: |
| Native | 50.9 |
| Original CSF decoder | 120.3 |
| CSF with explicit expert ID | 106.6 |
| CSF with packed load/reconstruction | 91.6 |
| Final CSF with capacity buckets | 91.2 |

This is about a 24% reduction in CSF GPU call time, not a claim that CSF is faster
than native. The timing samples and exact scope are retained in the operator
logs and `root-final-summary.json`.

The first post-resize run of the optimized source exposed 12.915-second and
4.542-second stream gaps from new exact-capacity programs. CSF programs now
compile per power-of-two capacity bucket. FC1, FC2 and prefetch route resolution
check the actual runtime extent; physical tensor/pointer bindings still use
the real capacity. A regression shrinks a four-slot view to three while retaining
the fourth row in backing storage, verifies out-of-range routes are masked,
and verifies no additional program is compiled. Disabling capacity reuse makes
that test fail (`3 != 2` compiled programs). Native compilation is unchanged.
First use of a different bucket can still require compilation.

## Final full-model validation and performance

`root-final-bucket-csf-full` runs with the production breakable-graph command,
CPU co-compute off, the original MTP settings, dynamic cache, borrowed arena and
KV stages. It does not load diagnostic `sitecustomize` modules. All seven
workloads pass (1,176 complete verifies), including the 170,064-token input and the post-resize 512-token
request. Exact addition, counting and JSON checks also pass after those timed
requests. `root-final-tests.log` records ten passing tests: two source/address
unit tests, two real cache-lifecycle tests and six GPU tests.

One-pass comparison, using identical commands except native/CSF manifest and
encoding, with the same cache byte budget:

| Workload | CSF before decoder optimization | Final CSF | Native control |
| --- | ---: | ---: | ---: |
| Median of three short generations, tok/s | 87.35 | 96.65 | 99.57 |
| Natural chat, tok/s | 79.02 | 92.20 | 105.84 |

Final post-resize throughput is 86.28 tok/s overall and 96.55 tok/s after token
64. Its largest stream gap is recorded in the summary; the 13/4.5-second stalls
are gone. The long-context request is correctness coverage, not a controlled
long-context speed comparison. Generated text and acceptance differ between
runs, so the summary also reports acceptance length and milliseconds per verify.
CSF is still slower than native in these measured cases; compression retains
its storage/cache-capacity savings, but no universal throughput win is claimed.

These final root-cause and performance results supersede the withdrawn split
workaround and older provisional serving conclusions. Production services were
not restarted or deployed.
