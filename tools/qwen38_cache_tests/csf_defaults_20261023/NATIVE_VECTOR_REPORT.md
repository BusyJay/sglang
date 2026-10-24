# Native 64/128-bit load evaluation (2026-10-24)

Implemented and measured native, uncompressed B12X FC1 weight loads. Default
dispatch remains unchanged: `native_load_bits=0`. Explicit 64/128 options are
available on `CachedB12xW4A16` for further tuning. No service was deployed or
restarted. M=3 remains one complete expert call; there is no row splitting.

## Implementations

Qwen38 has five K16 blocks per lane: 40 packed weight bytes. The original
generic aligned FC1 loop loads each block with two 32-bit instructions.

* 64-bit: merge each adjacent pair, preserving consumption/accumulation order.
* 128-bit final candidate: load gate weights with three aligned 128-bit reads,
  and consume up weights with 64-bit reads to limit simultaneous register
  demand. The API option denotes the maximum load width, not every load's width.
  Alternate lanes start at an 8-byte offset within a 16-byte boundary. The
  helper reads 48 aligned bytes and selects the useful 40; the extra eight
  bytes overlap neighboring lane data. The last lane ends at the row boundary.
* Also tested an exact-byte 128-bit implementation with alignment branches,
  both gate/up vectors loaded together, and the reversed up128/gate64 split.
  These were worse or less consistent; their timing artifacts are retained.

Source guards reject changed upstream load addresses rather than silently
merging non-adjacent weights. The vector adapter accepts only native
H=2560/I=640 weights; compressed weights retain their separate path. Generated
namespaces have separate compiler cache keys and a bounded source-code cache.

## Complete expert-call timings

RTX 5090 D, FlashInfer 0.7.2, native FP4-Local checkpoint, 147 physical slots,
top-k 10, int32 IDs, BF16 input/output. CPU co-compute is off. Both controls and
candidates use full-capacity alpha refresh and identical prepared launch ABI.
Weights are real checkpoint experts 0..9 placed in legal slots 1..10; activations
are seeded synthetic inputs. No illegal routes occur in performance tests.

Each graph contains four sweeps of the selected layers. Nine timed samples
alternate candidate order; each sample replays the graph 30 times. Values below
are median microseconds per complete adapter call, including alpha refresh.

| Working set | M | Original | 64-bit | Final 128-bit candidate |
| --- | ---: | ---: | ---: | ---: |
| Five layers in rotation | 1 | 35.089 | 35.144 | 35.284 |
| Five layers in rotation | 2 | 42.049 | 43.950 | 41.735 |
| Five layers in rotation | 3 | 53.369 | 54.961 | 52.946 |
| One repeatedly used layer | 1 | 26.786 | 26.727 | 26.880 |
| One repeatedly used layer | 2 | 37.487 | 37.726 | 37.214 |
| One repeatedly used layer | 3 | 50.159 | 50.321 | 50.168 |

For M=3, 64-bit is approximately 3.0% slower with five layers in rotation.
The final 128-bit candidate improves that case by approximately 0.8%, but
has effectively equal hot-layer performance and slightly worse M=1 results.
M=2 improves approximately 0.7% in both final comparisons. An earlier repeat
also showed a small hot-layer M=3 regression. These do not establish a broad
throughput improvement, so neither candidate is enabled automatically.

Raw samples: `native-vector-final-stream.json`, `native-vector-final-hot.json`.
Earlier experiments: `native-vector-initial.json`, `native-vector-aligned.json`,
`native-vector-hot.json`, `native-vector-gate128*.json`, `native-vector-up128.json`.

## Machine-code evidence

Extracted CUBINs from B12X compile-cache objects and disassembled with
`cuobjdump --dump-sass`. Files and resource reports are in
`native-vector-sass/`, especially `summary-final.json` and `final-*.sass`.
Counts are static across the complete compiled kernel, not dynamic per-call
instruction counts.

| Property, M=3/capacity=147 | Original | 64-bit | Final 128-bit candidate |
| --- | ---: | ---: | ---: |
| `LDG.E.CONSTANT` (32-bit) | 300 | 260 | 260 |
| `LDG.E.64.CONSTANT` | 0 | 20 | 10 |
| `LDG.E.128.CONSTANT` | 0 | 0 | 6 |
| Registers/thread | 128 | 128 | 128 |
| Stack frame, bytes | 8 | 16 | 48 |
| Static stack-store instructions | 1 | 3 | 9 |
| Static stack-load instructions | 1 | 6 | 12 |

Widening really reduced load instructions; it also increased stack traffic
under the kernel's register limit. That is a concrete countervailing cost.
This evidence alone does not attribute every timing difference to spills.

## Correctness and checks

* Two CPU tests pass: ordered word consumption and rejection of changed
  adjacency (`native-vector-unit.log`).
* GPU test passes for both widths and M=1/2/3, BF16/FP32 outputs, fused hybrid
  results, invalid/masked routes, and graph replay after routes/alphas change.
  Comparisons are bitwise (`rtol=0`, `atol=0`); references must be finite and
  nonzero. Performance scripts separately check exact native parity on all
  five actual checkpoint layers.
* Compute Sanitizer memcheck: zero errors (`native-vector-memcheck.log`). The
  run uses `--report-api-errors no`: the initial run reported seven
  `cuGetProcAddress_v2` import-time API probe errors, with tests otherwise
  passing; that full log is retained as
  `native-vector-memcheck-api-probes.log`. Memory-access checks stay enabled.
* Black checks pass on all five changed/new implementation and test files;
  `git diff --check` passes. Ruff is unavailable in this environment.

No full-model MTP round or model-level throughput benchmark was rerun for these
opt-in candidates. The results above cover complete expert calls and graph
replay, not a claim of full-model speedup or general model accuracy.

Reproduction:

```sh
PYTHONPATH=python SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1 \
python test/manual/kernels/moe/bench_qwen38_native_vector.py \
  --manifest /path/to/native/manifest.json --output stream.json
# Add --layers 5 for the repeatedly used single-layer case.
```
