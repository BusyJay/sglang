# MLA implicit index validation — 2026-11-05

Base: c8f74850649506db91563b92c2d96341452933d2, M5 Ultra 256 GiB,
MLX 0.32.4, TF32=0. Target revision
76348ddb2be2e35e39ecfc00e75dc07703f4f3d5; DFlash2 revision
bf582e4eacc1810f76656d1811693ff6c6737d2a.

Continuous MLA verification rows address positions directly instead of building
and padding an equivalent index table. Visible lengths and segment split remain
runtime inputs. Selected attention preserves its original indexer output.
No arithmetic, quantization, cache snapshots, or request publication changes.

Default D3, configured BF16 KV/prefill, W4A16 verify, eval groups 2/4/2,
prefill chunk 256, cache 1 GiB, serving process memory/wired budget .78:

- Two short-prompt ABBA blocks per prompt, 192 outputs, three prompts:
  71.150 vs 71.116 tok/s; no demonstrated short-output improvement.
- One fixed 1024-token-history ABBA: 59.433 / 60.300 / 60.534 / 59.346 tok/s;
  aggregate 59.389 → 60.417 (+1.73%). This is a single-history measurement.
- Tokens and acceptance statistics match. Peak model memory remains about
  172.54 GiB; no model-level memory reduction is claimed.

90 MLX device/runtime tests and HTTP 17 cases, concurrency and abort passed.
Before committing, the unchanged final source was checked again with the
6 ragged kernel and 66 GLM tests. Format, fatal lint and diff checks passed.
Namespace inventory previously passed 122 cases, with one unrelated import
probe failing because this Metal environment does not install Triton.

Raw workspace evidence: fullmodel.json, summary.json, final_tests.json,
http_final.json, format_equivalence.json in this directory. Timing does not
include per-layer synchronizations. No claim of 90 tok/s or complete paging.
Seal concatenation, tail padding, fixed workspaces and GPU acceptance remain
separate work. Numerical tests cover runtime lengths/split through 8K, empty
rows, snapshot continuation and packed FP8's general attention branch.
