# GLM5-Next MLX MTP

The fusion work in `60419a5930` is complete for its guarded decode contracts;
see [GLM5_NEXT_FUSIONS.md](GLM5_NEXT_FUSIONS.md). This follow-up implements the
next step in both the model and HTTP serving pipeline. Enable it with
`--mlx-mtp-num-steps N`: `0` disables it (default), `1`–`16` selects a fixed
draft count, and `-1` adapts among `0, 1, 2, 3, 4, 6, 8`. Use the existing
`--mlx-enable-sampling` for sampling, grammar masks, logit bias and logprobs.

## Implementation

- `Model(args, enable_mtp=True)` retains the original checkpoint's prediction
  layer at `model.layers.num_hidden_layers`. The target `layers` property and
  target caches exclude it. Ordinary model construction still omits the head.
- The prediction layer uses plain MLA and MoE, without mHC. Normalize the
  following-token embedding and the target's final-normalized hidden state,
  concatenate embedding first, project, then apply the decoder. The head's
  `shared_head.norm` and the target's `lm_head` produce draft logits. This
  agrees with the [reference head equations](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/glm5_next/mtp.py).
- The original layer paths preserve mixed-bit quantization overrides.
  `load_mtp_model` strictly loads the local checkpoint, including the head's
  BF16 projection and 4/5/6-bit MoE weights; nothing is requantized.
- `greedy_mtp_generate` owns target and MTP caches and proposes one draft at
  a time. Prompt priming uses `(hidden[i], token[i+1])` pairs. After accepting
  a draft, the head absorbs its committed target-hidden/token pair before
  drafting again. Rejection preserves the root position in both models.
- `verify_mtp` executes the two rows separately inside each target layer,
  preserving singleton decode arithmetic. It retains immutable cache-array
  references after the root row. Rejecting the draft restores all four KDA
  or MLA/indexer cache entries and their offsets, without replaying the root.
  This arrangement can reuse a layer's recently read weights across rows.

## HTTP serving and adaptive steps

`MlxMTPServing` is integrated into `MlxModelRunner` and the MLX worker.
It verifies a chain of `N + 1` target rows for `N` draft tokens, retains raw
logits and snapshots privately, then publishes only one consumed row per
ordinary scheduler step. The normal sampler applies current grammar masks,
logit bias, custom decode processors and sampling to that row. It returns
logprobs from that same target distribution. If the next actual input differs
from the guessed input, the remaining private rows are discarded before any
of their logits or caches can be used. Drafting consumes no sampling RNG.

This preserves the existing per-token stopping, streaming, token allowances,
sampling seeds and output accounting. Two requests have independent private
windows. Finishing, aborting, retracting or clearing a request releases its
window. Native GLM still uses its existing non-overlap scheduler setting.
Do not combine native MLX MTP with `--speculative-algorithm`.

The unified radix cache stores only committed target and head state, including
the final target hidden row and deferred known-token pairs when adaptive mode
chooses ordinary decode. Chunk boundaries preserve the following-token pair.
Speculative snapshots use `value.view(value.dtype)`: a distinct Python array
object sharing the existing storage. Sharing the same Python object would let
the indexer's slice assignment mutate old snapshots. `mx.array(value)` would
copy the cache data and adds substantial per-row recurrent-state traffic.

Adaptive mode first explores budgets, scores the actual number of consumed
rows divided by elapsed serving time (including rejected work), and updates
moving averages. It periodically explores again so a changed context or
request mix can change the choice. Zero-step cycles defer head work until
it is needed, allowing a return to ordinary decode for poor draft acceptance.
Final windows are capped by the remaining output allowance and do not train
the score of an unclipped budget. The policy optimizes measured throughput;
it does not enforce an inter-token latency limit.

The standalone `greedy_mtp_generate` remains a singleton greedy model-level
benchmark, independent of the serving controller.

## Final HTTP measurements and recommendation

On this M5 Ultra and checkpoint, **one draft token is the best measured fixed
budget**. Use `--mlx-mtp-num-steps 1`. Adaptive mode is available with `-1`,
but exploration and noisy short-cycle feedback made it slower on this panel.
It is a throughput heuristic, not a guarantee of the globally best budget.

Same weights and `affine_fused`, sampling enabled, deterministic seed handling,
three short chat-template prompts (English, Chinese, Python), 128 output tokens,
two prompt orders. Each timing request flushes prefix state. Aggregate decode
rate is total tokens after the first event divided by summed decode intervals.
The disabled-MTP control is reused from the preceding sweep: the view change
only affects enabled MTP. Loading is excluded; warm TTFT stays around 146 ms.

| Draft steps | HTTP decode tok/s | Worst observed token gap |
| --- | ---: | ---: |
| 0 (ordinary decode) | 27.57 | 47 ms |
| **1** | **32.18** | **65 ms** |
| 2 | 28.74 | 93 ms |
| 3 | 21.12 | 141 ms |
| 4 | 14.93 | 201 ms |
| 6 | 12.37 | 245 ms |
| 8 | 9.62 | 320 ms |
| -1 (adaptive) | 29.31 | 292 ms |

One draft improves this HTTP decode rate by **16.7%**. Adaptive mode mixes
ordinary decode and mostly small windows after its initial exploration; its
first English request reaches only 26.96 tok/s, while the second reaches
32.33 tok/s. It does not beat fixed step 1 on this workload. The report includes
per-request budget histograms, rather than inferring budgets from throughput.

All 48 timing outputs match the corresponding ordinary-decode token hashes.
Each of the eight configurations also passes 14 HTTP contract comparisons:
chunked prefill, exact/suffix prefix reuse, seeded sampling, logit bias, regex
grammar, logprobs/top-logprobs, short budgets, stop tokens, concurrent requests
and client disconnect recovery. Cold/hot prefix cases are compared against
the matching baseline case, not against each other.

Raw timings, token/logprob hashes, contract panels and per-request depth/budget
counters: [glm5_next_mtp_http_20261031.json](results/glm5_next_mtp_http_20261031.json).
These are bounded short-context measurements and correctness panels. They do
not establish one best budget for every model, context length or concurrency.

Launch on the existing Apple Silicon MLX environment with a local snapshot:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --mlx-enable-sampling --mlx-mtp-num-steps 1
```

Use `-1` instead of `1` for adaptive mode. The default `0` keeps the MTP head
unloaded. `N` counts drafts; verification contains at most `N + 1` rows.
The existing MLX sampling limitations, including unimplemented penalties and
prompt input logprobs, are unchanged; see `--mlx-enable-sampling` help.

Reproduce the sweep (one local server at a time):

```bash
python benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir MODEL_PATH --output OUTPUT_PATH \
  --steps 0 1 2 3 4 6 8 -1 --output-length 128 --repeats 2 \
  --check-contracts
```

## Why longer draft windows slow down

The same-weight English 128-token diagnostic separates acceptance from
verification cost. `N` below means draft tokens, so a full verifier window
contains `N + 1` rows. Mean committed rows is
`(verification rounds + correct drafts consumed) / verification rounds`;
ordinary-decode tail steps are excluded. The final window can be shorter.

| N | Rounds | Correct drafts consumed | Mean committed rows | Model decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 65 | 61 | 1.94 | 32.69 |
| 2 | 51 | 76 | 2.49 | 29.19 |
| 4 | 41 | 86 | 3.10 | 16.44 |
| 8 | 42 | 85 | 3.02 | 10.46 |

For N=8, conditional acceptance at draft depths 1-6 is 36/42, 28/36,
14/27, 6/14, 1/6 and 0/1. Larger windows keep verifying rows that will be
thrown away; committed length saturates around three rows. These rates are
conditional on reaching a depth, and root positions differ across budgets.

A forced true-AR-token-chain control eliminates rejection entirely. Warm
verification medians for 2, 3, 5 and 9 rows are 52.6, 78.7, 176.4 and
271.9 ms. Verification still executes each target row with singleton
arithmetic; it does not obtain a batched projection's weight amortization.
Synchronized attribution on a separate 32-output panel puts target verification
at about 89-92% of decode time. Head-body attribution includes head priming and
is reported separately; it is not a normal-overlap throughput measurement.

Counters at N=1/2/4/8 show all mHC, KDA, router, affine FFN and down/combine
calls hitting their guarded paths. Their internal row shapes and dtypes stay
unchanged (for example mHC `(1, 1, 4, 4096)` and affine FFN
`(1, 1, 4096)`). No SGLang fusion-guard fallback was observed. These are
operator-path counters, not a claim about every internal Metal shader variant.

An additional snapshot implementation cost was found and fixed: constructing
`mx.array(existing_array)` allocates a data copy. On a 4-MiB test array it adds
about 4 MiB of active allocation. `view(existing_array.dtype)` adds no payload
allocation while isolating subsequent slice assignment. The regression test
checks both storage cost and snapshot correctness. This was a secondary cost;
the acceptance/verification tradeoff remains after removing those copies.

Raw acceptance counts, fixed-chain timings, path hits and synchronized costs:
[glm5_next_mtp_diagnosis_20261031.json](results/glm5_next_mtp_diagnosis_20261031.json).
Reproduce with `bench_glm5_next_mtp_diagnose.py --model-dir MODEL_PATH --output OUTPUT_PATH`.

## Initial standalone measurements

Apple M5 Ultra / 256 GiB; MLX 0.32.4, mlx-lm 0.32.2, Python 3.12.16.
Local `mlx-community/GLM-5.3-Flash-4bit`, revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`. Both modes use the same loaded
weights, including the MTP head, and the `affine_fused` optimization profile.
Process wired limit: 240 GiB; peak active Metal allocation: 189.84 GiB.
No model download, dependency change, weight conversion or OS limit change.

Three chat-template prompts, 25-28 input tokens, 128 generated tokens,
two opposite-order AR/MTP runs. Rates below are means of the two decode
rates, excluding prefill and the first generated token. EOS stopping is
intentionally disabled to compare the same output length.

| Prompt | AR tok/s | MTP tok/s | Speedup | Correct drafts / proposals |
| --- | ---: | ---: | ---: | ---: |
| English explanation | 28.26 | 34.08 | 1.206x | 61 / 65 (93.8%) |
| Chinese explanation | 28.34 | 32.42 | 1.144x | 58 / 69 (84.1%) |
| Python intervals | 28.15 | 33.98 | 1.207x | 61 / 66 (92.4%) |

Every 128-token output matches its same-prompt AR output in both rounds.
A separate 32-position forced-token panel has bitwise-equal full-vocabulary
logits, final hidden states, and all target cache arrays/offsets. These are
bounded short-context checks, not a task-quality or long-context guarantee.

Raw timings, output hashes and counts:
[glm5_next_mtp_20261031.json](results/glm5_next_mtp_20261031.json).
A prior 16-input/32-output synthetic smoke also matched AR, with 14 / 16
correct drafts; its completion rate included prefill and is not comparable
to the decode rates in this table.

## Reproduce

Use the existing MLX environment and a local original-layout snapshot:

```bash
python benchmark/mlx/bench_glm5_next_mtp.py \
  --model-dir /path/to/local/snapshot \
  --output /tmp/glm_mtp.json \
  --output-length 128 --repeats 2 --text-prompts --check-verify \
  --wired-limit-gb 240
```

Set the process wired limit only to a value appropriate for the host.
The script records hashes rather than generated responses and stops on an
AR/MTP output or verification-panel mismatch.

The existing GLM MLX test fixture now checks the head against independent
PyTorch MLA/MoE equations, mixed-quantization verification across index-pool
boundaries, rejection/acceptance cache continuation, token budgets and EOS.
Serving tests additionally cover per-row cache publication, adaptive zero-step
handoff, head/prefix snapshot restoration, external token choices and periodic
budget exploration. A two-request HTTP check exposed an existing CPU-token to
MPS relay scatter error even with MTP disabled; token relays now convert to the
buffer device explicitly, with a regression that fails on the old code.
