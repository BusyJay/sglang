# Qwen3.8 Stage A benchmark

Revision `oct13-a2` addresses R1–R3 from [STAGE_A_REVIEW.md](STAGE_A_REVIEW.md);
see [STAGE_A_REVIEW_RESPONSE.md](STAGE_A_REVIEW_RESPONSE.md) for evidence.

Start with [GPU_HANDOFF.md](GPU_HANDOFF.md) for configuration and executable
commands, [RESULT_SCHEMA.md](RESULT_SCHEMA.md) for measurement/result contracts,
and [VALIDATION.json](VALIDATION.json) for CPU evidence and unverified items.

`python3 -m tools.qwen38_cache_tests.benchmark --help` lists implemented commands.
The suite includes short/28K/50K inputs, separate tune/holdout inputs, explicit
warm/cold-process states, and the user's Raft → arbiter reasoning session. Save
one baseline first answer with `freeze-answer` for fixed-input prefill comparisons.
All model inputs use immutable saved token files and local tokenizer preparation.

CPU protocol tests: `python3 -m unittest tools.qwen38_cache_tests.test_cpu_benchmark -v`.
Full available CPU regression: `python3 tools/qwen38_cache_tests/run_cpu.py`.
GPU compile/parity/performance remain `not_run` until executed on the GPU host.

No 06/05/01, deployment default change, or historical speedup claim is included.
Implementation base: d1d85bf29b855e511d2a96f9a7f6cec73fc2bb1f; performance base:
384c83da44a4334dcaa74c474195ab960126e029. After committing benchmark files,
create the review bundle (outside the checkout to avoid provenance pollution):

```bash
python3 tools/qwen38_cache_tests/benchmark/package_stage_a.py \
  --output /absolute/new/qwen38-stage-a \
  --cpu-log /absolute/cpu-regression.log
```

The package script reads only the committed diff from the exact base, rejects
unrelated changes, checks/applies the patch in a clean isolated checkout, verifies
the full resulting Git tree, reruns protocol tests and CLI help, and emits
SHA256SUMS and a tar.gz. It includes no weights, tokens, credentials, caches or
unrelated untracked files. GPU baseline review is required before stage B (06).
