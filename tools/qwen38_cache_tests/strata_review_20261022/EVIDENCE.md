# Evidence and reproduction

Tracked: diagnosis, source hashes, benchmark drivers and scalar timings, server commands, cropped trace summaries and the profiler skill's unedited three-table text reports.

Local only: full/cropped Chrome traces (`profile-*/trace/`, `profile-*/decode-only.json.gz`), server/build logs, copied CPU library variants (`pool-cv/`, `pool-spin/`), compiled `.so`. No model weights are added to Git. The original model remains in the existing startup shared cache; no source bytes were changed.

Environment: `swallow-sglang:nightly-dev-20261018-b234b5c3`, GPU 0 only, network none, IPC host. Mount this repository to `/workspace/sglang:ro`, `/home/user/workspace/Strata` to `/workspace/Strata:ro`, this evidence directory read/write to `/results`, existing `/home/user/workspace/prod/cache` to `/root/.cache`, the profiler skill to `/profile-skill:ro`, and the existing HF cache at `/hf-cache:ro`. The serving driver expects the offline model path and startup cache recorded in its command and env. Set `PYTHONPATH=/workspace/sglang/python`, `PYTHONDONTWRITEBYTECODE=1`, and offline HF/Transformers flags. CPU benchmarks reuse `/dev/shm/sglang-qwen38-startup`; do not create another 64 GiB cache.

Within that container, run sequentially (the CPU benchmarks consume dedicated physical cores):

```bash
bash /results/build_rows.sh
python /results/bench_rows.py
python /results/build_pool_variants.py
python /results/bench_pool.py
```

The row bridge uses `-march=native` for both linked row implementations; the pool comparison uses the production CPU library's explicit AVX512 compiler flags for both copies. Q8 activation conversion is excluded from the row timings. Cold eviction covers only the selected fixture weights/scales, not the machine's whole cache.

Only with GPU 0 free, run `python /results/profile_variants.py`. This launches and terminates its own benchmark servers, and does not manage production services. It generates GPU-only then LFU profiles serially. Then run `python /results/crop_decode.py`, followed by the skill on each cropped trace:

```bash
python /profile-skill/scripts/analyze_llm_torch_profile.py --framework sglang --input /results/profile-gpu/decode-only.json.gz
python /profile-skill/scripts/analyze_llm_torch_profile.py --framework sglang --input /results/profile-lfu/decode-only.json.gz
```

The cropped traces drop prefill plus the first decode/verify step and retain 1104 planner launches each. Metadata is retained; timed events are retained if their start falls in the cropped interval. The skill's automated fusion hints require manual review; see REPORT.md for rejected matches and limitations. Its three tables do not constitute proof of all suggested opportunities.

Both row and full-pool diagnostics completed with status `passed`. Row reference assertions and CV/spin bitwise output checks run in the benchmark scripts. There are 64 row configurations, 24 no-op pool configurations, and 48 full CPU configurations. Production code was unchanged after `8b02621849`; the existing 133-test/340-subtest result is not a new test run performed for this documentation-only diagnosis.

While analysis resumed, a production inference service was observed briefly using GPU 0 at idle; the user then stopped it. No production process was stopped by these scripts during the resumed CPU tests or trace analysis. CPU microbenchmark tails can be influenced by other host work; no isolated system-wide performance guarantee is implied. GPU 1's unrelated service was left untouched.
