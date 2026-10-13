# Final r3 response and independent experimental-defaults patch

FINAL_R2_REVIEW original R1/R2 and scalar ordering remain fixed. New P1 addressed:
server_options.py reads local ServerArgs namespace type/Arg metadata using AST,
covering bool/int/float/Optional/list/Literal and arity. BooleanOptionalAction
positive/negative flags share a canonical key. Custom pure parsers are compiled
from exact production functions in utils/common.py; other custom actions defer
to actual ServerArgs.add_cli_args and fail closed if dependencies unavailable.
Groups are consumed completely and merged profile < machine < explicit; no
old-assertion type guessing. Each final group's typed assertion is checked at
startup. Tests cover one/two encoder URLs, existing list assertion, complete
list replacement, Optional[int] mamba size, float ratio, SI/IEC context integer,
BooleanOptionalAction, JSON list parser, invalid inputs and fake execute.
Full container independently compares these results with actual ServerArgs CLI.

Independent argv commit: 49d67ac22ef5dcb1ee07f06860588829f520ed95,
parent 71b73e84161c2c3bdf9047f5a64abed9eb283aa2.
Independent defaults commit: 12d8a01e97101edfc7d229b231409e5798fa8862,
parent 49d67ac22ef5dcb1ee07f06860588829f520ed95.
Final metadata commit is listed in package.json (parent defaults commit).

User explicitly authorized all features enabled for testing. Actual missing
loader/runtime values are threaded/dynamic/packed/packed/group2. Shared resolver
preserves explicit overrides. Startup resolves/publishes loader JSON and overlap
off for L2; model_loader_extra_config is declared resolvable. Non-Qwen models
and unpublished CPU pools do not request L2. Unsupported L2 backing/rank/layout
errors include l2_mode=off shutdown guidance. No silently disabled feature.
Default byte/copy thresholds retain direct fallback. Packed DMA/descriptors
remain single-expert; group2 affects direct/fallback, not merged packed DMA.

Normal campaign experiment requires no profile and starts user-authorized
experimental settings; canary defaults match and validates effective loader
plus overlap before requests. Ordinary fake execute passes without GPU/profile.
Explicit off/direct/direct/group1 and whole-layer switches remain. Benchmark
baseline is explicitly all-off/overlap-on, each suite candidate must declare all
four feature modes; overlap-off controls remain. Packed matrix canonical group2
avoids duplicate candidates and includes actual experimental settings.

Verification: /data/workspace/qwen38-final-r3-review-20261013
- cpu.log: 251 tests passed, zero skip; real imports/ModelOpt passed.
- campaign.log: 20 CPU synthetic protocol tests passed.
- cuda.log: offline SM120 routing/scatter compiled/loaded; kernel not executed.
- matrix.json: 32 legal/idempotent unique candidates; CPU configuration only.
- cli-experiment-help.log: ordinary experimental entry exists.
- AST parse and git diff --check passed.
- Earlier failing preparation runs are retained as cpu-pre*.log history; final
  cpu.log supersedes them. Failures exposed incomplete old fake server_info and
  fixture assumptions; fake responses now carry actual requested configuration,
  legacy benchmark fixtures explicitly select their intended candidate set.

Commands (actual source during validation was the isolated worktree, mounted
as /workspace/sglang; final shared tree contains the same sources):
```
tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
python3 -m unittest tools.qwen38_cache_tests.test_cpu_campaign -v
docker run --rm --pull never --network none -v /data/workspace/sglang-experimental-defaults:/workspace/sglang:ro -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 lmsysorg/sglang:v0.5.21 tools/qwen38_cache_tests/compile_cuda.py
python3 tools/qwen38_cache_tests/benchmark/package_final.py --output /data/workspace/qwen38-all-patches-20261013-r3 --logs /data/workspace/qwen38-final-r3-review-20261013
```

Recommended series: a2 → consolidated06r3 →05 →01 →tools/argv fix →defaults
→delivery metadata. All seven starting points and remaining steps checked with
isolated index/target trees. Historical defective 06r1/r2 not replayed. Defaults
patch is independent and reversible, and historical packages remain intact.
Same-tree ablations are not independent 04 source trees.

GPU correctness/performance, actual KV/graph/session/long-context, normal/profile
overhead and independent holdout are not_run. Added GPU default-constructor test
attaches real borrowed VMM handles, checks payload and reclaims; prepared, not
executed. Existing transport tests explicitly choose legacy off/direct/group1
where that is their intended path. No validated acceleration profile generated.
Synthetic sealing/loading tests remain temporary CPU protocol fixtures only.
