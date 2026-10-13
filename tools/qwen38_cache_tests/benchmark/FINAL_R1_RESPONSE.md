# Final r2 response to R1 review

Parent: bb5127f7bade1dbf15fcdd78163653382fc77530. Suite oct13-final-r2;
campaign/selection/correctness/profile schemas v2. No 06/05/01 runtime changes.
Historical r1 verification is retained in FINAL_R1_VALIDATION.json and old package.
GPU testing remains deferred; defaults L2 off/direct/group1 unchanged.

R1: identity now records absolute input_artifacts locators with file byte SHA256,
matching inputs hashes. Selection reopens every bound input and checks hashes;
compares first requests and warmup with frozen tokens; rebuilds measured history
from actual prior input/output, repeatedly stripping configured tail IDs before
adding each frozen suffix. Warmup raw files must equal unit copies and meet the
same output/count/hash/finish/SSE timing checks as measured requests. Tests cover
normal two-turn repeated-tail stripping, same-length wrong first tokens, history,
suffix, warmup tokens/raw corruption and changed JSON bytes with unchanged IDs.
All input locators are included in stage/profile binding. Preserve their paths
through selection/sealing/loading; new paths require a fresh campaign.

R2: every launch --execute mode uses check_effective for normalized JSON string
or dict loader and final server assertions, then checks model/revision. Malformed
JSON and mismatches reject. Real CLI execute tests simulate only owned process,
socket/server_info; verify success receipt/wait and finally cleanup on success
and failure. No actual server/GPU was started by these tests.

R3: parse argv groups, merge profile < machine deployment < explicit settings,
and emit one occurrence per flag. Equal syntax and separate values supported;
boolean flags require flag syntax. Existing assertion types convert values,
new deployment/explicit options create string or boolean assertions. Illegal
managed options/L2 overlap/concurrency fail validation. Explicit overrides stay
unvalidated; changed machine assertions are unvalidated too. Startup in every
mode checks these final expectations. Tests cover machine chunk4096 + explicit
chunk2048, separate-value syntax, flags, invalid values and stale effective chunk.

Protocol positive test builds the full screen/tune/holdout chain with multi-turn
and warmup, then uses explicitly simulated GPU metadata/report/trace markers to
exercise real sealing, profile reconstruction, loading and fake CLI execution.
It creates files only inside TemporaryDirectory and deletes them; this is CPU
protocol testing, not GPU correctness/performance acceptance. Foreign code and
CPU small report fail closed. No validated deployment profile is distributed.

Verification: /data/workspace/qwen38-final-r2-review-20261013
- cpu.log: 242 tests passed, zero skip; imports and ModelOpt wiring passed.
- campaign.log: 16 synthetic protocol tests passed.
- cuda.log: SM120 extension compiled/loaded offline; kernel not executed.
- Source AST parsing and git diff --check passed.

Commands:
```
tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
python3 -m unittest tools.qwen38_cache_tests.test_cpu_campaign -v
docker run --rm --pull never --network none -v /data/workspace/sglang:/workspace/sglang:ro -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 lmsysorg/sglang:v0.5.21 tools/qwen38_cache_tests/compile_cuda.py
python3 tools/qwen38_cache_tests/benchmark/package_final.py --output /data/workspace/qwen38-all-patches-20261013-r2 --logs /data/workspace/qwen38-final-r2-review-20261013
```

Recommended package series remains a2 → consolidated06r3 →05 →01 →final-tools.
Final tools patch is cumulative from 01, replacing final r1 tools. Each step
from all five starts is checked with an isolated Git index and target tree;
package.json and SHA256SUMS record exact metadata. Do not apply 10 twice.
GPU payload/asynchrony/ring/event/KV/graph/model/session/long-context performance,
actual hardware correctness seal, independent holdout and normal/profile overhead
remain not_run. Real GPU evidence and semantic review are required for promotion.
