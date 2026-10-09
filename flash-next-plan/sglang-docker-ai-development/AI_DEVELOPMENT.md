> Historical scaffold, retained for reference. Apply both patches first and use
> `flash-next-plan/README-ARCHIVE.md`, `tools/qwen38_cache_tests/RUN_ON_GPU.md`,
> and `flash-next-plan/sglang-gpu-switch-workflow/README.md` for current commands.
> The old docker-job/client-canary templates are incompatible with the current
> canary and are not supported test entry points.

# AI Development Protocol

Last updated: 2026-10-09

## 1. Start or resume

Read in order:

```text
this file
.agent-workset/RESUME.md
.agent-workset/COMPACT.md
.agent-workset/STATE.md
referenced DECISIONS/FILES/TESTS entries
```

Then run:

```bash
git status --short
git rev-parse HEAD
git diff --stat
```

If the worktree and workset disagree, update the workset first.

First-time setup:

```bash
tools/qwen38_dev/bin/init-worktree
```

## 2. Docker contract

All tests run in the configured image derived from SGLang v0.5.21.

```text
repo → /workspace/sglang
results → /workspace/results
PYTHONPATH=/workspace/sglang/python
```

The image provides `HF_HOME`; scripts do not override it.

Mounted Python changes take effect through `PYTHONPATH`. Native changes follow
the repository JIT path (`python/sglang/kernels/jit`) when possible. Changes
under `python/sglang/kernels/aot` require an explicit rebuild step in the 5090
job; mounting source alone does not replace an installed native extension.

Local config:

```text
tools/qwen38_dev/env.local
```

Required values: image, results path, 4080 device, 5090 device, small-test GPU budget.

## 3. Daily tests

CPU self-contained tests:

```bash
tools/qwen38_dev/bin/test-cpu
```

4080 SUPER small GPU tests:

```bash
tools/qwen38_dev/bin/test-gpu-small
```

Test code lives in:

```text
tools/qwen38_cache_tests/
```

Rules:

- Use `unittest`; do not run the full SGLang test suite.
- CPU tests use small in-repo fixtures and no model/server.
- 4080 tests use at most 8GiB by default and stay below 10GiB.
- Skip a GPU test when its declared free-memory requirement is unavailable.
- 4080/SM89 may test common cache/H2D logic and Marlin with checkpoint-format fixtures. It cannot validate the SM120 FlashInfer CUTLASS runtime-pack layout.

## 4. 5090 tests

Edit:

```text
tools/qwen38_dev/jobs/current.json
```

Validate without starting a model:

```bash
tools/qwen38_dev/bin/check-job \
  --job tools/qwen38_dev/jobs/current.json \
  --env-file tools/qwen38_dev/env.local
```

Before dispatch, update:

```text
.agent-workset/RESUME.md
.agent-workset/COMPACT.md
.agent-workset/STATE.md
.agent-workset/NEXT.md
.agent-workset/TESTS.md
```

Dispatch with the host command recorded in `.agent-workset/ENV.md`.

After acceptance:

```text
report job accepted
do not stop the backend
do not poll
do not wait
end the turn immediately
```

The standalone supervisor stops Qwen3.8-27B, runs the Docker job on 5090, restores Qwen3.8-27B and writes `.agent-workset/artifacts/latest-test.json`.

For a 5090 kernel-only job, set `start_server=false` and mark the test
`gpu=true`. Do not mark a test `gpu=true` while a full-model server container is
already using the 5090.

## 5. Full model client tests

Full-model tests are small client scripts. They read:

```python
os.environ["SGLANG_TEST_BASE_URL"]
```

They send deterministic requests, write concise JSON output and return a nonzero exit code on failure. Server startup/shutdown belongs to `docker-job`.

## 6. After compact or 5090 test

Read:

```text
RESUME.md
COMPACT.md
STATE.md
NEXT.md
artifacts/latest-test.json
```

Update `STATE.md`, `TESTS.md`, `NEXT.md` and `RESUME.md` before editing code again.

## 7. Do not run routinely

```text
complete SGLang unit/kernel/model suites
test/registered/... as a whole
distributed or multi-node CI fixtures
```

Read upstream tests as implementation references; project-owned tests are the daily gate.
