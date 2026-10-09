# Test Matrix

Last updated: 2026-10-09

## Test Policy

Daily tests are project-owned, self-contained `unittest` cases under:

```text
tools/qwen38_cache_tests/
```

Default host command (runs the inner unittest command in Docker):

```bash
bash tools/qwen38_cache_tests/docker_cpu.sh "$SGLANG_IMAGE"
```

Do not run the full SGLang test suite as a routine step. Files under
`test/registered/` are implementation references unless a milestone explicitly
requires one of them.

Test levels:

```text
CPU_SELF_CONTAINED
  Python stdlib + PyTorch CPU + safetensors

GPU_SMALL
  Explicitly select device; SM120 is required for NVFP4 CUTLASS parity
  One expert or a small MoE fixture; no server and no full model
  Current fixtures require 1–2 GiB free; allocated peaks are checked below 1 GiB

FULL_MODEL_MILESTONE
  Manual milestone validation only
```

## `<test ID>` — `<name>`

Purpose:

<What this test proves.>

Command:

```bash
<command>
```

Requires full model: `<yes/no>`

Level: `<CPU_SELF_CONTAINED/GPU_SMALL/FULL_MODEL_MILESTONE>`

Dependencies:

- `<minimal dependency list>`

GPU:

- Device: `<RTX 4080 SUPER / RTX 5090 / CPU>`
- Required free memory: `<GiB>`
- Estimated peak: `<GiB>`
- Measured allocated peak: `<GiB>`
- Measured reserved peak: `<GiB>`

Pass criteria:

- <Criterion>

Latest result:

- Commit: `<commit>`
- Result: `<not run/pass/fail>`
- Evidence: `artifacts/latest-test.json`
- Failure summary: `<none or summary>`
