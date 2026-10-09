# Qwen3.8 Flash-Next archive entry point

This directory is sufficient to transport the implementation and test tooling:
all changed source, scripts and metadata are embedded in the two patches.
No model weights, Docker image or Python environment are included.

1. Prepare a clean SGLang checkout at
   `e00930c5489053f26d86b179cee0d087f846acbb` (v0.5.21).
2. Verify patch checksums and apply the two patches **in order**:

```bash
(cd /absolute/archive/flash-next-plan/patches && sha256sum -c SHA256SUMS)
cd /absolute/sglang
git apply --check /absolute/archive/flash-next-plan/patches/01-cpu-validated.patch
git apply /absolute/archive/flash-next-plan/patches/01-cpu-validated.patch
git apply --check /absolute/archive/flash-next-plan/patches/02-gpu-unvalidated.patch
git apply /absolute/archive/flash-next-plan/patches/02-gpu-unvalidated.patch
```

3. Read `RUN_ON_GPU.md` here or the restored
   `tools/qwen38_cache_tests/RUN_ON_GPU.md`; choose your local image, GPU,
   checkpoint and results paths explicitly. Start with CPU and small GPU tests,
   then one-layer conversion/parity, full conversion, and full-model canary.

Patch 1 is the original 23-test CPU snapshot. Patch 2 contains later fixes and
experimental GPU/async implementation; current regression is 57 CPU tests plus
container import/config checks. CUDA extensions cross-compiled/loaded for SM120.
Actual GPU copies, CUDA driver MemOps, CUTLASS parity and full-model execution
have not run. Use both patches on the target machine. MTP/Graph remain disabled.

The current development checkout already includes both patches: do not reapply.
`snapshot.json` in patches records the exact restored tree; SHA256SUMS verifies
transport integrity. Existing Markdown whitespace warnings in patch 1 preserve
its original snapshot and do not prevent normal application.

Archive only this directory, excluding any bytecode directories if present:

```bash
tar --exclude='__pycache__' --exclude='*.pyc' -czf qwen38-flash-next.tar.gz flash-next-plan
```

For Codex dispatch → background test → restore regular Docker SGLang → ready →
wake the same session, configure the current
`sglang-gpu-switch-workflow/config.example.json` and read its README. The local
wake command must be supplied on the target machine. The current template uses
the activity watchdog wrappers; legacy docker-job templates are not the current
entry point. CPU tests cover restoration and wake ordering, including failures.


Watchdog hardening: automatic kill remains default; 60s observation plus 30s
confirmation requires unchanged raw CPU/I/O counters and zero sampled GPU.
Logs, any work, unknown metrics and sampling gaps reset observation. Partial
stdout counts as activity. Tests cover bursty work, resets, missing accounting,
confirmation recovery and default kill/warn lifecycle. Local Docker raw CPU
stats were readable, but cumulative block IO was absent; idle kill correctly
remains unavailable in that case. GPU telemetry is unvalidated.
