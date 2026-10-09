# Script and documentation review — 2026-10-09

Current entry points: README-ARCHIVE.md, RUN_ON_GPU.md and
sglang-gpu-switch-workflow/README.md. Historical design notes are labelled;
old development-kit runners exit with current instructions.

Fixed:

- Missing block-I/O data is unknown, so it cannot trigger idle termination.
- Failed Docker cleanup makes a successful test exit nonzero.
- SIGTERM/SIGINT clean the owned test container and save an interrupted report.
- Queue dispatch is locked and saves an immutable configuration snapshot;
  preflight commands also run before execution, before stopping the service.
- Manual recovery excludes active work, removes only the persisted unique test
  container, retains identity after recovery failure and clears stale queue
  records only after successful restoration.
- Daemon stop rejects pending/running jobs. SIGKILL/reboot require manual recovery.
- Canary waits for /ready, logs phase transitions, validates positive limits and
  preserves JSON reporting even if shutdown fails.
- Runtime/converter descriptions and archive/setup commands match current code.

Evidence: 57 CPU tests passed in the locally available v0.5.21 Docker image;
SGLang imports and real ModelOpt config/runner wiring checks passed. Focused
supervisor/workflow checks passed after final recovery changes. Real Docker
short-job logging/cleanup passed. Python compilation, shell syntax, focused
Ruff E9/F/I checks and patch restoration checks passed.

Not exercised: GPU telemetry/kernels/copies/driver MemOps/CUTLASS parity,
checkpoint conversion, full-model generation, real regular-service restoration
and the target machine's same-session wake executable. The activity watchdog
is a heuristic: CPU/GPU busy loops and unknown metrics need an explicit deadline
or manual diagnosis. Regular-service recovery uses a health timeout (300s in
example), without the test-container activity watchdog. The double-fork daemon
is independent of a Codex turn but does not restart itself after host reboot.


Watchdog hardening: automatic kill remains default; 60s observation plus 30s
confirmation requires unchanged raw CPU/I/O counters and zero sampled GPU.
Logs, any work, unknown metrics and sampling gaps reset observation. Partial
stdout counts as activity. Tests cover bursty work, resets, missing accounting,
confirmation recovery and default kill/warn lifecycle. Local Docker raw CPU
stats were readable, but cumulative block IO was absent; idle kill correctly
remains unavailable in that case. GPU telemetry is unvalidated.
