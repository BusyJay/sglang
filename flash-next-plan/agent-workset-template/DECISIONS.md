# Design Decisions

Last updated: 2026-10-09

Use stable IDs. Never silently rewrite an accepted decision. Add a new decision that supersedes the old one.

## D-001 — Source Baseline

Status: accepted

Decision:

<Decision text.>

Reason:

<Short reason and evidence.>

Implications:

- <Implementation implication.>

## D-TEST-001 — Daily GPU Test Device

Status: accepted

Decision:

Use RTX 4080 SUPER for daily small GPU tests. Keep the configured budget at 8GiB and every test below 10GiB. Reserve RTX 5090 for SM120-specific FlashInfer CUTLASS checks and full-model milestones.

Reason:

The 4080 SUPER already runs other services but has enough capacity for one-expert, small-MoE, cache, staging and Marlin fallback tests.

Implications:

- GPU tests must declare estimated peak memory.
- Tests skip when the required free memory is unavailable.
- Passing on SM89 does not replace SM120 backend validation.

## D-KERNEL-001 — Native Extension Path

Status: accepted

Decision:

Prefer SGLang's in-tree JIT framework under `python/sglang/kernels/jit` for new
Qwen3.8 cache/ID-remap kernels. Use `python/sglang/kernels/aot` only when an AOT
extension is required, and add an explicit container rebuild step.

Reason:

Mounted Python/JIT sources are visible immediately in the v0.5.21 development
image; installed AOT extensions are not replaced merely by `PYTHONPATH`.

## D-002 — Example Superseding Decision

Status: proposed

Supersedes: `<decision ID or none>`

Decision:

<Decision text.>

Reason:

<Reason.>
