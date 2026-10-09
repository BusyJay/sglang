# Implementation Plan

Last updated: 2026-10-09

## Milestones

### M0 — Baseline

- [ ] Pin source and dependency versions.
- [ ] Record baseline commands and memory use.

### M1 — Runtime Expert Pack

- [ ] Build pack format.
- [ ] Validate source-to-runtime conversion.

### M2 — Dedicated Loader

- [ ] Skip routed expert checkpoint tensors.
- [ ] Load dense/shared/MTP/PLE paths.

### M3 — Decode Cache

- [ ] Implement synchronous cache path.
- [ ] Verify fixed routing parity.

### M4 — Prefill Arena

- [ ] Implement full-layer expert arena.
- [ ] Run 8K and 32K prefill.

### M5 — Fast Path

- [ ] Remove per-layer Python GPU synchronization.
- [ ] Add transfer stream and event protection.

### M6 — Long Context

- [ ] Validate FP8 KV.
- [ ] Run 128K and 220K tests.

### M7 — MTP

- [ ] Enable NEXTN.
- [ ] Protect verify-window expert slots.

