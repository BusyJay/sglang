# Selected configuration profile

Framework: SGLang. Model: `nvidia/Qwen3.8-Flash-Next-FP4-Local`.
Runtime arguments are in `server.json`; selection is prepared=0, CTA=8,
B12X W4A16, MTP=2, 6390 expert slots, breakable decode graphs.

- Raw single-rank trace: `selected-decode-1792318828.477386-TP-0.trace.json.gz`.
- Kernel, overlap-opportunity and fuse-pattern tables: `triage.txt`.
- Overlap-aware interval unions and operator groups: `analysis.json`.
- Capture protocol and initial control-client parsing issue: `capture.json`.

Single-trace triage; no graph-off mapping/formal pair. The GPU window is
30.96 ms/verify under with-stack profiling. Mapped-host expert fill occupies
11.88 ms (38.36%); B12X compute occupies 2.86 ms. Profiled timings include
instrumentation overhead and are excluded from service throughput results.

Automatic fusion-family recognition is not proof that the listed generic
implementations can replace this Qwen4 cache/B12X execution path.
Similarity note: low
