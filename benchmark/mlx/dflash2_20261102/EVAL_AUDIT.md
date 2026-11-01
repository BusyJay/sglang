# Verification evaluation audit, 2026-11-02

Hardware: remote M5 Ultra256GiB, MLX0.32.4, original mixed quantized GLM
checkpoint fa04cbc3622b16aa7475404060b05e506d81bd4e, DFlash2 BF16
bf582e4eacc1810f76656d1811693ff6c6737d2a. No nativeMTP weights.

Changing verification group4->64 changes compile partitions and submission.
A second experiment fixes compile partitions at4 and operator input shapes,
coalesces every1/2/4/12submission while retaining all cache output roots.
7 alternating trials per4/6/8rows and32/2050context; first2excluded.
Every target output, capture and per-position cache snapshot exact.

| Context | Rows | Every4layers | One48layer submission |
|---|---|---|---|
|32|6|49.484ms|56.029ms|
|2050|6|56.228ms|64.514ms|
|32|8|58.200ms|66.259ms|
|2050|8|68.979ms|78.521ms|

These are complete target verification timings, not fullgeneration throughput.
For short6rows async call host total36.412->46.918ms, overall graph call
44.340->52.885ms. Host time includes native waiting, not pure CPU compute.

Three-second installed-process /usr/bin/sample captures at longcontext6rows
show main-thread wait inside async_eval->eval_impl->condition_variable::wait:
605/1843samples(group4),965/1967samples(one48layers). This is direct evidence
of native scheduling wait; kernel-dispatch geometry not established as cause.
Archive MLX transforms.cpp has task/memory backpressure wait, but archive is
not asserted as source identity of installed library. Need command-buffer
batch experiments and memory/task disambiguation before finalcause claim.

Production candidates: auxiliary store_cache submits views across all layers in
one async_eval; proposal tokens read once perchain instead of perrow item. Device
checks dense7/aux6/DFlash8 pass on functional source; normal-generation host AB complete and exact; final formatted source
HTTP D3/D5 complete and exact, strict final checks and manifest passed.

No systemwired-memory parameters changed. Existing process wiredlimit and
cachelimit inherited from ordinary comparisons. No third-party installs.

Snapshot async sealing experiment completed: layers4/48,11alternatingtrials,
first2excluded; source arrays overwritten immediately after async submission,
restored per-layer state and DFlash feature tree exact. Small synthetic48layer
submit median sync.5663ms->async.4554ms,total.7347->.6181ms. This is cache
microbenchmark, not finaltargetthroughput. Production store_cache now submits
all immutable views once asynchronously; standalone _snapshot_cache still
seals synchronously for callers requiring its own completed snapshot.
Final functional-source devicechecks auxiliary6+DFlash8+GLM63pass.

Scalar proposal readback->onceperchain vector readback normalAB24pairedcases
exact tokens/stats. Short D3-.19%,D5-.01%,D7+.63%;long-.49/+.12/+.13%.
Within noise, no material throughput claim. Retains simpler fewer readbacks.

Commandbuffer threeprocessdiagnostic ops200,MB50/1024/8192 all reproduced
single-submit slowdown; knobchange did not remove it. Need keep remaining
cause as scheduling wait with unresolved task-vs-memory attribution. Do not
claim tuning these knobs has fixed cost. Production knobs remain unchanged.

Remaining explicit eval sites: preparation/load-only weights; CPU logits hooks
require materialized values; chunked reconstruction after missing auxiliary
prefix snapshot bounds large lazy graphs; standalone snapshot helper seals
its own result. Normal DFlash verification already uses async_eval at layer
group boundaries. No blanket function-name replacement adopted.


Final serving validation: D3 stream67.511tok/s vs D5 66.592tok/s; concurrent
73.705/65.292.17cases+concurrent2 signatures exact vs AR. Final system swap
grew1229.81->1467.69MiB(D3),1467.69->1624.06MiB(D5); unresolved attribution.
Do not infer cache microbenchmark gains translate into overall HTTP speedup.
All final21source hashes matched remote, formats/lint/registrations passed.
