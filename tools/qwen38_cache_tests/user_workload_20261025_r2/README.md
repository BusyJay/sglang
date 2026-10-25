# Hermes workload: cached-FC1 overlap enabled

This directory contains the second-round diagnosis of runtime commit
`9e1daf70469a6091de8ca9a508ab63cc5ca7c619`. Start with [REPORT.md](REPORT.md).
The existing Hermes → LLM / remote-ViT deployment was reused. The LLM runs on an
RTX 5090 D; the remote ViT runs on an RTX 4080 and was idle in this workload.

The deployment enables `model_loader_extra_config.decode_early_cache` for the
W4A16 preset. [deployment.patch](deployment.patch) records the exact installed
launcher change; [launcher-validation.json](launcher-validation.json) records
shell syntax, W4A16/A4 preset checks and byte-for-byte patch reproduction.
The loader option was absent in the baseline, where its effective default was
`false`. The option is explicitly `true` in the second run.

This is an observational comparison. Prompt length, generation, routing,
adaptive cache layout and random seed differ. The first capture also ran an
uncalibrated Python stack sampler; the second omitted it. A throughput increase
must not be read as the feature's isolated speedup.

## Evidence

- [effective-config.json](effective-config.json): model, loader, serving and
  graph configuration, source commit and startup state.
- [config-delta.json](config-delta.json): raw loader-option delta (`null` means
  the baseline omitted the key).
- [log-comparison.json](log-comparison.json): first-request log intervals and
  the small baseline summary needed for comparison.
- [workload-server.log](workload-server.log): second-round server log, including
  native early-cache invocation counters and adaptive-cache events.
- [full-metrics.json](full-metrics.json): metrics streamed from the full trace.
- [trace-manifest.json](trace-manifest.json): trace locations, SHA-256 hashes,
  capture settings, sampled windows and analysis provenance.
- [prefill-representative.trace.json.gz](prefill-representative.trace.json.gz)
  and [decode-representative.trace.json.gz](decode-representative.trace.json.gz):
  bounded, independently inspectable Chrome/Perfetto traces.
- [rebalance-transition.trace.json.gz](rebalance-transition.trace.json.gz),
  [transition-summary.json](transition-summary.json) and
  [cold-load-evidence.json](cold-load-evidence.json): the long host-side cold
  transition and subsequent Phase1/Phase2 kernel loads.
- [prefill-triage.md](prefill-triage.md) and [decode-triage.md](decode-triage.md):
  unedited three-table output of the repository's profiler triage tool. Its
  generic fusion labels require the source checks described in the report.

The full capture is indexed by its manifest rather than embedded in Git. The
first-round report and raw evidence remain outside this commit.

## Collection

1. Apply the deployment patch to the launchers, restart `homelab-llm`, wait for
   `/ready` and validate generation. Keep the existing Hermes endpoint and ViT.
2. Verify the effective loader option through `/server_info`. Compare its parsed
   `model_loader_extra_config` against the baseline; this run added only
   `decode_early_cache: true` to that object.
3. Start CPU/GPU profiling using [start-request.json](start-request.json):

   ```bash
   curl -sS http://127.0.0.1:8329/start_profile \
     -H 'Content-Type: application/json' --data-binary @start-request.json
   python3 telemetry.py
   ```

4. Send the real workload through Hermes. Once the complete response finishes:

   ```bash
   touch stop-telemetry
   curl -sS -X POST http://127.0.0.1:8329/stop_profile \
     -H 'Content-Type: application/json' --data '{}'
   ```

   Export can take several minutes. Wait for completion before analysis.
   `telemetry.py` is an optional one-second host/GPU recorder for this Docker
   topology. Remove an old `stop-telemetry` marker before starting another run.

## Recompute

Use Python with `ijson` and `numpy`; the exact versions used here are recorded
in [analysis-environment.json](analysis-environment.json).

```bash
python3 analyze_capture.py --trace /path/to/full.trace.json.gz --output-dir /tmp/r2-analysis
python3 summarize_logs.py --log workload-server.log --output /tmp/r2-log-summary.json
python3 analyze_capture.py --trace /path/to/baseline.trace.json.gz --output-dir /tmp/baseline-overlap --overlap-only
```

If the baseline server log is available, pass `--baseline-log /path/to/log` to
recompute the comparison. Throughput summaries use the first request only and
drop its first decode interval, which contains prefill / preceding idle time.
Cycle duration estimated from logs uses rounded acceptance and throughput
values; the trace's VERIFY start intervals are separate measurements.

The analyzer processes the full trace twice without retaining CPU op objects.
It groups GPU forward annotations by the CPU forward's external ID, retaining
all GPU streams. Macro prefill regions end at the next VERIFY; draft, sampling
and graph work outside the target forward remain in the decode phase. Events
are classified by their start timestamp. `forward_window_seconds` is a union
of matched forward annotations, not total pipeline wall time.

GPU activity shares include kernels, memcpy and memset. Their cumulative
durations overlap and must not be added as elapsed time. Device-busy time is
the interval union. `fill_overlap` measures simultaneous execution of fill and
any kernel on another stream; it is evidence of concurrency, not causal latency
savings or proof that a dependency can safely be overlapped.

Representative defaults are the first 500 ms of the second target prefill
chunk and five VERIFY start intervals beginning at index 200. All complete
events whose start falls in a window are retained; an event can end outside
the window. Long parent annotations remain contextual and do not make a
bounded slice a complete forward. Sampling affects only the small traces;
`full-metrics.json` always uses the entire capture.
