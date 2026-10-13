# Startup memory diagnostic supplement (2026-10-13)

This single tar contains the complete previous final-r3 series plus independent
13-startup-memory.patch, this manual, investigation, and raw CPU/compile evidence.
Preserve previous archives. All experimental runtime defaults stay enabled:
dynamic L2 / packed decode / packed prefill / group2; L2 requires overlap-off.
This supplement is not GPU-reviewed or GPU-validated. GPU execution is not_run.

## Apply

At the implementation base from package.json apply every filename in series in
order. At final r3 af9aac635231f587268859ec3ceec0ab10b34050 apply only
13-startup-memory.patch. Other supported starts and target trees are in package.json.
Run SHA256SUMS from the extracted package directory before applying. For example:

```bash
sha256sum -c SHA256SUMS
# Run from the target source checkout; DELIVERY points to extracted tar directory.
git apply --check "$DELIVERY/13-startup-memory.patch"
git apply "$DELIVERY/13-startup-memory.patch"
```

At older starts apply the remaining entries sequentially, checking each before
applying. Source tools and docs are contained in the patches; after applying they
live under tools/qwen38_cache_tests. External checkpoint, expert pack, local image
and GPU are prerequisites, not embedded weights or an image.

## One diagnostic launch

Direct server launch: prepend
SGLANG_QWEN38_STARTUP_MEMORY_DIR=/absolute/writable/path to the existing exact
launch command. The probe activates only for the Qwen38 cached loader on CUDA.
Ordinary startup without this environment variable retains no snapshots/history
and no diagnostic synchronization. All diagnostic samples synchronize but never
add gc.collect or empty_cache; paired stages surround existing cleanup calls.
Allocator history is bounded at 200000 events; unsupported history APIs are
reported, never replaced with invented stack evidence. Full stage snapshots may
produce hundreds of files and substantial CPU/disk overhead. Run in a clean
process, not a throughput benchmark. Formal benchmark runner refuses this env.

Using the existing canary/container launcher (adapt image/checkpoint/pack/slots,
context/chunk and remote encoder arguments to the actual deployment):

```bash
tools/qwen38_cache_tests/docker_model.sh \
  --image "$SGLANG_IMAGE" --device 0 --checkpoint "$CHECKPOINT" \
  --pack "$PACK" --results "$RESULTS/adaptive-on" -- canary \
  --slots 96 --contexts 8192 --adaptive-cache \
  --startup-memory-dir /workspace/results/memory
```

One run captures the whole startup. To quantify the adaptive-only difference,
run a second fresh process with exactly the same deployment/model/pack/image/
allocator environment and arguments, omit only --adaptive-cache, and use a new
--results "$RESULTS/adaptive-off" directory. Match the user's actual context,
chunk, mem fraction, PLE/vision configuration instead of inferring it from this
example. L2/packed/group/overlap settings stay the same across this adaptive
comparison. The old all-off performance baseline remains separately explicit in
the benchmark suites. Canary report.json records diagnostic directory and labels
timing startup_diagnostic_not_benchmark. Startup logs print Qwen38StartupMemory.

Each output has memory/startup-memory-pid<PID>-<NS>/events.jsonl plus per-stage
.pickle snapshots, .segments.json and selected .inventory.json. PID separates
processes; CUDA aliases are normalized so one device has one probe/history.
Copy back the entire results directory, including server.log and report.json.
To summarize on any CPU machine after applying the patch:

```bash
python3 tools/qwen38_cache_tests/summarize_startup_memory.py \
  "$RESULTS/adaptive-on/memory/startup-memory-pid<PID>-<NS>" \
  --output "$RESULTS/adaptive-on/memory-summary.json" \
  > "$RESULTS/adaptive-on/memory-summary.txt"
```

Repeat for off. The summarizer only reads JSON. Pickles are optional native
PyTorch memory_viz artifacts: inspect only trusted snapshots in an appropriate
PyTorch tool. Segments in the summary are filtered to the sampled CUDA device.

## Stages and interpretation

- runner load begin and loader before/after model construction: constructor
  allocations, live parameter/buffer storage and alias names.
- weights begin/end before/after existing loader cleanup, iterator finish and
  checkpoint_filter_counts: early skipped MTP/rotary/routed/vision and read counts.
- per-module quantization postprocess before/after device_loading_context:
  identify repack/replacement staging that changes reserved or live memory.
- adaptive six global pool tensors before/after individually (logical bytes),
  base per-layer tensors/scales/arena; routing JIT; resident alpha tables;
  adaptive plan metadata; packed scratch/descriptors: allocation-order evidence.
- runner precompile and Load weight end before/after existing empty_cache:
  exact point of the observed ~500 MB difference.
- KV before/after gc and existing empty_cache: cleanup effect before driver-free
  budgeting, allocated token count and component construction afterward.
- shared graph output, eager warmup, prefill/decode capture, finalization and
  post-capture KV budget/finalize when that resize path is used: later consumers.

Every event has absolute values and previous-stage deltas for driver free/total,
allocated/active/reserved/inactive_split, segment/allocation counts, retries/OOMs.
The summary lists the ten largest mixed active/inactive segments and their active
block sizes, requested sizes and allocation frames. An inactive block in a
segment with a live allocation cannot be released as an entire segment by
empty_cache. Fragmentation is a hypothesis until actual on/off snapshots show it.
Reserved minus allocated includes padding and pending allocations, not only
inactive splits; inspect active_awaiting_free and allocator backend limitations.

Inventory deduplicates actual parameter/buffer backing storage, retaining aliases
and views. Runtime expert caches, packed scratch, graph pools and tensors outside
registered parameters/buffers are NOT part of model inventory; full allocator
snapshots cover PyTorch allocations and call stacks. driver-used-minus-reserved
also includes CUDA context, libraries and OTHER processes; it is not proof of a
leak. Use an idle GPU and retain external GPU process/memory observations.
History is enabled when load_model starts; earlier allocations may lack frames.
The diagnostic synchronization changes pending-free timing and startup latency;
repeat without diagnostics for actual performance/capacity acceptance afterward.

## Unused modules/weights inspection

Found prior exp/main commit a9ee187badcd83f00148227e4adb6ca60dcce54d
(Share target embed/lm_head with Qwen3.5 MTP draft). It avoids constructing draft
embedding/head that were immediately replaced by shared target tensors. Its
recorded 5.53 GB to 0.80 GB result belongs to that older code/model path, not this
new delivery. Avoiding an allocation is more reliable than clearing the cache
when later live allocations share its segments.

Current Qwen4ExpForConditionalGeneration builds the target language model and
optional visual tower; the Qwen38 hook rejects speculative MTP execution. No MTP
draft embedding/head constructor was found in this path. Previously its
load_weights skipped names containing mtp only AFTER checkpoint CPU get_tensor.
This patch matches that existing skip and rotary_inv_freq skip BEFORE get_tensor,
with regression coverage and audit counters. It avoids unnecessary CPU reads;
we do not claim this recovers 500 MB of GPU memory. Target embed_tokens/lm_head
are used and must stay. inventory exposes unexpected mtp/visual named storage.

Remote-vision mode already skips constructing the local ViT and its checkpoint
reads (historical b265c969e38069dab569c1b7a321e79df404a567). Local visual tower may
be legitimately needed; do not remove it based only on its name. PLE fp8
replacement already deletes old tensor and removes stale params_dict references
before empty_cache (qwen4_exp.py load_qwen4_exp_ple_shard). Lazy/pinned PLE uses
host storage and rejects that GPU dtype-swap path. Quantized regular modules
can allocate during postprocess; per-module snapshots are provided to investigate.

Adaptive uses six global tensors and narrow views; off uses six tensors per
layer interleaved with scale allocation. Logical expert cache budget may be
equal while segment reservation differs. Startup adaptive plan is small;
migration scratch is allocated during later rebalance, not load-weight-end.
No confirmed duplicate GPU cache or unnecessary target weights found. Actual
500 MB cause and recovered KV size remain unverified pending these GPU logs.

## Validation / return

See STARTUP_MEMORY_VALIDATION.json and raw/ for new-code tests and commands.
CPU protocol fixtures are synthetic; real CPU inventory and loader paths are
also exercised. Offline SM120 compile loads the extension but executes no kernel.
GPU probe regression is prepared in test_gpu_cache.py; full canary/on-off allocator
comparison, CUDA async/graph/KV correctness and performance remain not_run.
For GPU acceptance follow FINAL_GPU_HANDOFF.md, with diagnostics disabled.
No validated machine profile or speed/capacity improvement is included.
