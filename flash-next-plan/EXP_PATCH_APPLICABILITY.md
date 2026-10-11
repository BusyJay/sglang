# exp memory-saving patches: Flash-Next applicability

Base: `fa36530a1dc8700269af4312362930ebe893f2b7` (tested decode graph/startup
warmup integration). This adaptation targets the existing
`qwen38_nvfp4_cache` deployment, not the Qwen3.8 27B Qwen3.5 deployment.

## HEAD comparison

The merged HEAD retains the submitted expert eager boundary/capture stub and
adds two GPU integration fixes: structured `LogitsProcessorOutput` buffer
allocation/copy/slicing in the breakable backend, and explicit capture-stream
contexts around segment begin/end. Both are preserved unchanged here. GPU test
success was reported by the user; this local environment cannot reproduce it.

## Three patches

| exp commit | Purpose | Flash-Next offload decision |
| --- | --- | --- |
| `478c2c50ff` | Skip sampled-token TP all-reduce for a single rank | Adapted: useful for grammar/forced token sync; avoids a needless NCCL collective and its lazy resources. |
| `b265c969e3` | Skip local vision encoder in language-only EPD | Already achieved for this deployment: offload forces `language_model_only=True`, inherited Qwen3VL constructor sets `visual=None`, and the loader filters visual tensors before `get_tensor()`. No EPD flag/model rewrite added. |
| `a9ee187bad` | Avoid duplicate MTP embedding/lm_head allocation | Not active here: offload rejects speculation and creates no draft worker. Changing Qwen3.5 draft allocation does not save this deployment's VRAM. No MTP code changed. |

The 27B patch's reported ~0.86 GB vision and 5.53→0.80 GB draft footprints must
not be presented as Flash-Next savings. Its metadata identifies
`Qwen4ExpForConditionalGeneration`, with 48 layers, 512 experts and PLE/HC; it is
not the Qwen3.5 27B architecture even though some backbone classes are shared.

## Sampler adaptation

The current sampler also uses `tp_sync_group` for sampling-mask status reduction.
The older exp patch changed this field to a GroupCoordinator; copying it as-is
would break those process-group consumers. Instead the field stays the device
process group, and `tp_sync_world_size` stores the selected coordinator width.
Under DP attention this is **attention TP**, not global TP. Single-rank or absent
device-group token synchronization returns early; multi-rank MIN reduction and
sampling-mask/CP status synchronization remain unchanged.

No switch is required, and no output/token logic changes. This removes only the
sampler-triggered single-rank collective; other subsystems can still legitimately
initialize NCCL/P2P, so it is not a promise of zero NCCL logs or a measured VRAM
saving. Plain non-grammar requests with token sync disabled already skipped it.

CPU protocol tests execute the production constructor group-selection prefix and
token-sync method with stand-ins, covering grammar and forced sync, single/multi
rank, absent device group and DP-attention group selection. No weights download.
On GPU compare a first Hermes strict tool/JSON request and ordinary text request;
check visible answers/tool outputs and whether sampler-induced NCCL initialization
is gone. Existing synthetic GPU and full-model canaries should remain passing.

Local validation: 107 CPU tests, 93 passed and 14 skipped because PyTorch is
unavailable; the four new sampler tests passed. Syntax and diff checks passed.
No GPU execution or model weights were used for this adaptation.

## If deployment changes later

If Flash-Next offload is extended to EPD with vision inputs, retain multimodal
metadata and accept remote embeddings while excluding local visual construction;
`language_model_only` must not be blindly substituted for EPD language-only.
If offload gains MTP, adapt `Qwen4ExpForCausalLMMTP` itself: it overrides the draft
constructor and still allocates `ParallelLMHead`. Preserve target quantization
metadata, vocabulary sharding, DP-LM-head groups and PP fallback, and budget its
expert weights before enabling speculation. The 27B tensor wrappers alone do not
solve that task. Neither expansion is included in this patch.
