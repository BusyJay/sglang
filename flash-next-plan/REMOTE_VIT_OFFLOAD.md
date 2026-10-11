# Flash-Next offload with ViT on a separate GPU (EPD)

This replaces the incorrect same-GPU resident-ViT patch. Do not apply the earlier
`02-flash-next-vision.patch` package. Base remains HEAD
`fa36530a1dc8700269af4312362930ebe893f2b7`; apply sampler patch 1 then remote-ViT patch 2.

## Topology and behavior

- Encoder process/GPU: existing Qwen4Exp ViT, normal loader, `--encoder-only`.
- Language process/GPU: `qwen38_nvfp4_cache`, `--language-only`, remote encoder
  URLs. No visual tower allocation and no visual weight reads. Expert and PLE
  offload remain as before.
- The existing encoder transport supplies precomputed visual embeddings; the
  language worker retains image/video token metadata, mRoPE and the normal
  multimodal embedding merge. `--language-model-only` is **not** used in EPD.

The exp 27B patch's design is adapted to Qwen4Exp: EPD architecture allowlist,
ModelConfig skip-local-encoder state, inherited Qwen3VL construction guard,
Qwen4Exp deepstack metadata/weight guard, offload startup and loader invariants.
Static encoder URLs are optional in direct serving: existing dynamic encoder
registration is supported when adaptive local dispatch is disabled. No local
ViT fallback or custom CUDA P2P transport is introduced.

## Launch

On the separate encoder GPU, use the normal existing encoder deployment. For a
new encoder, a schematic invocation is:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m sglang.launch_server \
  --model-path /absolute/full-local-checkpoint --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler --host 0.0.0.0 --port 31000
```

On the language GPU retain the tested model/pack/cache/memory flags and add:

```bash
--language-only --encoder-urls http://encoder-host:31000 \
--encoder-transfer-backend zmq_to_scheduler
```

Remove `--language-model-only`. Do not pass `enable_vision` in loader JSON (that
incorrect same-GPU option has been removed). Do not use the offload loader/expert
pack on the encoder. Keep TP/DP/EP/PP=1 and speculation disabled on language GPU.
The encoder and language workers must use matching model/tokenizer/processor
metadata and revision. ViT weights live in the encoder checkpoint; an expert
pack alone is not a vision checkpoint.

Existing encoder backend choices remain; start with ZMQ for a simple network
path, then evaluate your already configured transport. This is not P/D KV
disaggregation: language prefill and decode stay together. No running encoder is
started/stopped by the canary, and it never downloads weights.

## Image canary and Docker

The canary adds `--encoder-urls`, `--encoder-transfer-backend`, `--image`,
`--image-prompt`, `--image-answer-pattern` and `--image-max-new-tokens`:

```bash
python3 tools/qwen38_cache_tests/full_model_canary.py \
  --checkpoint /absolute/local-checkpoint --manifest /absolute/pack/manifest.json \
  --output /absolute/results/epd --runtime-mode threaded \
  --encoder-urls http://encoder-host:31000 \
  --contexts 128 --context-length 4096 --chunk 512 \
  --image /absolute/local-red-square.png \
  --image-prompt 'What color is the square? Reply with the color.' \
  --image-answer-pattern '(?i)red'
```

It sends a local image as a base64 OpenAI chat request and checks a nonempty
visible answer/optional regex. First test eagerly, then with `--decode-graph`;
repeat with another image/resolution and ordinary arithmetic/text probes.
Text startup warmup remains text-only and does not warm the encoder ViT.

`docker_model.sh` defaults to `--network none`, so **its default configuration
cannot reach an encoder URL**. Use a target-machine container configuration with
working encoder connectivity (or run the canary in the existing connected
container). Mount the local image and pass its container-visible path. Do not
confuse Hugging Face OFFLINE with disabling the EPD service network.

## Encoder-only loading and memory

Qwen4Exp encoder-only construction allocates the visual tower, not language
layers, experts, PLE, LM head, language KV cache or decode graphs. The current
vision configuration has approximately 449M parameters: BF16 weights alone are
about 0.84 GiB on TP=1, before CUDA context, activations and allocator reserves.
The `flashinfer_cudnn` vision backend also uses a shared 128 MiB workspace.
ViT CUDA graphs remain disabled by default.

For local safetensors checkpoints the default loader selects visual shards from
the index before tensor reads. Without an index it scans headers only. Mixed
shards load only `model.visual.*` / `visual.*` tensors. This Qwen4Exp encoder-only
path deliberately uses selective CPU mmap even if fastsafetensors, multithreaded
loading, disable-mmap or whole-checkpoint prefetch options were requested; it
does not GPU-stage or prefetch entire mixed shards. MTP shards are not added.
Missing visual shards, absent visual weights or incompletely loaded visual
parameters fail startup. Non-safetensors checkpoint formats retain their existing
loading behavior and do not get selective-before-read savings.

The encoder embedding cache defaults to a 4096 MiB *limit*, not preallocation.
With `zmq_to_scheduler`, embeddings are moved to CPU before caching. With
`mooncake`, normal-request embeddings can stay on GPU and grow to that limit.
For a low-memory Mooncake deployment set `SGLANG_VLM_CACHE_SIZE_MB=0` on the
encoder to disable admission, or choose a smaller cache budget. This does not
eliminate in-flight embeddings or attention/activation peaks; bound resolution,
image count and concurrency according to target GPU measurements.

## Validation and limitations

CPU tests execute production startup/visual construction/raw-feature guards with
mocks, check no local ViT allocation, and exercise canary remote-URL/image request
wiring. Existing offload iterator continues to skip visual tensors before reads.
Encoder CPU tests also verify index/header selection, no language tensor reads
from mixed shards, loader mode routing and visual completeness failures.
Local CPU regression: 120 tests, 106 passed, 14 skipped (no PyTorch).

This host has no CUDA/weights and no encoder service. Actual EPD transport,
Qwen4Exp encoder load, image accuracy and remote-image prefill followed by decode
graph replay are **not GPU-validated**. Prior user GPU passes predate this patch.
Verify both GPU memory footprints, encoder receipt/forward logs, no language-GPU
visual parameters, correct image answers and timeout/error propagation on target.
Video uses the inherited processor/transport but is not covered by the image
canary. No MTP changes are included.
