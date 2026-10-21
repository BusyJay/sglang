# Qwen3.8 compressed scale mode

This opt-in mode stores lossless CSF scales in both the host pack and GPU expert
slots. A16 decode and A4 prefill share the same FP4 weights and bitmap/rank
scale slabs. GPU W4A16 kernels decode at scale loads; B12X W4A4 reconstructs
its scale operands inside the MoE kernel. No standalone GPU decompression
launch, full expanded scale buffer, or second weight layout exists. The CPU miss operator expands
only its current expert inside its worker pool; it owns one expert of scratch.
Prefill runs entirely on GPU and never dispatches CPU work.

Convert an existing post-transformed runtime pack (NumPy required):

```bash
PYTHONPATH=python python -m sglang.srt.model_loader.qwen38_nvfp4_pack.csf \
  --source /path/native-manifest.json \
  --payload /path/csf-experts.bin --manifest /path/csf-manifest.json
```

The converter selects row bases maximizing coverage of a 16-value byte window.
Nibbles follow the existing CUTLASS swizzle. A bitmap identifies native
four-byte words requiring replacement, and rank addresses the replacement
words. The operands match B12X `prepare_inline_scales`; restored
E4M3 bytes are exact. Every plane uses one 16-byte-aligned stride across all
layers and experts, including exception padding. A plane whose maximum stride
would exceed native storage stays native. Each expert slab has a 1024-byte zero/padding header, interleaved
128-row bases and nibble tiles, 32-byte metadata per 512 native scale bytes,
and aligned replacement words. The header records the replacement count and
codec marker at bytes 1008/1012; the first 704 bytes match B12X padding.
The runtime keeps expert-local offsets so slot migration is an opaque row copy.
Existing version-2 packs stay native; compressed packs are version 3 with
`encoding="b12x_csf_bitmap_v1"`. Earlier experimental `csf_native_v1` packs
must be regenerated offline; inference rejects them.

Configure the loader with `"scale_compression":"csf"` and the new manifest.
Keep the existing threaded/adaptive cache settings. For the same cache byte
budget as 7000 native TP1 slots, replace `cache_slots_total` with:

```json
{
  "manifest_path": "/path/csf-manifest.json",
  "scale_compression": "csf",
  "cache_budget_bytes": 19353656000,
  "runtime_mode": "threaded",
  "adaptive_cache": true
}
```

19353656000 = 7000 × (2457600 weight bytes + 307200 scale bytes + 8 alpha bytes).
The byte budget covers expert rows; metadata, activation buffers and staging
are allocated separately before KV sizing. Do not combine byte and slot budget
options. Explicit `cache_slots_total` remains a slot count; it is never silently
multiplied by 1.05. VMM loans use the compressed row sizes and physical page
alignment. Transfer counters count actual slab bytes, including padding.

Use the existing environment options:

```bash
export SGLANG_QWEN38_CACHE_W4A16=1
export SGLANG_QWEN38_CACHE_W4A16_BACKEND=b12x
export SGLANG_QWEN38_CACHE_PREFILL_W4A16=0
```

Decode/verify uses the pinned FlashInfer 0.7.2 B12X adapter on SM12x. With
`SGLANG_QWEN38_CACHE_PREFILL_W4A16=0`, prefill uses B12X NVFP4 W4A4
dynamic MoE, including cache hits and GPU arena misses. Set it to `1` to use
Triton W4A16 prefill over the very same pack. A4 requires BF16 inputs, ordinary
SwiGLU/output routing, and H/I divisible by 128. It quantizes activations
inside the B12X kernel using the pack's input global scales; the dispatcher's
optional prequant cache is not used. A single runtime-owned scratch workspace
is allocated before KV sizing and shared across serial layers; 128-token
chunks bound routed-activation padding independently of attention chunks.
Only native scale addressing is adapted for slot-strided storage; the B12X
shared operand decoder and its consumer barriers remain in the MoE kernel.
Replacement-word exceptions read the slot's original global payload.
W4A4 prebinds the vendor-selected kernel, resident grid and static scratch
pointers by current pool address, capacity, token count and ID dtype. Bindings
are capped at eight and cleared when the pool changes; they retain no expert
storage or activation tensors. Configure B12X launch-policy environment options
before creating the adapter. Environment-enabled external routing plans retain
the vendor launch path. Each invocation updates live input/output pointers and
the current stream. Runtime-owned completion events protect shared scratch
across streams, including external graph dependencies. Native and compressed
W4A4 both use this binding path. W4A16 reuses alpha views, launch scalars and
bounded raw pointer descriptors; compressed decode already enables its prepared
path, while the existing native prepared-launch option remains unchanged.
Contiguous BF16 activations, contiguous routes and FP32 router weights borrow
caller addresses. Other accepted layouts/dtypes are normalized once per call.
Outputs keep independent storage. The existing four small W4A4 alpha/global
scale refresh copies remain live after slot fills; no weight or expanded scale
copies are added.
CSF selects the direct Triton projection path instead of the optional split-K
verify kernels. GPU kernels allocate no expanded scale workspace. Existing
cache reader events and opaque row migration still apply.

CPU/GPU hybrid remains gated by its existing enable flag and calibrated cost
file. Compression changes the runtime fingerprint, so native-mode calibration
must be regenerated. Without a compatible calibration, use GPU-only routing.
The CPU library keeps ABI 2 and adds a CSF entry point; old explicit libraries
without CSF ABI 2 (`qwen38_cpu_csf_abi`) must be rebuilt. Default lazy builds hash the changed source.

Validation commands:

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/model_loader/test_qwen38_csf.py
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/moe/test_qwen38_csf_gpu.py
PYTHONPATH=python python test/registered/kernels/benchmark/moe/bench_qwen38_csf.py
```

CPU parity, byte-for-byte B12X operand compatibility, prefill launch ABI
and capacity rebinding, and SM120 offline compilation have been checked on the
development host. A4 native/CSF parity covers duplicate/masked routes, native
plane fallback, relocated/shrinking pools, and 1/3/17/129-token prefill. GPU numerical tests, benchmark timing, real-checkpoint compression ratio,
and full serving replay still require the target GPU and checkpoint. The mode
is opt-in; synthetic slot savings do not establish a serving speedup.
