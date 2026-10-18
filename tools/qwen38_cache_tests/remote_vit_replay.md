# Remote ViT replay fix

Hermes resends the complete conversation, including old inline images. The
previous tokenizer dispatch submitted those images to the remote encoder again
before scheduler KV prefix matching.

The fix defers immutable inline image dispatch to the scheduler. A bounded CPU
LRU retains received CPU embeddings and encoder metadata. On a hit, the processor
rebuilds input IDs, image offsets and mRoPE for the current prompt, and normal KV
prefix matching follows. No /encode or receive registration is submitted on a
hit. A miss submits remote encoding and receives its results through the existing
transport. All TP ranks must agree on a hit and successful reconstruction.

This is embedding/metadata reuse ahead of KV prefix matching, not a query against
the KV radix tree. KV and embedding cache eviction are independent: after KV
is evicted, cached embeddings can still avoid encoding but prompt prefill runs
again. After the CPU embedding entry is evicted, the image is encoded again even
if some KV survives. Exact KV-only lazy encoding would additionally require a
separate long-lived image-layout metadata cache and scheduler changes.

Scope: default HTTP receiver, zmq_to_scheduler, image-only immutable data URLs,
unpooled CPU receive. Mutable HTTP/file URLs, videos/audio, mixed modalities,
Mooncake, gRPC, zmq_to_tokenizer and an explicitly enabled GPU embedding pool
retain their previous dispatch behavior. Image list order, content, preprocessing
options and cache_salt are part of the key. Client-provided content hashes do not
replace actual inline content identity. There is no partial image match handling.
A new image added to the image list currently misses the whole-list cache.

No new resident GPU memory: cached tensors must be CPU tensors and the cache
retains the original CPU receive results. Capacity uses the existing
SGLANG_VLM_CACHE_SIZE_MB setting (default 100 MiB per scheduler process).
Setting that existing capacity to zero disables deferred replay dispatch.
No GPU pool is introduced and ordinary per-request forward allocations remain.
For unpooled language-only `zmq_to_scheduler` serving, this CPU capacity does not
enable the separate GPU embedding cache or deduct its size from the GPU KV
budget. Explicit GPU receive pools keep their existing embedding-cache behavior.

CPU tests cover content/settings/namespace identity, mutable URL bypass, byte
bounds, GPU tensor rejection, current-prompt reconstruction and metadata isolation.
HTTP/TP integration and end-to-end latency have not been run locally.

GPU/integration acceptance:

1. Send a data-URL image in turn 1, wait for completion, resend it with the full
   conversation in turn 2. Expect one initial /encode, then the CPU replay log.
2. Verify answers and prompt layout against replay disabled; append text before
   and after the image, checking offsets/mRoPE belong to the new prompt.
3. Change image bytes, detail, order, and cache_salt: expect a miss.
4. Force CPU cache eviction: expect re-encode. Evict KV alone: expect replay
   embedding reuse with normal prompt recomputation.
5. Repeat with TP ranks disagreeing on CPU cache contents: all ranks encode.
6. Exercise encoder errors, cancellation, receive registration before encode,
   and repeated distinct request IDs. Confirm no receive resource leak.
7. Measure resident/peak GPU memory with replay on/off; new cache must remain CPU.
