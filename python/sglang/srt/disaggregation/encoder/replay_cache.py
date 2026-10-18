# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU-only reuse of immutable remote image encoder results."""

import hashlib
import json
from collections import OrderedDict


def gpu_embedding_cache_size_mb():
    """Remote CPU replay must not also enable the GPU embedding cache."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg

    disagg = get_disagg()
    if (
        disagg.language_only
        and disagg.encoder_transfer_backend == "zmq_to_scheduler"
        and not (
            envs.SGLANG_EMBEDDING_POOL_SIZE_MB.is_set()
            and envs.SGLANG_EMBEDDING_POOL_SIZE_MB.get() > 0
        )
    ):
        return 0
    return envs.SGLANG_VLM_CACHE_SIZE_MB.get()


def replay_key(items, cache_salt=None):
    # Mutable HTTP/file URLs must be reloaded: URL equality is not content
    # equality. Inline data URLs carry the complete immutable image bytes.
    if not items or any(
        getattr(item.get("modality"), "name", None) != "IMAGE"
        or not isinstance(item.get("url"), str)
        or not item["url"].startswith("data:image/")
        for item in items
    ):
        return None
    payload = [
        {k: v for k, v in item.items() if k not in ("modality", "content_hash")}
        for item in items
    ]
    try:
        encoded = json.dumps(
            [cache_salt, payload], sort_keys=True, separators=(",", ":")
        ).encode()
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


class EncoderReplayCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, max_bytes)
        self.bytes = 0
        self.entries = OrderedDict()

    def get(self, key):
        record = self.entries.get(key)
        if record is not None:
            self.entries.move_to_end(key)
            return record[0], record[1]
        return None

    def remove(self, key):
        record = self.entries.pop(key, None)
        if record is not None:
            self.bytes -= record[2]

    def put(self, key, embeddings, metadata):
        if key is None or self.max_bytes == 0:
            return
        # Never retain GPU receive buffers or allocate a GPU cache copy.
        tensors = list(embeddings.values()) + [
            v for v in metadata.values() if hasattr(v, "device")
        ]
        if any(t.device.type != "cpu" for t in tensors):
            return
        size = sum(t.untyped_storage().nbytes() for t in tensors)
        size += len(
            repr(
                {k: v for k, v in metadata.items() if not hasattr(v, "device")}
            ).encode()
        )
        if size > self.max_bytes:
            return
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[2]
        while self.entries and self.bytes + size > self.max_bytes:
            self.bytes -= self.entries.popitem(last=False)[1][2]
        self.entries[key] = (embeddings, metadata, size)
        self.bytes += size
