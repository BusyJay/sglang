# SPDX-License-Identifier: Apache-2.0
"""Immutable per-slot descriptors, one-expert packed transfers, direct fallback."""
from collections import Counter


def validate_packed_config(decode="direct", prefill="direct", min_bytes=4096,
                           max_bytes=64 << 20, min_copies=2):
    if decode not in ("direct", "packed") or prefill not in ("direct", "packed"):
        raise ValueError("packed_decode/packed_prefill must be direct or packed")
    if any(type(n) is not int or n < 1 for n in (min_bytes, max_bytes, min_copies)) or min_bytes > max_bytes:
        raise ValueError("invalid packed byte/copy thresholds")


def page_layout(sizes, page_bytes):
    """Precompute (tensor index, source row offset, page offset, count) segments."""
    if type(page_bytes) is not int or page_bytes <= 0 or any(type(s) is not int or s <= 0 for s in sizes):
        raise ValueError("invalid packed layout")
    pages, page, used = [], [], 0
    for index, size in enumerate(sizes):
        offset = 0
        while offset < size:
            count = min(size - offset, page_bytes - used)
            page.append((index, offset, used, count))
            used += count
            offset += count
            if used == page_bytes:
                pages.append(tuple(page)); page, used = [], 0
    if page:
        pages.append(tuple(page))
    return tuple(pages)


class PackedExpertTransport:
    def __init__(self, runtime):
        import torch
        self.runtime = runtime
        self.torch = torch
        self.scratch = torch.empty(runtime.ring[0].numel(), dtype=torch.uint8, device=runtime.device)
        self.layouts = {}
        self.tables = {}
        self.counters = Counter({"descriptor_h2d_calls_runtime": 0})

    def refresh_targets(self):
        """Called only at drained startup/migration, before publishing new targets."""
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS
        self.tables.clear()
        r, torch = self.runtime, self.torch
        for kind, targets, names in (("decode", r.caches, r.decode_copy_names),
                                     ("prefill", {0: r.arena} if r.arena is not None else {}, REQUIRED_TENSORS)):
            mode = r.packed_decode if kind == "decode" else r.packed_prefill
            sizes = tuple(r.manifest.tensors[0][name].nbytes // r.manifest.num_experts for name in names)
            layout = page_layout(sizes, self.scratch.numel())
            self.layouts[kind] = (tuple(names), sizes, layout)
            if mode == "direct":
                continue
            for layer, tensors in targets.items():
                rows = next(iter(tensors.values())).shape[0]
                if any(not tensors[name].is_contiguous() or tensors[name][0].numel() * tensors[name].element_size() != size
                       for name, size in zip(names, sizes)):
                    raise ValueError("packed targets must be contiguous matching manifest rows")
                pointers = tuple(tensors[name].data_ptr() for name in names)
                # One GPU allocation/H2D per layer table, never per job or row.
                self.counters["descriptor_table_uploads"] += 1
                table = torch.tensor([
                    [[pointers[i] + slot * sizes[i] + offset, packed, count]
                     for page in layout for i, offset, packed, count in page]
                    for slot in range(rows)], dtype=torch.int64, device=r.device)
                for slot in range(rows):
                    offset = 0
                    parts = []
                    for page in layout:
                        parts.append(table[slot, offset:offset + len(page)])
                        offset += len(page)
                    self.tables[kind, layer, slot] = tuple(parts)
        r.transfer.wait_stream(torch.cuda.current_stream(r.device))
        # Descriptor publication/old table retirement is outside serving jobs.
        torch.cuda.current_stream(r.device).synchronize()
        self.counters["target_refreshes"] += 1

    def eligible(self, kind):
        r = self.runtime
        _, sizes, pages = self.layouts[kind]
        total = sum(sizes)
        # Fragmented pages only; contiguous large copies keep direct DMA.
        return (r.packed_min_bytes <= total <= r.packed_max_bytes and
                any(len(page) >= r.packed_min_copies for page in pages))

    def fill(self, layer, expert, slot, kind):
        """One expert only. Same transfer stream protects scratch/descriptor reuse."""
        from sglang.kernels.ops.moe.qwen38_expert_cache import packed_scatter
        r, torch = self.runtime, self.torch
        names, sizes, pages = self.layouts[kind]
        descriptors = self.tables[kind, layer if kind == "decode" else 0, slot]
        views = []
        try:
            for name in names:
                views.append(r.pack.expert_tensor_bytes(layer, expert, name))
            for page, descriptor in zip(pages, descriptors):
                index = r.ring_index
                event = r.ring_events[index]
                if event is not None:
                    event.synchronize()  # pinned host bytes may still be in DMA
                host = memoryview(r.ring[index].numpy()).cast("B")
                try:
                    for tensor, offset, packed, count in page:
                        part = views[tensor][offset:offset + count]
                        try:
                            host[packed:packed + count] = part
                        finally:
                            part.release()
                finally:
                    host.release()
                used = page[-1][2] + page[-1][3]
                with torch.cuda.stream(r.transfer):
                    try:
                        self.scratch[:used].copy_(r.ring[index][:used], non_blocking=True)
                        r.transferred_bytes += used
                        r.h2d_copy_calls += 1
                        packed_scatter(self.scratch, descriptor)
                        self.counters["scatter_calls"] += 1
                        self.counters["payload_bytes"] += used
                        self.counters["direct_copy_calls_saved"] += len(page) - 1
                    finally:
                        # Even a failed scatter/partial DMA must protect ring bytes.
                        if event is None:
                            event = torch.cuda.Event(blocking=True)
                            r.ring_events[index] = event
                        event.record(r.transfer)
                        r.staging_batches += 1
                        r.ring_index = (index + 1) % len(r.ring)
            self.counters[kind + "_experts"] += 1
        finally:
            for view in views:
                view.release()
