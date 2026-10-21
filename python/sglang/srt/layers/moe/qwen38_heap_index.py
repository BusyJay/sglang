# SPDX-License-Identifier: Apache-2.0
"""Compact, persistent GPU LFU heap state built at cache handoff boundaries."""

import torch


def build_heap_index(frequency, owners, keys, experts, max_slots):
    """Return a CPU-side packed index for one copy into a persistent GPU tensor."""
    layers = len(frequency)
    slots = len(owners)
    if not 0 < slots <= max_slots:
        raise ValueError("cache slots exceed index capacity")
    heap = [-1] * (layers * experts)
    pos = [-1] * (layers * experts)
    sizes = [0] * layers
    top = [-1] * layers
    top_pos = [-1] * layers
    free = []
    for slot, (layer, expert) in enumerate(zip(owners, keys)):
        if layer == -2:
            continue
        if expert < 0:
            free.append(slot)
        else:
            assert 0 <= layer < layers and 0 <= expert < experts
            sizes[layer] += 1
    for layer in range(layers):
        residents = [
            expert
            for slot, (owner, expert) in enumerate(zip(owners, keys))
            if owner == layer and expert >= 0
        ]
        residents.sort(key=lambda expert: (frequency[layer][expert], expert))
        # A sorted array is already a valid min heap.
        start = layer * experts
        for index, expert in enumerate(residents):
            heap[start + index] = expert
            pos[start + expert] = index
    live_layers = [layer for layer in range(layers) if sizes[layer]]
    live_layers.sort(
        key=lambda layer: (
            frequency[layer][heap[layer * experts]],
            layer,
            heap[layer * experts],
        )
    )
    for index, layer in enumerate(live_layers):
        top[index] = layer
        top_pos[layer] = index
    free_ring = free + [-1] * (max_slots - len(free))
    packed = (
        heap
        + pos
        + sizes
        + top
        + top_pos
        + [len(live_layers)]
        + free_ring
        + [0, len(free) % max_slots, len(free)]
    )
    return torch.tensor(packed, dtype=torch.int32)
