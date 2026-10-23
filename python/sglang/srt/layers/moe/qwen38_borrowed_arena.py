# SPDX-License-Identifier: Apache-2.0
"""Exclusive prefill lease of the expert pool tail; metadata stays unchanged."""


def tail_overlaps(layout, rows):
    budget = sum(layout)
    if rows <= 0 or rows > budget:
        raise ValueError("arena must fit in the expert pool")
    cursor = 0
    result = {}
    for layer, slots in enumerate(layout):
        start = max(0, budget - rows - cursor)
        if start < slots:
            result[layer] = (start, slots)
        cursor += slots
    return result


class BorrowedArenaMixin:
    def _drain_arena_readers(self):
        # Caller owns the runtime's non-reentrant lock. Do not call the public
        # synchronize(), whose base implementation acquires that same lock.
        self.jobs.join()
        if self.miss_mailbox is not None:
            self.miss_mailbox.synchronize()
        self.metadata_stream.synchronize()
        self.transfer.synchronize()
        for event in self.layer_computed.values():
            event.synchronize()
        if self.arena_last_use is not None:
            self.arena_last_use.synchronize()

    def _borrow_arena(self):
        if not self.borrowed_arena or self.borrowed_records is not None:
            return
        # No pending planner or reader may race the snapshot or tail overwrite.
        self._drain_arena_readers()
        self.borrowed_layers = tail_overlaps(
            self.layer_cache_slots, self.manifest.num_experts
        )
        self.borrowed_records = {
            layer: [
                (slot, expert)
                for slot, expert in enumerate(
                    self.device_states[layer]["keys"].cpu().tolist()
                )
                if slot >= start and expert >= 0
            ]
            for layer, (start, _) in self.borrowed_layers.items()
        }

    def _return_arena(self):
        if self.borrowed_records is None:
            return
        self._drain_arena_readers()
        try:
            for layer, records in self.borrowed_records.items():
                for slot, expert in records:
                    # Includes slot alphas; the original lookup/policy is retained.
                    self._fill(layer, expert, self.caches[layer], slot)
                    self.borrowed_restore_experts += 1
                banks = getattr(self, "csf_banks", None)
                if records and banks is not None:
                    import torch

                    # CPU source headers contain file offsets, not GPU pointers.
                    # Restored slots keep their hit metadata, so decode will not
                    # run the miss-fill kernel that normally binds these words.
                    with torch.cuda.stream(self.transfer):
                        banks.patch(layer, self.caches[layer])
            self.transfer.synchronize()
        except BaseException:
            self.poisoned = True
            raise
        self.borrowed_records = None
        self.borrowed_layers = {}
        self.arena_layer = None
        self.arena_residents = set()
        self.arena_last_use = None
        self.borrowed_restore_runs += 1
