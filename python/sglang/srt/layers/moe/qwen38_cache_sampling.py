# SPDX-License-Identifier: Apache-2.0
"""Collect decode routes without making expert execution wait for the CPU."""

import queue
import threading

import torch


class RouteSampleBatch(tuple):
    """A batch holds its bank credit until the analyzer consumes it."""

    def __new__(cls, fields, released):
        batch = super().__new__(cls, fields)
        batch.released = released
        return batch

    def release(self):
        self.released.set()


class DeviceRouteSampler:
    """Two banks per layer, owned until the CPU has consumed the copied rows.

    Only the foreground submits CUDA work. The collector waits for completed
    D2H batches and emits the existing (layer, cycle, index, active IDs) protocol.
    A slow observer can drop samples, but never blocks a decode fill or reuses
    a buffer still read by CUDA/the collector. The analyzer rejects gaps.
    """

    def __init__(self, runtime, abort):
        self.runtime, self.abort = runtime, abort
        self.layers = runtime.manifest.num_layers
        self.window, self.width = (
            runtime.adaptive_cache_window,
            # The fused mailbox supports up to 32 routes, including the union
            # across a multi-token verify batch. Reserve the count plus IDs.
            max(32, runtime.manifest.top_k) + 1,
        )
        shape = (2, self.layers, self.window, self.width)
        self.device = torch.empty(shape, dtype=torch.int32, device=runtime.device)
        self.host = torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=True)
        self.stream = torch.cuda.Stream(device=runtime.device)
        self.produced = [
            [torch.cuda.Event() for _ in range(self.layers)] for _ in range(2)
        ]
        self.copied = [
            [torch.cuda.Event() for _ in range(self.layers)] for _ in range(2)
        ]
        compute = torch.cuda.current_stream(runtime.device)
        for bank in range(2):
            for layer in range(self.layers):
                self.produced[bank][layer].record(compute)
                self.copied[bank][layer].record(compute)
        self.free = [[threading.Event() for _ in range(self.layers)] for _ in range(2)]
        for bank in self.free:
            for event in bank:
                event.set()
        self.current = [None] * self.layers
        self.dropped_cycle = [None] * self.layers
        self.next_bank = [0] * self.layers
        self.jobs = queue.Queue()
        self.worker = threading.Thread(
            target=self._work, name="qwen38-route-collector", daemon=True
        )
        self.worker.start()

    def begin(self, layer, sample, compute):
        if sample is None or not self.runtime.adaptive_cache:
            return 0
        cycle, index = sample
        if self.dropped_cycle[layer] == cycle:
            self.runtime.adaptive_sample_drops += 1
            return 0
        current = self.current[layer]
        if current is not None and (current[1] != cycle or current[3] != index):
            self._submit(layer, compute)
            current = None
        if current is None:
            first = self.next_bank[layer]
            bank = next(
                (b for b in (first, 1 - first) if self.free[b][layer].is_set()), None
            )
            if bank is None:
                self.dropped_cycle[layer] = cycle
                self.runtime.adaptive_sample_drops += 1
                return 0
            self.free[bank][layer].clear()
            self.next_bank[layer] = 1 - bank
            # bank, cycle, first index, exclusive next index.
            current = self.current[layer] = [bank, cycle, index, index]
        return self.device[current[0], layer, index].data_ptr()

    def commit(self, layer, sample, compute):
        if sample is None:
            return
        current = self.current[layer]
        if current is None or current[1] != sample[0]:
            return
        current[3] = sample[1] + 1
        if current[3] == self.window:
            self._submit(layer, compute)

    def _submit(self, layer, compute):
        current = self.current[layer]
        if current is None:
            return
        bank, cycle, first, end = current
        if first == end:
            self.current[layer] = None
            self.free[bank][layer].set()
            return
        # compute already orders every preceding writer of this layer's rows.
        self.produced[bank][layer].record(compute)
        self.stream.wait_event(self.produced[bank][layer])
        with torch.cuda.stream(self.stream):
            self.host[bank, layer, first:end].copy_(
                self.device[bank, layer, first:end], non_blocking=True
            )
            self.copied[bank][layer].record(self.stream)
        self.jobs.put_nowait((bank, layer, cycle, first, end))
        self.current[layer] = None

    def _work(self):
        with torch.cuda.device(self.runtime.device):
            while True:
                item = self.jobs.get()
                if item is None:
                    self.jobs.task_done()
                    return
                bank, layer, cycle, first, end = item
                queued = False
                try:
                    self.copied[bank][layer].synchronize()
                    rows = self.host[bank, layer, first:end].tolist()
                    records = []
                    for row in rows:
                        count = row[0]
                        active = row[1 : count + 1]
                        if not 0 <= count < self.width or any(
                            not 0 <= expert < self.runtime.manifest.num_experts
                            for expert in active
                        ):
                            raise ValueError("invalid GPU route sample")
                        records.append(active)
                    if self.runtime.adaptive_cache:
                        try:
                            # One queue item per batch avoids a 48*window burst
                            # overflowing the per-route legacy analysis queue.
                            self.runtime.adaptive_samples.put_nowait(
                                RouteSampleBatch(
                                    (layer, cycle, first, end, records),
                                    self.free[bank][layer],
                                )
                            )
                            queued = True
                        except queue.Full:
                            self.runtime.adaptive_sample_drops += end - first
                except BaseException as error:
                    self.abort(error)
                finally:
                    # This also bounds queued Python batches to two banks per
                    # layer, rather than turning a 4096-item queue into 4096
                    # complete route windows when the analyzer is slow.
                    if not queued:
                        self.free[bank][layer].set()
                    self.jobs.task_done()

    def synchronize(self):
        # Caller drains layer-completion events first, including writers from
        # other compute streams. Flush partial rows before switching transports.
        compute = torch.cuda.current_stream(self.runtime.device)
        for layer in range(self.layers):
            self._submit(layer, compute)
        self.jobs.join()

    def close(self):
        self.jobs.put(None)
        self.worker.join()

    @property
    def device_bytes(self):
        return self.device.nbytes

    @property
    def pinned_bytes(self):
        return self.host.nbytes
