# SPDX-License-Identifier: Apache-2.0
"""Opt-in detailed host/stream intervals; never enabled by normal benchmarks."""
import contextlib
import threading
import time


class CacheProfile:
    def __init__(self, cuda):
        self.cuda = cuda
        self.host = []
        self.gpu = []
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def host_range(self, kind):
        start = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                self.host.append(dict(kind=kind, begin_s=start, end_s=time.perf_counter()))

    @contextlib.contextmanager
    def gpu_range(self, kind, stream):
        a, b = self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True)
        a.record(stream)
        try:
            yield
        finally:
            b.record(stream)
            with self.lock:
                self.gpu.append((kind, a, b))

    def results(self):
        # Caller drains runtime and synchronizes before reading; no per-copy waits.
        return dict(host_intervals=list(self.host), gpu_intervals=[
            dict(kind=kind, elapsed_ms=a.elapsed_time(b)) for kind, a, b in self.gpu],
            note="Host CPU staging includes dispatch; transfer intervals include stream waits/gaps. Nested ranges overlap; do not sum them as exclusive time.")
