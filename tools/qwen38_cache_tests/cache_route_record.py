"""Explicit diagnostic overlay for recording ordered target verify routes.

Enabled only by putting record_overlay on the recording worker's PYTHONPATH.
CPU route readback perturbs serving latency; never time this worker.
"""

import json
import os
import threading
from pathlib import Path


class RouteRecorder:
    def __init__(self, path, layers, limit):
        self.path = Path(path)
        self.layers, self.limit = layers, limit
        self.sweep, self.pending = 0, []
        self.lock = threading.Lock()
        if layers < 1 or limit < 1:
            raise ValueError("invalid recorder geometry or limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse stale recordings: append would invent cache continuity.
        with self.path.open("x"):
            pass

    def append(self, layer, routes):
        with self.lock:
            if self.sweep >= self.limit:
                return
            if layer != len(self.pending):
                raise ValueError(
                    "recording worker must have one ordered target request"
                )
            if not routes or len(routes) > 3:
                raise ValueError("expected 1..3 verify rows")
            if self.pending and len(routes) != len(self.pending[0]["routes"]):
                raise ValueError("verify shape changed inside sweep")
            self.pending.append(
                dict(
                    record_version=1,
                    sweep=self.sweep,
                    layer=layer,
                    phase="verify",
                    routes=routes,
                )
            )
            if len(self.pending) == self.layers:
                with self.path.open("a") as stream:
                    for row in self.pending:
                        stream.write(json.dumps(row) + "\n")
                self.pending = []
                self.sweep += 1


def install():
    # These are tool-owned overlay controls, not serving runtime env fields.
    path = os.environ.get("QWEN38_ROUTE_RECORD_PATH")
    if not path:
        return
    from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime

    original = Qwen38ThreadedExpertRuntime.run
    recorder = None
    owner = None
    # An isolated server can execute synthetic verify calls during graph setup.
    # Arm only after readiness, immediately before the one recorded request.
    arm_path = os.environ.get("QWEN38_ROUTE_RECORD_ARM")

    def record(
        self, layer_id, topk_ids, execute, *, execution_phase=None, during_fill=None
    ):
        nonlocal recorder, owner
        if execution_phase == "verify" and (
            arm_path is None or Path(arm_path).exists()
        ):
            if recorder is None:
                recorder = RouteRecorder(
                    path,
                    self.manifest.num_layers,
                    int(os.environ.get("QWEN38_ROUTE_RECORD_SWEEPS", "2048")),
                )
                owner = self
                provenance = dict(
                    record_version=1,
                    scope="target verify expert routes only",
                    layers=self.manifest.num_layers,
                    experts=self.manifest.num_experts,
                    top_k=self.manifest.top_k,
                    pid=os.getpid(),
                    manifest=str(self.manifest),
                    limits=[
                        "Recording synchronizes CUDA; not timing evidence",
                        "No KV/draft/accepted-prefix state recorded",
                        "Replay reconstructs cache from common prefix, not production cache snapshot",
                    ],
                )
                Path(path + ".meta.json").write_text(
                    json.dumps(provenance, indent=2) + "\n"
                )
            if owner is not self:
                raise ValueError("multiple target runtimes writing one tape")
            if recorder.sweep < recorder.limit:
                recorder.append(layer_id, topk_ids.detach().cpu().tolist())
        return original(
            self,
            layer_id,
            topk_ids,
            execute,
            execution_phase=execution_phase,
            during_fill=during_fill,
        )

    Qwen38ThreadedExpertRuntime.run = record
