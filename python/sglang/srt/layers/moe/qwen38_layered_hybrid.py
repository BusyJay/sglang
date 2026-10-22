# SPDX-License-Identifier: Apache-2.0
"""CPU contribution to one layer's original LFU cache without global ownership."""

import logging
import os
import threading
from pathlib import Path

import torch

from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4
from sglang.srt.environ import envs
from sglang.srt.layers.moe.qwen38_layered_policy import (
    COLUMNS,
    calibration_rows,
    execution_metadata,
    fingerprint,
)

logger = logging.getLogger(__name__)


class LayeredCpuHybrid:
    def __init__(self, runtime, backend, mode):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        self.runtime = runtime
        self.backend = backend
        self.mode = mode
        self.max_cpu_experts = runtime.layered_cpu_max_experts if mode == "lfu" else 0
        self.expert_frequency = (
            torch.zeros(
                (runtime.manifest.num_layers, runtime.manifest.num_experts),
                dtype=torch.float32,
                device=runtime.device,
            )
            if mode == "lfu"
            else None
        )
        self.parallel_miss_plan = int(
            os.getenv("SGLANG_QWEN38_PARALLEL_MISS_PLAN", "2")
        )
        m = runtime.manifest
        if m.top_k > 10 or m.num_experts > 512 or not runtime.miss_mailbox:
            raise ValueError(
                "layered CPU hybrid requires top-k <= 10 and device mailbox"
            )
        if runtime.cache_policy != "lfu" or not runtime.adaptive_cache:
            raise ValueError("layered CPU hybrid requires adaptive per-layer LFU")
        if runtime.decode_early_cache:
            raise ValueError("layered CPU hybrid and early-cache decode are exclusive")
        self.cpu = None
        if mode != "zero":
            self.cpu = CpuNvfp4(
                envs.SGLANG_QWEN38_CPU_LIBRARY.get(),
                runtime.pack,
                tuple(int(c) for c in envs.SGLANG_QWEN38_CPU_CORES.get()),
                gpu_backend=backend,
            )
            if not all(
                hasattr(self.cpu.library, symbol)
                for symbol in ("qwen38_cpu_layered_service", "qwen38_cpu_layered_stop")
            ):
                self.cpu.close()
                raise ValueError("CPU library lacks layered service ABI; rebuild it")
        from sglang.srt.layers.moe.qwen38_hybrid_device import _MappedBatchPage

        self.stop = torch.zeros(1, dtype=torch.int32)
        self.failed = runtime.miss_mailbox.failed
        self.descriptor = torch.zeros(2, dtype=torch.int32, device=runtime.device)
        self.stats = torch.zeros(
            (m.num_layers, 5), dtype=torch.int64, device=runtime.device
        )
        self.gpu_weights = {
            n: torch.empty((n, m.top_k), dtype=torch.float32, device=runtime.device)
            for n in (1, 2, 3)
        }
        self.cpu_ids = {
            n: torch.empty((n, m.top_k), dtype=torch.int32, device=runtime.device)
            for n in (1, 2, 3)
        }
        sources = (
            Path(__file__),
            Path(__file__).with_name("qwen38_layered_policy.py"),
            Path(__file__).parents[3]
            / "kernels/jit/csrc/moe/qwen38_layered_policy.cuh",
            Path(__file__).parents[3] / "kernels/jit/csrc/moe/qwen38_expert_cache.cu",
            Path(__file__).parents[3] / "kernels/jit/csrc/moe/qwen38_hybrid_plan.cuh",
            Path(__file__).parents[3] / "kernels/jit/csrc/moe/qwen38_route_alphas.cuh",
            Path(__file__).with_name("qwen38_cache_mailbox.py"),
            Path(__file__).with_name("qwen38_cache_overlap.py"),
            Path(__file__).with_name("qwen38_cache_threaded.py"),
            Path(__file__).parents[1] / "quantization/qwen38_b12x_fp32.py",
            Path(__file__).parents[1] / "quantization/qwen38_nvfp4_cache.py",
            Path(__file__).parents[3] / "kernels/ops/moe/cpu_nvfp4/kernel.cpp",
            Path(__file__).parents[3] / "kernels/ops/moe/cpu_nvfp4/rows.hpp",
            Path(__file__).parents[3] / "kernels/ops/moe/cpu_nvfp4/base.hpp",
            Path(__file__).parents[1] / "quantization/qwen38_cached_b12x.py",
            Path(__file__).parents[1] / "quantization/qwen38_cached_w4a16.py",
        )
        metadata = dict(
            backend=backend,
            gpu=torch.cuda.get_device_name(runtime.device),
            layers=m.num_layers,
            experts=m.num_experts,
            cpu=self._cpu_name() if self.cpu is not None else None,
            cores=list(self.cpu.cores) if self.cpu is not None else [],
            hidden=m.hidden_size,
            width=m.intermediate_size,
            top_k=m.top_k,
            weight_format="nvfp4",
            scale_encoding=m.scale_encoding,
            transport="registered_mapped_page",
            source_fingerprint=fingerprint(*sources),
            **execution_metadata(self.cpu, runtime, self.parallel_miss_plan),
            torch_version=torch.__version__,
        )
        self.cost_metadata = metadata
        logger.info("Qwen38 layered CPU calibration metadata: %s", metadata)
        calibration = envs.SGLANG_QWEN38_LAYERED_CPU_CALIBRATION.get()
        try:
            rows = (
                calibration_rows(calibration, metadata) if mode == "calibrated" else []
            )
        except BaseException:
            if self.cpu is not None:
                self.cpu.close()
            raise
        if mode == "calibrated" and not any(row[3] > 0 for row in rows):
            self.cpu.close()
            self.cpu = None
        try:
            self.page = _MappedBatchPage(m.hidden_size, m.top_k, runtime.device)
            self.table = torch.tensor(
                rows, dtype=torch.int32, device=runtime.device
            ).reshape(-1, COLUMNS)
            _module()
        except BaseException:
            if hasattr(self, "page"):
                self.page.close(runtime.device)
            if self.cpu is not None:
                self.cpu.close()
            raise
        self.b12x = {}
        self.has_cpu_work = (
            mode == "probe"
            or (mode == "lfu" and self.max_cpu_experts > 0)
            or any(row[3] > 0 for row in rows)
        )
        self.calibrated_branches = frozenset(
            (row[4], row[42], row[5], bool(row[46])) for row in rows if row[3] > 0
        )
        self.sequence = 0
        self.closed = False
        self.failure = None
        self.thread = None
        if self.has_cpu_work:
            self.thread = threading.Thread(
                target=self._serve, name="qwen38-layered-cpu", daemon=True
            )
            self.thread.start()
        logger.info("Qwen38 layered CPU mode=%s calibrated_rows=%d", mode, len(rows))

    @staticmethod
    def _cpu_name():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
        return "unknown"

    def supports(self, layer, tokens, slots, shared_overlap):
        """Keep uncalibrated branch configurations on their original GPU path."""
        return (
            self.mode != "calibrated"
            or (layer, tokens, slots, shared_overlap) in self.calibrated_branches
            or (-1, tokens, -1, shared_overlap) in self.calibrated_branches
        )

    def _serve(self):
        import ctypes

        m = self.runtime.manifest
        lib = self.cpu.library
        service = lib.qwen38_cpu_layered_service
        service.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        service.restype = ctypes.c_int
        offsets = (ctypes.c_uint64 * (8 * m.num_layers))(
            *(int(v) for layer in range(m.num_layers) for v in self.cpu.offsets[layer])
        )
        strides = None
        if self.cpu.csf_strides is not None:
            strides = (ctypes.c_uint64 * (2 * m.num_layers))(
                *(int(v) for _ in range(m.num_layers) for v in self.cpu.csf_strides)
            )
        try:
            rc = service(
                self.cpu.address,
                len(self.cpu.mapping),
                offsets,
                strides,
                m.num_layers,
                m.num_experts,
                m.hidden_size,
                m.intermediate_size,
                m.top_k,
                len(self.cpu.cores),
                self.cpu.cores,
                self.page.batch.head.data_ptr(),
                self.failed.data_ptr(),
                self.stop.data_ptr(),
                self.cpu.post_scale,
            )
            if rc:
                raise RuntimeError(
                    lib.qwen38_cpu_error().decode() or "CPU service failed"
                )
        except BaseException as error:
            self.failure = error
            self.runtime.miss_mailbox._abort(error)

    def next_sequence(self):
        if self.failure is not None:
            raise RuntimeError("layered CPU service failed") from self.failure
        self.sequence += 1
        if self.sequence >= 2**31 - 1:
            raise RuntimeError("layered CPU sequence exhausted")
        return self.sequence

    def run_gpu(self, layer, x, ids, weights):
        tensors = {**self.runtime.caches[layer], **self.runtime.layer_scales[layer]}
        if self.backend == "b12x":
            return self.b12x[layer].run(
                x, ids, weights, tensors, output_dtype=torch.float32
            )
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe

        return cached_w4a16_moe(
            x,
            ids,
            weights,
            tensors,
            split_k=0,
            verify_mode="baseline",
            output_dtype=torch.float32,
        )

    def close(self):
        if self.closed:
            return
        # Caller has drained all GPU readers before unregistering the page.
        if self.thread is not None:
            import ctypes

            stop = self.cpu.library.qwen38_cpu_layered_stop
            stop.argtypes = [ctypes.c_void_p]
            stop.restype = None
            stop(self.stop.data_ptr())
            self.thread.join(timeout=envs.SGLANG_QWEN38_CPU_READY_TIMEOUT.get())
            if self.thread.is_alive():
                raise TimeoutError("layered CPU service did not stop")
            self.cpu.release_worker()
            self.thread = None
        if self.page is not None:
            self.page.close(self.runtime.device)
            self.page = None
        if self.cpu is not None:
            self.cpu.close()
            self.cpu = None
        self.closed = True
