# SPDX-License-Identifier: Apache-2.0
"""CPU-side format and mmap reader for Qwen3.8 NVFP4 runtime experts.

The pack stores tensors after ModelOpt's backend-specific post-load transform.
It is deliberately tied to an SGLang revision, FlashInfer backend and compute
capability; consuming a pack under a different ABI is unsafe.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

FORMAT_NAME = "sglang-qwen38-nvfp4-cache"
FORMAT_VERSION = 2
REQUIRED_TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
    "g1_alphas",
    "g2_alphas",
)
LAYER_TENSORS = ("w13_input_scale_quant", "w2_input_scale_quant")
PACK_TENSORS = REQUIRED_TENSORS + LAYER_TENSORS
SUPPORTED_DTYPES = {
    "uint8": 1,
    "float8_e4m3fn": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorRecord:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorRecord:
        return cls(
            offset=int(value["offset"]),
            nbytes=int(value["nbytes"]),
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
        )

    def validate(self, *, payload_nbytes: int, label: str) -> list[str]:
        errors: list[str] = []
        if self.offset < 0 or self.nbytes <= 0:
            errors.append(f"{label} has an invalid byte range")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            errors.append(f"{label} has an invalid shape {self.shape}")
        itemsize = SUPPORTED_DTYPES.get(self.dtype)
        if itemsize is None:
            errors.append(f"{label} has unsupported dtype {self.dtype!r}")
        elif self.shape:
            elements = 1
            for dim in self.shape:
                elements *= dim
            if elements * itemsize != self.nbytes:
                errors.append(
                    f"{label} nbytes={self.nbytes} does not match "
                    f"shape={self.shape} dtype={self.dtype}"
                )
        if self.offset + self.nbytes > payload_nbytes:
            errors.append(f"{label} extends beyond the payload")
        return errors


@dataclass(frozen=True)
class Qwen38ExpertPackManifest:
    payload_path: Path
    payload_nbytes: int
    checkpoint_revision: str
    sglang_commit: str
    flashinfer_backend: str
    compute_capability: str
    num_layers: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    tensors: dict[int, dict[str, TensorRecord]]

    def validate(self) -> None:
        errors: list[str] = []
        if self.num_layers <= 0 or self.num_experts <= 0 or self.top_k <= 0:
            errors.append("layer/expert/top-k counts must be positive")
        if self.num_experts <= 0:
            raise ValueError(
                "Invalid Qwen3.8 expert pack: expert count must be positive"
            )
        if self.top_k > self.num_experts:
            errors.append("top_k cannot exceed num_experts")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            errors.append("hidden and intermediate sizes must be positive")
        if self.flashinfer_backend != "flashinfer_cutlass":
            errors.append("runtime packs must target flashinfer_cutlass")
        if self.compute_capability != "sm120":
            errors.append("runtime packs must target sm120")
        if set(self.tensors) != set(range(self.num_layers)):
            errors.append("tensor table must cover every layer exactly once")

        ranges: list[tuple[int, int, str]] = []
        for layer, records in sorted(self.tensors.items()):
            missing = sorted(set(PACK_TENSORS) - set(records))
            extra = sorted(set(records) - set(PACK_TENSORS))
            if missing:
                errors.append(f"layer {layer} is missing tensors: {missing}")
            if extra:
                errors.append(f"layer {layer} has unknown tensors: {extra}")
            for name, record in records.items():
                label = f"layer {layer} {name}"
                errors.extend(
                    record.validate(payload_nbytes=self.payload_nbytes, label=label)
                )
                if name in LAYER_TENSORS:
                    if record.shape != (1,) or record.dtype != "float32":
                        errors.append(f"{label} must be a float32 [1] input scale")
                elif record.shape and record.shape[0] != self.num_experts:
                    errors.append(
                        f"{label} first dimension must contain all "
                        f"{self.num_experts} experts, got {record.shape[0]}"
                    )
                if name not in LAYER_TENSORS and record.nbytes % self.num_experts:
                    errors.append(f"{label} byte size is not divisible by num_experts")
                ranges.append((record.offset, record.offset + record.nbytes, label))

        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] > current[0]:
                errors.append(
                    f"tensor byte ranges overlap: {previous[2]} / {current[2]}"
                )
        if errors:
            raise ValueError("Invalid Qwen3.8 expert pack:\n- " + "\n- ".join(errors))

    def validate_cutlass_layout(self):
        """Check consumer view/shape requirements before reserving GPU memory."""
        self.validate()
        expected = {
            "w13_weight": (
                self.num_experts,
                2 * self.intermediate_size,
                self.hidden_size // 2,
            ),
            "w2_weight": (
                self.num_experts,
                self.hidden_size,
                self.intermediate_size // 2,
            ),
        }
        for layer, records in self.tensors.items():
            for name, shape in expected.items():
                record = records[name]
                if (
                    record.dtype != "uint8"
                    or record.shape != shape
                    or record.shape[-1] % 8
                ):
                    raise ValueError(
                        f"layer {layer} {name}: invalid packed CUTLASS weight layout"
                    )
            for name in ("g1_alphas", "g2_alphas"):
                record = records[name]
                if record.dtype != "float32" or record.shape != (self.num_experts,):
                    raise ValueError(
                        f"layer {layer} {name}: invalid CUTLASS alpha layout"
                    )
            for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
                record = records[name]
                if (
                    SUPPORTED_DTYPES[record.dtype] != 1
                    or len(record.shape) < 2
                    or record.shape[-1] % 4
                ):
                    raise ValueError(
                        f"layer {layer} {name}: invalid CUTLASS scale view"
                    )

    def assert_runtime_compatible(
        self,
        *,
        checkpoint_revision: str | None = None,
        sglang_commit: str | None = None,
        compute_capability: str = "sm120",
    ) -> None:
        expected = {
            "checkpoint_revision": checkpoint_revision,
            "sglang_commit": sglang_commit,
            "compute_capability": compute_capability,
        }
        actual = {
            "checkpoint_revision": self.checkpoint_revision,
            "sglang_commit": self.sglang_commit,
            "compute_capability": self.compute_capability,
        }
        mismatches = [
            f"{name}: pack={actual[name]!r}, runtime={value!r}"
            for name, value in expected.items()
            if value is not None and actual[name] != value
        ]
        if mismatches:
            raise ValueError(
                "Qwen3.8 expert pack ABI mismatch: " + "; ".join(mismatches)
            )

    def to_dict(self, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
        payload = self.payload_path
        if manifest_path is not None:
            parent = Path(manifest_path).resolve().parent
            try:
                payload_value = str(payload.relative_to(parent))
            except ValueError:
                payload_value = str(payload)
        else:
            payload_value = str(payload)
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "payload": payload_value,
            "checkpoint_revision": self.checkpoint_revision,
            "sglang_commit": self.sglang_commit,
            "flashinfer_backend": self.flashinfer_backend,
            "compute_capability": self.compute_capability,
            "model": {
                "num_layers": self.num_layers,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            },
            "layers": {
                str(layer): {
                    name: {
                        "offset": record.offset,
                        "nbytes": record.nbytes,
                        "shape": list(record.shape),
                        "dtype": record.dtype,
                    }
                    for name, record in records.items()
                }
                for layer, records in sorted(self.tensors.items())
            },
        }


def write_manifest(manifest: Qwen38ExpertPackManifest, path: str | Path) -> None:
    manifest.validate()
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(manifest_path=output), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def load_manifest(path: str | Path) -> Qwen38ExpertPackManifest:
    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("format") != FORMAT_NAME
        or int(value.get("version", -1)) != FORMAT_VERSION
    ):
        raise ValueError("unsupported Qwen3.8 expert-pack format or version")
    payload_path = Path(value["payload"])
    if not payload_path.is_absolute():
        payload_path = manifest_path.parent / payload_path
    payload_path = payload_path.resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(
            f"Qwen3.8 expert-pack payload is missing: {payload_path}"
        )
    tensors = {
        int(layer): {
            name: TensorRecord.from_dict(record) for name, record in records.items()
        }
        for layer, records in value["layers"].items()
    }
    result = Qwen38ExpertPackManifest(
        payload_path=payload_path,
        payload_nbytes=payload_path.stat().st_size,
        checkpoint_revision=str(value["checkpoint_revision"]),
        sglang_commit=str(value["sglang_commit"]),
        flashinfer_backend=str(value["flashinfer_backend"]),
        compute_capability=str(value["compute_capability"]),
        num_layers=int(value["model"]["num_layers"]),
        num_experts=int(value["model"]["num_experts"]),
        top_k=int(value["model"]["top_k"]),
        hidden_size=int(value["model"]["hidden_size"]),
        intermediate_size=int(value["model"]["intermediate_size"]),
        tensors=tensors,
    )
    result.validate()
    return result


class Qwen38ExpertPack:
    """Read-only byte views over one file mapping, also shared by CUDA readers."""

    def __init__(self, manifest: Qwen38ExpertPackManifest):
        self.manifest = manifest
        self._stream = manifest.payload_path.open("rb", buffering=0)
        self._mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)
        self._cuda_source_private = False
        self._prefetch_thread = None
        self._prefetch_stop = threading.Event()
        self._prefetch_error = None
        self._source_mode = os.getenv("SGLANG_QWEN38_SOURCE_MODE", "anonymous_thp")
        if self._source_mode not in ("file_cow", "cow_prefault", "anonymous", "anonymous_thp"):
            self._mapping.close()
            self._stream.close()
            raise ValueError("invalid SGLANG_QWEN38_SOURCE_MODE")
        self._source_prepared = False
        self._prefetch_active = False
        self._prefetch_ready = threading.Event()
        if self._source_mode == "cow_prefault":
            self._mapping.close()
            self._mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_COPY)
            self._cuda_source_private = True
        elif self._source_mode.startswith("anonymous"):
            self._mapping.close()
            self._mapping = mmap.mmap(-1, self.manifest.payload_nbytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE)
            if self._source_mode == "anonymous_thp":
                self._mapping.madvise(mmap.MADV_HUGEPAGE)
            self._cuda_source_private = True

    def start_prefetch(self) -> None:
        """Populate the shared file page cache while model weights load."""
        if self._prefetch_thread is not None:
            return
        self._prefetch_active = True
        self._prefetch_thread = threading.Thread(
            target=self._prefetch, name="qwen38-pack-prefetch", daemon=True
        )
        self._prefetch_thread.start()

    def _prefetch(self) -> None:
        if self._source_mode != "file_cow":
            try:
                self._prepare_source()
            except BaseException as exc:
                self._prefetch_error = exc
            finally:
                self._prefetch_ready.set()
            return
        started = time.perf_counter()
        size = self.manifest.payload_nbytes
        offset = 0
        try:
            fd = os.open(self.manifest.payload_path, os.O_RDONLY)
            try:
                while offset < size and not self._prefetch_stop.is_set():
                    count = min(8 << 20, size - offset)
                    data = os.pread(fd, count, offset)
                    if len(data) != count:
                        raise OSError(f"short expert-pack read at offset {offset}")
                    offset += count
            finally:
                os.close(fd)
        except OSError as exc:
            self._prefetch_error = exc
            logger.warning("Expert-pack prefetch stopped at %d bytes: %s", offset, exc)
        else:
            if offset == size:
                logger.info(
                    "Expert-pack prefetch completed: %.2f GiB in %.2f s",
                    offset / (1 << 30), time.perf_counter() - started,
                )

    def _prepare_source(self):
        if self._source_prepared:
            return
        started = time.perf_counter()
        from sglang.srt.layers.moe.qwen38_observation import observe
        with observe("source_prefault_or_load", startup=True, mode=self._source_mode,
                     source_bytes=self.manifest.payload_nbytes):
            if self._source_mode == "cow_prefault":
                # Writable population preserves file bytes, but materializes COW.
                self._mapping.madvise(getattr(mmap, "MADV_POPULATE_WRITE", 23))
            else:
                view = memoryview(self._mapping)
                try:
                    offset = 0
                    while offset < len(view):
                        if self._prefetch_stop.is_set():
                            raise RuntimeError("expert source preparation cancelled")
                        end = min(offset + (8 << 20), len(view))
                        part = view[offset:end]
                        try:
                            count = os.preadv(self._stream.fileno(), [part], offset)
                        finally:
                            part.release()
                        if count <= 0:
                            raise OSError("short expert source read")
                        offset += count
                finally:
                    view.release()
            self._source_prepared = True
        logger.info(
            "Qwen38 startup expert source ready: mode=%s GiB=%.2f seconds=%.3f",
            self._source_mode,
            self.manifest.payload_nbytes / (1 << 30),
            time.perf_counter() - started,
        )

    def wait_prefetch(self) -> None:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
        if self._source_mode != "file_cow":
            if self._prefetch_error is not None:
                raise RuntimeError("expert source preparation failed") from self._prefetch_error
            self._prepare_source()

    def _wait_source_readable(self):
        # Early alpha/scalar CPU reads must never observe uninitialized source.
        if self._source_mode.startswith("anonymous") and not self._source_prepared:
            if self._prefetch_active:
                self._prefetch_ready.wait()
                if self._prefetch_error is not None:
                    raise RuntimeError("expert source preparation failed") from self._prefetch_error
            else:
                self._prepare_source()

    def cuda_source_mapping(self) -> mmap.mmap:
        """Promote at startup to the single CUDA-registerable source mapping.

        GeForce cannot register CPU-read-only pages. All transports therefore
        share one private writable mapping; CUDA registration materializes its
        COW pages. Application byte views stay read-only and writes cannot reach
        the file. The pack owns the mapping beyond each mailbox's registration.
        Existing exported views must have been released before promotion.
        """
        self.wait_prefetch()
        if not self._cuda_source_private:
            mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_COPY)
            try:
                self._mapping.close()
            except BaseException:
                mapping.close()
                raise
            self._mapping = mapping
            self._cuda_source_private = True
        return self._mapping

    def discard_source_file_cache(self) -> None:
        """Release redundant prefetch pages after CUDA materializes private pages."""
        discard = os.getenv("SGLANG_QWEN38_DISCARD_SOURCE_FILE_CACHE", "0")
        if discard not in ("0", "1"):
            raise ValueError("SGLANG_QWEN38_DISCARD_SOURCE_FILE_CACHE must be 0 or 1")
        if discard == "0":
            logger.info("Expert-pack file cache retained for warm restarts")
            return
        if self._cuda_source_private and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(
                    self._stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                )
            except OSError as exc:
                logger.warning("Could not discard expert-pack file cache: %s", exc)

    def tensor_bytes(self, layer: int, name: str) -> memoryview:
        self._wait_source_readable()
        record = self.manifest.tensors[layer][name]
        return memoryview(self._mapping)[
            record.offset : record.offset + record.nbytes
        ].toreadonly()

    def expert_tensor_bytes(self, layer: int, expert: int, name: str) -> memoryview:
        if not 0 <= expert < self.manifest.num_experts:
            raise IndexError(f"expert {expert} is outside the runtime pack")
        if name in LAYER_TENSORS:
            raise ValueError(f"{name} is a layer scalar, not an expert tensor")
        self._wait_source_readable()
        record = self.manifest.tensors[layer][name]
        expert_nbytes = record.nbytes // self.manifest.num_experts
        start = record.offset + expert * expert_nbytes
        return memoryview(self._mapping)[start : start + expert_nbytes].toreadonly()

    def layer_tensors(
        self, layer: int
    ) -> Iterator[tuple[str, TensorRecord, memoryview]]:
        for name in PACK_TENSORS:
            record = self.manifest.tensors[layer][name]
            yield name, record, self.tensor_bytes(layer, name)

    def close(self) -> None:
        self._prefetch_stop.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
        if not self._mapping.closed:
            self._mapping.close()
        self._stream.close()

    def __enter__(self) -> Qwen38ExpertPack:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
