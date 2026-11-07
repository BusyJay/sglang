"""Sampling kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    DeviceType,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="sampling.softmax",
        backend=KernelBackend.CAKE,
        target="sglang.kernels.cake_kernels.sampling:softmax",
        capabilities=frozenset(
            {
                CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 0)),
                CapabilityRequirement.cuda(min_sm=(10, 3), max_sm=(10, 3)),
            }
        ),
    )
)


def cake_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Explicit Cake softmax entry point for Blackwell qualification."""
    return get_kernel("sampling.softmax", KernelBackend.CAKE)(logits)


def softmax(
    logits: torch.Tensor, temperatures: torch.Tensor | None = None
) -> torch.Tensor:
    """Softmax for sampler logits, optionally with per-row temperatures."""
    import torch

    if temperatures is not None and logits.is_cuda and torch.version.hip is not None:
        from sglang.kernels.ops.sampling.temperature_softmax import (
            temperature_softmax,
        )

        return temperature_softmax(logits, temperatures)

    if temperatures is not None:
        logits = logits / temperatures
    from sglang.kernels.cake_kernels.sampling import supports_softmax

    if supports_softmax(logits):
        return cake_softmax(logits)
    return torch.softmax(logits, dim=-1)


__all__ = ["cake_softmax", "softmax"]


# Migrated from srt/layers/utils/hash.py (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="sampling.murmur_hash32",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.murmur_hash:murmur_hash32",
    )
)


register_kernel(
    KernelSpec(
        op="sampling.denoiser_statistics",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.denoiser_statistics:denoiser_statistics",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="sampling.softcap_row_argmax",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.softcap_row_argmax:softcap_row_argmax",
        capabilities=CUDA_ONLY,
    )
)


register_kernel(
    KernelSpec(
        op="sampling.mlx_topk",
        backend=KernelBackend.MLX,
        target="sglang.kernels.ops.sampling.mlx_topk:mlx_topk",
        capabilities=frozenset({CapabilityRequirement(device=DeviceType.MPS)}),
    )
)


def mlx_topk(logits, k=16):
    """Exact MLX top-k, descending values with lower-id ties."""
    return get_kernel("sampling.mlx_topk", KernelBackend.MLX)(logits, k)


__all__.append("mlx_topk")

register_kernel(
    KernelSpec(
        op="sampling.mlx_topk_candidates",
        backend=KernelBackend.MLX,
        target="sglang.kernels.ops.sampling.mlx_topk:mlx_topk_candidates",
        capabilities=frozenset({CapabilityRequirement(device=DeviceType.MPS)}),
    )
)


def mlx_topk_candidates(logits, token_ids, k=16, *, valid_count=None):
    """Select distinct global token IDs from a padded candidate vocabulary."""
    return get_kernel("sampling.mlx_topk_candidates", KernelBackend.MLX)(
        logits, token_ids, k, valid_count=valid_count
    )


__all__.append("mlx_topk_candidates")

for _name in ("mlx_vocabulary_set", "mlx_vocabulary_candidates"):
    register_kernel(
        KernelSpec(
            op=f"sampling.{_name}",
            backend=KernelBackend.MLX,
            target=f"sglang.kernels.ops.sampling.mlx_vocabulary_set:{_name}",
            capabilities=frozenset({CapabilityRequirement(device=DeviceType.MPS)}),
        )
    )
