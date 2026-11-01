# SPDX-License-Identifier: Apache-2.0
"""Shared local/EPD vision embedding bridge and absolute prompt spans."""

from dataclasses import dataclass
from itertools import pairwise

import mlx.core as mx
import numpy as np


def to_mlx(value):
    if isinstance(value, mx.array):
        return value
    if hasattr(value, "detach"):
        # CPU torch does not expose BF16 through numpy. Widen on transfer only;
        # callers cast back to the checkpoint compute dtype before projection.
        value = value.detach().cpu().float().numpy()
    return mx.array(value)


@dataclass(frozen=True)
class VisionPrompt:
    spans: tuple
    source: object = None

    def embed(self, embed_tokens, tokens, offset):
        stop = offset + tokens.shape[1]
        overlaps = [
            (start, end, values)
            for start, end, values in self.spans
            if start < stop and end > offset
        ]
        if not overlaps:
            return None
        # Hash sentinels are outside the vocabulary. Only covered image rows
        # may be replaced before the embedding table lookup.
        safe = tokens
        for start, end, _ in overlaps:
            left, right = max(start, offset) - offset, min(end, stop) - offset
            safe = mx.concatenate(
                [
                    safe[:, :left],
                    mx.zeros((1, right - left), dtype=safe.dtype),
                    safe[:, right:],
                ],
                1,
            )
        hidden = embed_tokens(safe)
        for start, end, values in overlaps:
            left, right = max(start, offset), min(end, stop)
            hidden = mx.concatenate(
                [
                    hidden[:, : left - offset],
                    values[left - start : right - start].astype(hidden.dtype)[None],
                    hidden[:, right - offset :],
                ],
                1,
            )
        return hidden


def prepare_vision_prompt(mm_inputs, encoder):
    spans = []
    for item in mm_inputs.mm_items:
        offsets = item.offsets or []
        if not offsets:
            raise ValueError("Vision items must have absolute token offsets")
        if item.precomputed_embeddings is not None:
            values = to_mlx(item.precomputed_embeddings)
        else:
            if encoder is None:
                raise ValueError("Language-only MLX requires encoder embeddings")
            grid = item.model_specific_data.get(
                "image_grid_thw", item.model_specific_data.get("video_grid_thw")
            )
            if grid is None:
                raise ValueError("GLM vision requires image/video grid_thw")
            if hasattr(grid, "cpu"):
                grid = grid.cpu().numpy()
            values = encoder(to_mlx(item.feature), grid)
        values = values.reshape(-1, values.shape[-1])
        expected = sum(end - start + 1 for start, end in offsets)
        if values.shape[0] != expected:
            raise ValueError(
                f"Vision embedding rows {values.shape[0]} do not match prompt spans {expected}"
            )
        cursor = 0
        for start, end in offsets:
            count = end - start + 1
            if start < 0 or count <= 0:
                raise ValueError("Invalid vision token span")
            spans.append((start, end + 1, values[cursor : cursor + count]))
            cursor += count
    spans.sort(key=lambda span: span[0])
    if any(a[1] > b[0] for a, b in pairwise(spans)):
        raise ValueError("Overlapping vision token spans")
    mx.async_eval([span[2] for span in spans])
    return VisionPrompt(tuple(spans), mm_inputs)


class MlxEncoderModel:
    """Torch CPU transport surface around a native MLX-only vision tower."""

    def __init__(self, model_dir):
        import torch
        from sglang.srt.hardware_backend.mlx.models.glm5_vision import load_glm5_vision

        self.visual = load_glm5_vision(model_dir)
        self._dtype_marker = torch.empty(0, dtype=torch.bfloat16)

    def parameters(self):
        return iter((self._dtype_marker,))

    def get_image_feature(self, items):
        import torch

        grids, features = [], []
        for item in items:
            grid = item.model_specific_data.get(
                "image_grid_thw", item.model_specific_data.get("video_grid_thw")
            )
            if hasattr(grid, "cpu"):
                grid = grid.cpu().numpy()
            grids.append(np.asarray(grid))
            features.append(to_mlx(item.feature))
        output = self.visual(mx.concatenate(features, 0), np.concatenate(grids, 0))
        mx.eval(output)
        return torch.from_numpy(np.array(output.astype(mx.float32))).to(torch.bfloat16)

    get_video_feature = get_image_feature
