# SPDX-License-Identifier: Apache-2.0
"""Refresh only routed B12X scale rows during concurrent global-cache fills."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module(int64_ids):
    dtype = "int64_t" if int64_ids else "int32_t"
    return load_jit(
        "qwen38_route_alphas",
        cuda_files=["moe/qwen38_route_alphas.cuh"],
        cuda_wrappers=[
            ("run", f"qwen38_route_alphas<{dtype}>"),
            ("prepare_fc1", "prepare_qwen38_fc1_routes"),
            ("prepare_fc1_resident", "prepare_qwen38_fc1_resident_routes"),
        ],
    )


def refresh_route_alphas(ids, alpha1, alpha2, scale1, scale2, output1, output2):
    """Caller pins routed source rows; output tables must not alias the sources."""
    # Cached W4A16 accepts any scalar shape. The common scalar/vector forms
    # pass straight through; preserve higher-rank singleton compatibility.
    if scale1.ndim > 1:
        scale1 = scale1.reshape(())
    if scale2.ndim > 1:
        scale2 = scale2.reshape(())
    _module(ids.dtype == torch.int64).run(
        ids, alpha1, alpha2, scale1, scale2, output1, output2
    )


def prepare_fc1_routes(
    early,
    remapped,
    late,
    alpha1,
    alpha2,
    scale1,
    scale2,
    output1,
    output2,
    *,
    resident_alphas=None,
):
    """Build complementary FC1 tasks and refresh safe routed alpha rows.

    With resident_alphas=(plan, table1, table2), update planned cache alpha
    rows and refresh all remapped globals. The caller must exclude alpha rows
    from concurrent fills. Without it, read protected hits only.
    """
    module = _module(False)
    launch = (
        module.prepare_fc1 if resident_alphas is None else module.prepare_fc1_resident
    )
    launch(
        early,
        remapped,
        late,
        alpha1,
        alpha2,
        scale1.reshape(()),
        scale2.reshape(()),
        output1,
        output2,
        *(resident_alphas if resident_alphas is not None else ()),
    )
