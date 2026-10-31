# SPDX-License-Identifier: Apache-2.0
# Adapted from jundot/omlx decode_kernels.py, commit 5c6cbd6314ca.
# Kernel arithmetic originally follows Apple's MLX Metal implementations.
"""Short-convolution, normalization, delta recurrence and gate for MLX KDA.

One sequence, 1..8 tokens, head width 128. Conv states remain separate to
match the caller's cache layout without concatenating snapshots each step.
Gate projections are explicit inputs; this also supports dense BF16 weights.
"""

from functools import cache

import mlx.core as mx

_HEADER = r"""
template <typename T>
inline T glm_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

"""

_SOURCE = r"""

  constexpr int CK = 4;
  constexpr int NROW = CK - 1 + TOK;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const float q_scale = params[0];
  const float l2_eps = params[1];
  const float norm_eps = params[2];
  const float lower = params[3];
  const float inv_n = params[4];

  threadgroup T qs[TOK][DK];
  threadgroup T ks[TOK][DK];
  threadgroup T vs[TOK][DK];
  threadgroup T as_[TOK][DK];
  threadgroup T gates[TOK][DK];
  threadgroup T ys[TOK][DK];
  threadgroup float gs[TOK][DK];
  threadgroup T betas[TOK];

  // ---- 1. short conv + SiLU -------------------------------------------------
  if (tid < uint(3 * DK)) {
    const int part = int(tid) / DK;
    const int i = int(tid) % DK;
    const int gc = part * QKV + h * DK + i;
    T win[NROW];
    for (int r = 0; r < CK - 1; r++) {
#if HAS_CONV_STATE
      const device T* history =
          part == 0 ? state_q : (part == 1 ? state_k : state_v);
      win[r] = history[r * QKV + h * DK + i];
#else
      win[r] = static_cast<T>(0);
#endif
    }
    for (int t = 0; t < TOK; t++) {
      win[CK - 1 + t] = proj[t * PROJ_W + gc];
    }
    const device T* w = part == 0 ? conv_q : (part == 1 ? conv_k : conv_v);
    w += (h * DK + i) * CK;
    for (int t = 0; t < TOK; t++) {
      float acc = 0.0;
      for (int j = 0; j < CK; ++j) {
        acc += static_cast<float>(win[t + j]) * w[j];
      }
      T co = static_cast<T>(acc);
      T sgm = glm_sigmoid<T>(co);
      T sv = co * sgm;
      if (part == 0) {
        qs[t][i] = sv;
      } else if (part == 1) {
        ks[t][i] = sv;
      } else {
        vs[t][i] = sv;
      }
    }
    for (int r = 0; r < CK - 1; r++) {
      device T* history = part == 0 ? next_q : (part == 1 ? next_k : next_v);
      history[r * QKV + h * DK + i] = win[TOK + r];
    }
  }

  // ---- 3a. low-rank gate projections (qmv_quad rows of this head) ----------

  for (int e = int(tid); e < TOK * DK; e += 1024) {
    const int t = e / DK;
    const int i = e % DK;
    as_[t][i] = a_pre[t * QKV + h * DK + i];
    gates[t][i] = gate_pre[t * QKV + h * DK + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 2. l2norm(q) * scale, l2norm(k) --------------------------------------
  if (sg < uint(2 * TOK)) {
    const int t = int(sg) / 2;
    const bool is_q = (sg % 2) == 0;
    threadgroup T* row = is_q ? qs[t] : ks[t];
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(row[4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float mean = tot * inv_n;
    float r = metal::precise::rsqrt(mean + l2_eps * inv_n);
    for (int e = 0; e < 4; e++) {
      T normalized = static_cast<T>(x[e] * r);
      T scaled = normalized * T(is_q ? inv_n : q_scale);
      row[4 * lane + e] = scaled;
    }
  }

  // ---- 3b. g = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta -------
  if (tid < uint(TOK * DK)) {
    const int t = int(tid) / DK;
    const int i = int(tid) % DK;
    float ea = metal::precise::exp(a_log[h]);
    float af = static_cast<float>(as_[t][i]);
    float s1 = af + dt_bias[h * DK + i];
    float s2 = ea * s1;
    float s3 = glm_sigmoid<float>(s2);
    float s4 = lower * s3;
    gs[t][i] = metal::precise::exp(s4);
  }
  if (tid < uint(TOK)) {
    betas[tid] = glm_sigmoid<T>(proj[tid * PROJ_W + OFF_B + h]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 4. vector-gated delta rule --------------------------------------------
  for (int j = 0; j < DK / 32; j++) {
    const int dv_idx = int(sg) + 32 * j;
    constexpr int n_per_t = DK / 32;
    const int dk_idx = int(lane);
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
#if HAS_STATE
      state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
#else
      state[i] = 0.0f;
#endif
    }
    for (int t = 0; t < TOK; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gs[t][s_idx];
        kv_mem += state[i] * ks[t][s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + ks[t][s_idx] * delta;
        out += state[i] * qs[t][s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        ys[t][dv_idx] = static_cast<T>(out);
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 5. RMSNormGated ---------------------------------------------------------
  if (sg < uint(TOK)) {
    const int t = int(sg);
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(ys[t][4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float var = tot * inv_n;
    float u = var + norm_eps;
    float r = metal::precise::rsqrt(u);
    for (int e = 0; e < 4; e++) {
      const int c = 4 * lane + e;
      float xn = x[e] * r;
      float wf = static_cast<float>(norm_w[c]);
      float wx = wf * xn;
      float gf = static_cast<float>(gates[t][c]);
      float gsg = glm_sigmoid<float>(gf);
      float o = wx * gsg;
      y[t * QKV + h * DK + c] = static_cast<T>(o);
    }
  }

"""


@cache
def _kernel(has_conv, has_state):
    source = (
        f"#define HAS_CONV_STATE {int(has_conv)}\n"
        f"#define HAS_STATE {int(has_state)}\n" + _SOURCE
    )
    return mx.fast.metal_kernel(
        name=f"sglang_kda_c{int(has_conv)}_s{int(has_state)}",
        input_names=[
            "proj",
            "conv_q",
            "conv_k",
            "conv_v",
            "a_log",
            "dt_bias",
            "norm_w",
            "params",
            "state_q",
            "state_k",
            "state_v",
            "state_in",
            "a_pre",
            "gate_pre",
        ],
        output_names=["y", "next_q", "next_k", "next_v", "state_out"],
        header=_HEADER,
        source=source,
    )


def mlx_kda_decode(
    proj,
    conv_weights,
    conv_states,
    state,
    a,
    gate,
    a_log,
    dt_bias,
    norm_weight,
    *,
    heads,
    norm_eps,
    lower_bound,
):
    """Return (gated output, three conv states, FP32 recurrence), or None.

    Inputs are immutable. Unsupported shape/dtype combinations leave all
    caller-owned cache state untouched for the ordinary MLX path.
    """
    if (
        proj.ndim != 3
        or proj.shape[0] != 1
        or not 1 <= proj.shape[1] <= 8
        or proj.dtype not in (mx.bfloat16, mx.float16)
        or lower_bound is None
    ):
        return None
    tokens, dim = proj.shape[1], heads * 128
    if proj.shape[-1] < 3 * dim or len(conv_weights) != 3 or len(conv_states) != 3:
        return None
    if any(w.shape != (dim, 4, 1) or w.dtype != proj.dtype for w in conv_weights):
        return None
    if any(v.shape != (1, tokens, dim) or v.dtype != proj.dtype for v in (a, gate)):
        return None
    if (
        norm_weight.shape != (128,)
        or norm_weight.dtype != proj.dtype
        or a_log.size != heads
        or dt_bias.size != dim
        or a_log.dtype != mx.float32
        or dt_bias.dtype != mx.float32
    ):
        return None
    has_conv = conv_states[0] is not None
    if any((s is not None) != has_conv for s in conv_states):
        return None
    if has_conv and any(
        s.shape != (1, 3, dim) or s.dtype != proj.dtype for s in conv_states
    ):
        return None
    if state is not None and (
        state.shape != (1, heads, 128, 128) or state.dtype != mx.float32
    ):
        return None
    # A weight is a harmless dummy pointer for a compile-time absent state.
    conv = conv_states if has_conv else conv_weights
    params = mx.array([128**-0.5, 1e-6, norm_eps, lower_bound, 1 / 128], mx.float32)
    out = _kernel(has_conv, state is not None)(
        inputs=[
            proj,
            *conv_weights,
            a_log.reshape(-1),
            dt_bias.reshape(-1),
            norm_weight,
            params,
            *conv,
            state if state is not None else a_log,
            a,
            gate,
        ],
        template=[
            ("T", proj.dtype),
            ("TOK", tokens),
            ("DK", 128),
            ("QKV", dim),
            ("PROJ_W", proj.shape[-1]),
            ("OFF_B", proj.shape[-1] - heads),
        ],
        grid=(1024 * heads, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, tokens, dim), *[(1, 3, dim)] * 3, (1, heads, 128, 128)],
        output_dtypes=[proj.dtype] * 4 + [mx.float32],
    )
    return out[0], out[1:4], out[4]
