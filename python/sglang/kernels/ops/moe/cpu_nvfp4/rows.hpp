// SPDX-License-Identifier: Apache-2.0
// Direct NVFP4 -> BF16 lookup; no float conversion in the inner decode loop.
#pragma once
#include "base.hpp"
namespace a16global {
struct alignas(64) Tables {
  alignas(32) uint16_t w[256][16];
  explicit Tables(float global) {
    const float v[16] = {0, .5,  1,  1.5,  2,  3,  4,  6,
                         0, -.5, -1, -1.5, -2, -3, -4, -6};
    for (int s = 0; s < 256; s++) {
      const float sf = a16::scale(s);
      for (int q = 0; q < 16; q++)
        w[s][q] = a16::to_bf16(v[q] * sf * global);
    }
  }
};
// CUTLASS cache uses adjacent low/high nibbles, unlike ggml block_nvfp4.
inline __m512bh decode32(const uint8_t *p, uint8_t s0, uint8_t s1,
                         const Tables &t) {
  __m128i bytes = _mm_loadu_si128((const __m128i *)p);
  __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(15));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(15));
  __m256i q =
      _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
  __m512i idx = _mm512_cvtepu8_epi16(q);
  idx = _mm512_add_epi16(idx, _mm512_inserti64x4(_mm512_setzero_si512(),
                                                 _mm256_set1_epi16(16), 1));
  __m512i tab = _mm512_inserti64x4(
      _mm512_castsi256_si512(_mm256_load_si256((const __m256i *)t.w[s0])),
      _mm256_load_si256((const __m256i *)t.w[s1]), 1);
  return (__m512bh)_mm512_permutexvar_epi16(idx, tab);
}
inline size_t scale_offset(int row, int col, int columns) {
  int padded = (columns + 3) / 4 * 4;
  return (((row / 128 * (padded / 4) + col / 4) * 32 + row % 32) * 4 +
          (row % 128) / 32) *
             4 +
         col % 4;
}

template <int NT, int NR>
inline void project(const uint8_t *weights, const uint8_t *scales,
                    const int (&rows)[NR], const uint16_t *const *a, int n,
                    const Tables &table, float (&out)[NR][NT]) {
  const uint8_t *wp[NR], *sp[NR];
#pragma GCC unroll 8
  for (int r = 0; r < NR; ++r) {
    wp[r] = weights + size_t(rows[r]) * (n / 2);
    sp[r] = scales + scale_offset(rows[r], 0, n / 16);
  }
  __m512 acc[NR][NT];
#pragma GCC unroll 8
  for (int r = 0; r < NR; ++r) {
#pragma GCC unroll 3
    for (int t = 0; t < NT; ++t)
      acc[r][t] = _mm512_setzero_ps();
  }
  for (unsigned k = 0; k < unsigned(n); k += 32) {
    const unsigned so = (k / 64) * 512 + (k / 16) % 4;
    __m512bh acts[NT];
#pragma GCC unroll 3
    for (int t = 0; t < NT; ++t)
      acts[t] = (__m512bh)_mm512_loadu_si512(a[t] + k);
#pragma GCC unroll 8
    for (int r = 0; r < NR; ++r) {
      __m512bh w = decode32(wp[r] + k / 2, sp[r][so], sp[r][so + 1], table);
#pragma GCC unroll 3
      for (int t = 0; t < NT; ++t)
        acc[r][t] = _mm512_dpbf16_ps(acc[r][t], w, acts[t]);
    }
  }
#pragma GCC unroll 8
  for (int r = 0; r < NR; ++r) {
#pragma GCC unroll 3
    for (int t = 0; t < NT; ++t)
      out[r][t] = _mm512_reduce_add_ps(acc[r][t]);
  }
}

} // namespace a16global
