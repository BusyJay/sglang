// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
namespace a16 {
inline float from_bf16(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
inline uint16_t to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffU) > 0x7f800000U)
    return uint16_t(u >> 16) | 0x40;
  u += 0x7fff + ((u >> 16) & 1);
  return u >> 16;
}
inline float scale(uint8_t x) {
  int v = x & 127;
  float f = v == 127 ? NAN
                     : (v < 8 ? std::ldexp(float(v), -9)
                              : std::ldexp(1.f + (v & 7) / 8.f, (v >> 3) - 7));
  return x & 128 ? -f : f;
}
} // namespace a16
