// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <limits>

namespace sglang {
namespace details {
__global__ void partition_routes(
    const int32_t* ids,
    const int32_t* lookup,
    int32_t* cache,
    int32_t* arena,
    int64_t routes,
    int64_t experts,
    int64_t limit) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= routes) return;
  int32_t expert = ids[i];
  bool valid = expert >= 0 && expert < experts;
  int32_t slot = valid ? lookup[expert] : -1;
  bool hit = slot >= 0 && slot < limit;
  cache[i] = hit ? slot : -1;
  arena[i] = valid && !hit ? expert : -1;
}
}  // namespace details

/// Partition expert routes using the live GPU lookup and a protected slot prefix.
inline void qwen38_partition(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView lookup,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView arena,
    int64_t limit) {
  using namespace host;
  SymbolicSize tokens("tokens"), topk("topk"), experts("experts");
  SymbolicDevice device;
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(ids);
  TensorMatcher({experts}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(lookup);
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cache);
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(arena);
  // Global slots span layers; lookup length only bounds original expert IDs.
  CHECK_HOST(limit >= 0 && limit <= std::numeric_limits<int32_t>::max());
  int64_t n = tokens.unwrap() * topk.unwrap();
  if (n == 0) return;
  LaunchKernel(div_ceil(n, int64_t(256)), 256, device.unwrap())(
      sglang::details::partition_routes,
      static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const int32_t*>(lookup.data_ptr()),
      static_cast<int32_t*>(cache.data_ptr()),
      static_cast<int32_t*>(arena.data_ptr()),
      n,
      experts.unwrap(),
      limit);
}
}  // namespace sglang
