// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

template <typename Id>
__global__ void qwen38_route_alphas_kernel(
    const Id* __restrict__ ids,
    const float* __restrict__ alpha1,
    const float* __restrict__ alpha2,
    const float* __restrict__ scale1,
    const float* __restrict__ scale2,
    float* __restrict__ output1,
    float* __restrict__ output2,
    int32_t routes,
    int64_t capacity) {
  const int32_t index = threadIdx.x;
  if (index >= routes) return;
  const int64_t slot = ids[index] < 0 ? 0 : static_cast<int64_t>(ids[index]);
  if (slot >= capacity) return;
  // MTP and padding repeat slots. Only the first occurrence may write, so
  // duplicate routes do not introduce concurrent writers to an alpha row.
  for (int32_t previous = 0; previous < index; ++previous) {
    const int64_t other = ids[previous] < 0 ? 0 : static_cast<int64_t>(ids[previous]);
    if (other == slot) return;
  }
  output1[slot] = alpha1[slot] * scale1[0];
  output2[slot] = alpha2[slot] * scale2[0];
}

/// Refresh only the pinned routed slots, including immutable padding slot zero.
/// Unrouted alpha rows may belong to another layer's concurrent cache fills.
template <typename Id>
void qwen38_route_alphas(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView alpha1,
    tvm::ffi::TensorView alpha2,
    tvm::ffi::TensorView scale1,
    tvm::ffi::TensorView scale2,
    tvm::ffi::TensorView output1,
    tvm::ffi::TensorView output2) {
  using namespace host;
  SymbolicSize M{"tokens"}, K{"top_k"}, C{"capacity"};
  SymbolicDevice device;
  TensorMatcher({M, K}).with_device<kDLCUDA>(device).with_dtype<Id>().verify(ids);
  TensorMatcher({C})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(alpha1)
      .verify(alpha2)
      .verify(output1)
      .verify(output2);
  const auto scalar = [&](tvm::ffi::TensorView value) {
    if (value.ndim() == 0) {
      TensorMatcher({}).with_dtype<float>().with_device<kDLCUDA>(device).verify(value);
    } else {
      TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(value);
    }
  };
  scalar(scale1);
  scalar(scale2);
  CHECK_HOST(M.unwrap() >= 1 && M.unwrap() <= 3);
  CHECK_HOST(K.unwrap() >= 1 && K.unwrap() <= 32);
  CHECK_HOST(C.unwrap() > 0);
  LaunchKernel(1, 128, device.unwrap())(
      qwen38_route_alphas_kernel<Id>,
      static_cast<const Id*>(ids.data_ptr()),
      static_cast<const float*>(alpha1.data_ptr()),
      static_cast<const float*>(alpha2.data_ptr()),
      static_cast<const float*>(scale1.data_ptr()),
      static_cast<const float*>(scale2.data_ptr()),
      static_cast<float*>(output1.data_ptr()),
      static_cast<float*>(output2.data_ptr()),
      static_cast<int32_t>(M.unwrap() * K.unwrap()),
      C.unwrap());
}

}  // namespace sglang
