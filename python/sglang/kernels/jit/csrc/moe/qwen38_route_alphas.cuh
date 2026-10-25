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

namespace sglang {
template <bool Resident = false>
__global__ void prepare_fc1_routes_kernel(
    const int32_t* early,
    const int32_t* remapped,
    int32_t* late,
    const float* alpha1,
    const float* alpha2,
    const float* scale1,
    const float* scale2,
    float* output1,
    float* output2,
    int32_t routes,
    int32_t capacity,
    const int32_t* plan,
    const float* table1,
    const float* table2,
    int32_t experts) {
  int32_t i = threadIdx.x;
  if constexpr (Resident) {
    if (plan[0] || plan[3]) return;
    for (int32_t miss = i; miss < plan[1] && miss < capacity; miss += blockDim.x) {
      const int32_t expert = plan[4 + 2 * miss], slot = plan[5 + 2 * miss];
      if (expert >= 0 && expert < experts && slot >= 0 && slot < capacity) {
        const_cast<float*>(alpha1)[slot] = table1[expert];
        const_cast<float*>(alpha2)[slot] = table2[expert];
      }
    }
    __syncthreads();
  }
  if (i >= routes) return;
  late[i] = early[i] >= 0 ? -1 : remapped[i];
  int32_t slot = Resident ? remapped[i] : early[i];
  // Unlike ordinary post-fill alpha refresh, inactive routes MUST NOT read
  // slot zero: this layer may be filling it concurrently.
  if (slot < 0 || slot >= capacity) return;
  for (int32_t j = 0; j < i; ++j)
    if ((Resident ? remapped[j] : early[j]) == slot) return;
  output1[slot] = alpha1[slot] * scale1[0];
  output2[slot] = alpha2[slot] * scale2[0];
}

/// Complete complementary FC1 routes and refresh only protected hit alphas.
template <bool Resident = false>
inline void prepare_qwen38_fc1_routes_impl(
    tvm::ffi::TensorView early,
    tvm::ffi::TensorView remapped,
    tvm::ffi::TensorView late,
    tvm::ffi::TensorView alpha1,
    tvm::ffi::TensorView alpha2,
    tvm::ffi::TensorView scale1,
    tvm::ffi::TensorView scale2,
    tvm::ffi::TensorView output1,
    tvm::ffi::TensorView output2,
    const int32_t* plan = nullptr,
    const float* table1 = nullptr,
    const float* table2 = nullptr,
    int32_t experts = 0) {
  using namespace host;
  SymbolicSize M{"tokens"}, K{"top_k"}, C{"capacity"};
  SymbolicDevice device;
  TensorMatcher({M, K}).with_device<kDLCUDA>(device).with_dtype<int32_t>().verify(early).verify(remapped).verify(late);
  TensorMatcher({C})
      .with_device<kDLCUDA>(device)
      .with_dtype<float>()
      .verify(alpha1)
      .verify(alpha2)
      .verify(output1)
      .verify(output2);
  CHECK_HOST(M.unwrap() >= 1 && M.unwrap() <= 3);
  CHECK_HOST(K.unwrap() >= 1 && K.unwrap() <= 32);
  CHECK_HOST(C.unwrap() >= 1 && C.unwrap() <= 512);
  for (auto scale : {scale1, scale2}) {
    if (scale.ndim() == 0)
      TensorMatcher({}).with_device<kDLCUDA>(device).with_dtype<float>().verify(scale);
    else
      TensorMatcher({1}).with_device<kDLCUDA>(device).with_dtype<float>().verify(scale);
  }
  LaunchKernel(1, 128, device.unwrap())(
      prepare_fc1_routes_kernel<Resident>,
      static_cast<const int32_t*>(early.data_ptr()),
      static_cast<const int32_t*>(remapped.data_ptr()),
      static_cast<int32_t*>(late.data_ptr()),
      static_cast<const float*>(alpha1.data_ptr()),
      static_cast<const float*>(alpha2.data_ptr()),
      static_cast<const float*>(scale1.data_ptr()),
      static_cast<const float*>(scale2.data_ptr()),
      static_cast<float*>(output1.data_ptr()),
      static_cast<float*>(output2.data_ptr()),
      static_cast<int32_t>(M.unwrap() * K.unwrap()),
      static_cast<int32_t>(C.unwrap()),
      plan,
      table1,
      table2,
      experts);
}

inline void prepare_qwen38_fc1_routes(
    tvm::ffi::TensorView early,
    tvm::ffi::TensorView remapped,
    tvm::ffi::TensorView late,
    tvm::ffi::TensorView alpha1,
    tvm::ffi::TensorView alpha2,
    tvm::ffi::TensorView scale1,
    tvm::ffi::TensorView scale2,
    tvm::ffi::TensorView output1,
    tvm::ffi::TensorView output2) {
  prepare_qwen38_fc1_routes_impl(early, remapped, late, alpha1, alpha2, scale1, scale2, output1, output2);
}

/// Resident alpha rows are excluded from concurrent fill layouts. Scatter the
/// planner's miss rows and prepare every routed global in the same CTA.
inline void prepare_qwen38_fc1_resident_routes(
    tvm::ffi::TensorView early,
    tvm::ffi::TensorView remapped,
    tvm::ffi::TensorView late,
    tvm::ffi::TensorView alpha1,
    tvm::ffi::TensorView alpha2,
    tvm::ffi::TensorView scale1,
    tvm::ffi::TensorView scale2,
    tvm::ffi::TensorView output1,
    tvm::ffi::TensorView output2,
    tvm::ffi::TensorView plan,
    tvm::ffi::TensorView table1,
    tvm::ffi::TensorView table2) {
  using namespace host;
  SymbolicDevice device;
  SymbolicSize E{"experts"}, P{"plan"}, M{"tokens"}, K{"top_k"};
  TensorMatcher({M, K}).with_device<kDLCUDA>(device).with_dtype<int32_t>().verify(early);
  TensorMatcher({P}).with_device<kDLCUDA>(device).with_dtype<int32_t>().verify(plan);
  TensorMatcher({E}).with_device<kDLCUDA>(device).with_dtype<float>().verify(table1).verify(table2);
  const auto routes = M.unwrap() * K.unwrap();
  CHECK_HOST(P.unwrap() >= 4 + 2 * (routes < alpha1.size(0) ? routes : alpha1.size(0)));
  CHECK_HOST(E.unwrap() > 0 && E.unwrap() <= 512);
  prepare_qwen38_fc1_routes_impl<true>(
      early,
      remapped,
      late,
      alpha1,
      alpha2,
      scale1,
      scale2,
      output1,
      output2,
      static_cast<const int32_t*>(plan.data_ptr()),
      static_cast<const float*>(table1.data_ptr()),
      static_cast<const float*>(table2.data_ptr()),
      static_cast<int32_t>(E.unwrap()));
}
}  // namespace sglang
