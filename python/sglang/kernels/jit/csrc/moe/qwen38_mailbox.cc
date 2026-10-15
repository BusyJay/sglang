// SPDX-License-Identifier: Apache-2.0
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>

#include <Python.h>
#include <chrono>
using tvm::ffi::TensorView;

struct Qwen38MailboxPoll {
  static int64_t run(TensorView heads, TensorView seen) {
    TVM_FFI_ICHECK_EQ(heads.device().device_type, kDLCPU);
    TVM_FFI_ICHECK_EQ(seen.device().device_type, kDLCPU);
    TVM_FFI_ICHECK_EQ(heads.ndim(), 1);
    TVM_FFI_ICHECK_EQ(seen.ndim(), 1);
    TVM_FFI_ICHECK_EQ(heads.dtype().code, kDLInt);
    TVM_FFI_ICHECK_EQ(heads.dtype().bits, 32);
    TVM_FFI_ICHECK_EQ(heads.dtype().lanes, 1);
    TVM_FFI_ICHECK_EQ(seen.dtype().code, kDLInt);
    TVM_FFI_ICHECK_EQ(seen.dtype().bits, 32);
    TVM_FFI_ICHECK_EQ(seen.dtype().lanes, 1);
    TVM_FFI_ICHECK_EQ(heads.size(0), seen.size(0));
    auto* h = static_cast<const int32_t*>(heads.data_ptr());
    auto* s = static_cast<const int32_t*>(seen.data_ptr());
    // Python must be able to enqueue compute/transport while this thread polls.
    PyThreadState* state = PyGILState_Check() ? PyEval_SaveThread() : nullptr;
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    int64_t result = -1;
    do {
      for (int64_t i = 0; i < heads.size(0); ++i) {
        if (__atomic_load_n(h + i, __ATOMIC_ACQUIRE) != s[i]) {
          result = i;
          break;
        }
      }
      if (result >= 0) break;
#if defined(__x86_64__)
      __builtin_ia32_pause();
#endif
    } while (std::chrono::steady_clock::now() < end);
    if (state) PyEval_RestoreThread(state);
    return result;
  }
};
