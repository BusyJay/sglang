# SPDX-License-Identifier: Apache-2.0
"""CUDA driver stream memory operations; no host wait or SM spin kernel."""

import ctypes
import ctypes.util


class StreamMemoryOps:
    def __init__(self):
        path = ctypes.util.find_library("cuda") or "libcuda.so.1"
        self.driver = ctypes.CDLL(path)
        self.wait = self._function("cuStreamWaitValue32")
        self.write = self._function("cuStreamWriteValue32")

    def _function(self, name):
        function = getattr(self.driver, name + "_v2", None)
        if function is None:
            function = getattr(self.driver, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        return function

    def _call(self, function, stream, tensor, value, flags):
        result = function(stream.cuda_stream, tensor.data_ptr(), value, flags)
        if result:
            raise RuntimeError(
                f"CUDA stream memory operation failed ({result}); threaded runtime requires driver MemOps support"
            )

    def wait_ready(self, stream, gate):
        # GEQ 1 releases both successful (1) and failed (2) jobs. A guard kernel
        # turns failed transfer into a CUDA assertion, never stale MoE output.
        self._call(self.wait, stream, gate, 1, 0)

    def signal(self, stream, gate, failed=False):
        # Default barrier orders prior H2D writes before the ready signal.
        self._call(self.write, stream, gate, 2 if failed else 1, 0)
