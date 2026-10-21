"""Opt-in NVTX probes for the handoff capture; never imported by serving code."""

import functools
import sys


def install():
    import numpy as np
    import nvtx
    import torch
    from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4
    from sglang.srt.layers.moe.qwen38_cache_mailbox import DecodeMissMailbox
    from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
    from sglang.srt.layers.moe.qwen38_hybrid import HybridCache

    if getattr(HybridCache, "_handoff_probes_installed", False):
        return
    HybridCache._handoff_probes_installed = True
    from concurrent.futures import Future

    def scope(name):
        return nvtx.annotate(name, domain="qwen38_handoff")

    original_result = Future.result

    @functools.wraps(original_result)
    def result(future, *args, **kwargs):
        caller = sys._getframe(1).f_code
        if caller.co_filename.endswith("/qwen38_hybrid.py") and caller.co_name == "run":
            state = "ready" if future.done() else "pending"
            with scope(f"hybrid.result_join.{state}"):
                return original_result(future, *args, **kwargs)
        return original_result(future, *args, **kwargs)

    Future.result = result
    original_sync = torch.cuda.Event.synchronize

    @functools.wraps(original_sync)
    def synchronize(event):
        caller = sys._getframe(1).f_code
        if (
            caller.co_filename.endswith("/qwen38_hybrid.py")
            and caller.co_name == "_cpu_work"
        ):
            with scope("hybrid.cpu.input_wait"):
                return original_sync(event)
        return original_sync(event)

    torch.cuda.Event.synchronize = synchronize
    original_cpu = torch.Tensor.cpu

    @functools.wraps(original_cpu)
    def cpu(tensor, *args, **kwargs):
        caller = sys._getframe(1).f_code
        if caller.co_filename.endswith("/qwen38_hybrid.py") and caller.co_name in (
            "prefill",
            "run",
        ):
            with scope(f"hybrid.{caller.co_name}.ids_readback"):
                return original_cpu(tensor, *args, **kwargs)
        return original_cpu(tensor, *args, **kwargs)

    torch.Tensor.cpu = cpu
    original_init = CpuNvfp4.__init__

    @functools.wraps(original_init)
    def initialize(operator, *args, **kwargs):
        original_init(operator, *args, **kwargs)
        native = operator.native_run

        def native_run(*native_args):
            with scope("hybrid.cpu.native"):
                return native(*native_args)

        operator.native_run = native_run

    CpuNvfp4.__init__ = initialize
    original_run = CpuNvfp4.run

    @functools.wraps(original_run)
    def run(operator, layer, x, ids, weights, output):
        routes = ids.numpy()
        experts = len(np.unique(routes[routes >= 0]))
        with scope(f"hybrid.cpu/M={x.shape[0]}/E={experts}"):
            return original_run(operator, layer, x, ids, weights, output)

    CpuNvfp4.run = run

    original_snapshot = Qwen38ThreadedExpertRuntime.stats_snapshot

    @functools.wraps(original_snapshot)
    def snapshot(runtime, *args, **kwargs):
        with scope("hybrid.stats_snapshot"):
            return original_snapshot(runtime, *args, **kwargs)

    Qwen38ThreadedExpertRuntime.stats_snapshot = snapshot
    original_flush = DecodeMissMailbox.flush_stats

    @functools.wraps(original_flush)
    def flush(mailbox, *args, **kwargs):
        with scope("hybrid.mailbox.flush_stats"):
            return original_flush(mailbox, *args, **kwargs)

    DecodeMissMailbox.flush_stats = flush
