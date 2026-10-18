"""Diagnostic launcher hook. Enable only in isolated, non-performance runs.

Put this directory on PYTHONPATH and set QWEN38_AUDIT_OUTPUT. The existing
/start_profile MEM and /stop_profile API captures idle process ownership alongside
allocator snapshots. No inference, weight-loading, or allocator policy is changed.
"""

import os

if os.environ.get("QWEN38_AUDIT_OUTPUT"):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "qwen38_memory_audit_capture", Path(__file__).parent.parent / "capture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()

    # Keep full allocation history, but avoid hundreds of redundant startup
    # snapshots. This diagnostic hook only changes evidence sampling.
    from sglang.srt.layers.moe.qwen38_startup_memory import StartupMemoryProbe

    original_mark = StartupMemoryProbe.mark

    def selected_mark(self, stage, **kwargs):
        if stage in ("probe.created", "kv.budget.after_empty_cache",
                     "runner.graph.after_capture"):
            return original_mark(self, stage, **kwargs)

    StartupMemoryProbe.mark = selected_mark
