"""Trace aggregation must exclude idle/prefill and overlapping GPU work."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from .profile_service import summarize, union_us


class ProfileSummaryTests(unittest.TestCase):
    def test_queued_gpu_work_is_not_cut_off_at_cpu_return(self):
        events = [
            dict(ph="X", cat="user_annotation", name="step[VERIFY bs=1]",
                 ts=0, dur=1000, args={"External id": 9}),
            dict(ph="X", cat="gpu_user_annotation", name="step[VERIFY bs=1]",
                 ts=1000, dur=2000, args={"External id": 9}),
            dict(ph="X", cat="kernel", name="async_compute", ts=2000, dur=500,
                 args={"device": 0}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            result = summarize(path)
        self.assertEqual(result["decode_wall_ms"], 1)
        self.assertEqual(result["gpu_mean_step_ms"], 2)
        self.assertEqual(result["top"]["kernel"][0]["total_ms"], 0.5)
        self.assertEqual(result["gpu_window_source"], "matched GPU annotations")

    def test_union_handles_nested_and_disjoint_intervals(self):
        self.assertEqual(union_us([(2, 8), (1, 9), (12, 15), (8, 11)]), 13)

    def test_decode_clipping_and_gpu_overlap_for_plain_and_gzip(self):
        def event(cat, name, begin, duration, device=0):
            return dict(ph="X", cat=cat, name=name, ts=begin, dur=duration,
                        args={"device": device})
        events = [
            event("user_annotation", "scheduler.idle", 0, 100000),
            event("user_annotation", "step[EXTEND bs=1]", 0, 1000),
            event("user_annotation", "step[VERIFY bs=1]", 1000, 1000),
            event("user_annotation", "step[VERIFY bs=1]", 3000, 1000),
            event("kernel", "prefill", 0, 900),
            event("kernel", "compute", 900, 900),
            event("gpu_memcpy", "copy", 1500, 1000),
            event("kernel", "compute", 3500, 1000),
            event("kernel", "another-device", 3500, 1000, device=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for suffix, opener in ((".json", open), (".json.gz", gzip.open)):
                path = Path(directory) / ("trace" + suffix)
                with opener(path, "wt") as stream:
                    json.dump({"traceEvents": events}, stream)
                result = summarize(path)
                self.assertEqual(result["decode_steps"], 2)
                self.assertEqual(result["decode_wall_ms"], 2)
                self.assertEqual(result["gpu_busy_ms_by_device"], {"0": 1.5, "1": 0.5})
                compute = next(r for r in result["top"]["kernel"] if r["name"] == "compute")
                self.assertEqual(compute["total_ms"], 1.3)
                self.assertEqual(compute["calls"], 2)
                self.assertNotIn("prefill", str(result["top"]))

    def test_missing_decode_spans_is_not_a_misleading_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text('{"traceEvents": []}')
            with self.assertRaisesRegex(ValueError, "No complete"):
                summarize(path)
