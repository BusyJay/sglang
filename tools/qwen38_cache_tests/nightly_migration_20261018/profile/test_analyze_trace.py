import json
from pathlib import Path
import tempfile
import unittest

from analyze_trace import analyze, intersect, merged


class TraceAnalysisTests(unittest.TestCase):
    def test_union_merges_overlap_and_touching_ranges(self):
        self.assertEqual(merged([(4, 6), (0, 3), (2, 4), (9, 9)]), [(0, 6)])

    def test_intersection_preserves_gaps(self):
        self.assertEqual(intersect([(0, 3), (5, 9)], [(2, 7)]), [(2, 3), (5, 7)])

    def test_gpu_annotations_define_window_and_overlap_is_not_added(self):
        events = [
            {'ph': 'X', 'cat': 'user_annotation', 'name': 'step[VERIFY bs=1]',
             'ts': 0, 'dur': 10, 'args': {'External id': 7}},
            {'ph': 'X', 'cat': 'gpu_user_annotation', 'name': 'step[VERIFY bs=1]',
             'ts': 10, 'dur': 10, 'args': {'External id': 7}},
            {'ph': 'X', 'cat': 'kernel', 'name': 'mailbox_fill_kernel', 'ts': 10, 'dur': 8},
            {'ph': 'X', 'cat': 'kernel', 'name': 'DeviceGemmMxfp8', 'ts': 14, 'dur': 6},
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'trace.json'
            trace.write_text(json.dumps({'traceEvents': events}))
            result = analyze(trace)
        self.assertAlmostEqual(result['target_gpu_active_union_ms'], 0.010)
        self.assertAlmostEqual(result['fill_union_ms_per_verify'], 0.008)
        self.assertAlmostEqual(result['fill_overlap_with_other_gpu_ms_per_verify'], 0.004)
        self.assertAlmostEqual(result['fill_without_other_traced_gpu_work_ms_per_verify'], 0.004)
        self.assertAlmostEqual(result['target_gpu_active_fraction'], 1.0)

    def test_missing_gpu_correlation_is_rejected(self):
        events = [{'ph': 'X', 'cat': 'user_annotation', 'name': 'step[VERIFY bs=1]',
                   'ts': 0, 'dur': 10, 'args': {'External id': 7}}]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'trace.json'
            trace.write_text(json.dumps({'traceEvents': events}))
            with self.assertRaisesRegex(ValueError, 'correlated GPU'):
                analyze(trace)


if __name__ == '__main__':
    unittest.main()
