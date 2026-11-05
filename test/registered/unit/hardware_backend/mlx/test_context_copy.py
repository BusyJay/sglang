# SPDX-License-Identifier: Apache-2.0
"""Consumed context, prefix isolation and bounded context-copy proposals."""

import unittest

from sglang.srt.hardware_backend.mlx.context_copy import ContextCopyProposer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestContextCopy(CustomTestCase):
    def test_complete_suffix_match_and_recent_complete_continuation(self):
        proposer = ContextCopyProposer()
        span = list(range(8))
        proposer.extend(span + [91, 92, 93] + [20] + span + [71, 72, 73] + span[:-1])
        self.assertEqual(proposer.propose(7, 3), (71, 72, 73))
        self.assertEqual(proposer.propose(99, 3), ())
        self.assertEqual(proposer.propose(7, 0), ())
        self.assertEqual(proposer.propose(7, 4), (71, 72, 73, 0))

    def test_prefix_snapshots_are_immutable_across_page_boundaries(self):
        proposer = ContextCopyProposer()
        proposer.extend(range(63))
        snapshot = proposer.snapshot
        proposer.extend([63, 64, 65])
        restored = ContextCopyProposer(snapshot)
        restored.extend([100, 101])
        self.assertEqual(restored.tokens, list(range(63)) + [100, 101])
        self.assertEqual(proposer.tokens, list(range(66)))
        self.assertIsNone(snapshot.page)
        self.assertEqual(len(snapshot.tail), 63)
        second = ContextCopyProposer(proposer.snapshot)
        self.assertEqual(second.tokens, proposer.tokens)
        second.extend([102])
        self.assertEqual(proposer.tokens, list(range(66)))

    def test_anchor_and_unverified_candidates_do_not_enter_history(self):
        proposer = ContextCopyProposer(ngram_size=3)
        proposer.extend([1, 2, 3, 4, 5, 9, 1, 2])
        snapshot = proposer.snapshot
        self.assertEqual(proposer.propose(3, 2), (4, 5))
        self.assertEqual(proposer.propose(3, 2), (4, 5))
        self.assertIs(proposer.snapshot, snapshot)
        proposer.extend([3, 99])
        self.assertEqual(proposer.tokens[-2:], [3, 99])


if __name__ == "__main__":
    unittest.main()
