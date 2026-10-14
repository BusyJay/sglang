import unittest

from .benchmark.core import StreamMeasurement
from .mtp_throughput import decode_metrics


class TestMtpThroughput(unittest.TestCase):
    def test_coalesced_first_chunk_is_excluded_from_decode_rate(self):
        stream = StreamMeasurement()
        meta = {
            "spec_verify_ct": 4,
            "spec_num_correct_drafts": 6,
            "spec_num_proposed_drafts": 8,
        }
        stream.feed({"output_ids": [1, 2, 3], "meta_info": meta}, 0.2)
        stream.feed({"output_ids": list(range(1, 9)), "meta_info": meta}, 0.7)
        value = decode_metrics(
            {
                "events": stream.events,
                "ttft_s": 0.2,
                "output_tokens": 8,
                "duration_s": 1,
            }
        )
        self.assertEqual(value["decode_tokens"], 5)
        self.assertAlmostEqual(value["decode_tok_s"], 10)
        self.assertAlmostEqual(value["tpot_ms"], 100)
        self.assertEqual(value["mean_verify_tokens"], 2.5)
        self.assertEqual(value["accept_rate"], 0.75)

    def test_single_token_bearing_event_cannot_establish_decode_rate(self):
        with self.assertRaisesRegex(ValueError, "two token-bearing"):
            decode_metrics(
                {
                    "events": [{"new_tokens": 4}],
                    "ttft_s": 0.1,
                    "output_tokens": 4,
                    "duration_s": 1,
                }
            )
