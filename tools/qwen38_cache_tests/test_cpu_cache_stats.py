import tempfile
import unittest
from pathlib import Path

from ._load_source import load_source

canary_module = load_source(
    "qwen38_canary_cache_stats_under_test",
    "tools/qwen38_cache_tests/full_model_canary.py",
)


class TestCanaryCacheStats(unittest.TestCase):
    def test_extract_cache_stats_reads_latest_json_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                "noise\n"
                "INFO Qwen38ExpertCacheStats[periodic] {\"hits\": 1}\n"
                "INFO Qwen38ExpertCacheStats[final] {\"hits\": 7, \"cache_hit_rate\": 0.875}\n"
            )
            self.assertEqual(
                canary_module._extract_cache_stats(path),
                {"hits": 7, "cache_hit_rate": 0.875},
            )

    def test_extract_decode_throughput_summarizes_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                "Decode batch, gen throughput (token/s): 10.50, #queue-req: 0\n"
                "Decode batch, gen throughput (token/s): 12.25, #queue-req: 0\n"
            )
            self.assertEqual(
                canary_module._extract_decode_throughput(path),
                {
                    "samples": 2,
                    "avg_token_per_second": 11.375,
                    "max_token_per_second": 12.25,
                    "last_token_per_second": 12.25,
                },
            )


if __name__ == "__main__":
    unittest.main()
