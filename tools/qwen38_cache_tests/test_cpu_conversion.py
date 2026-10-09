import json
import unittest

from ._load_source import REPO_ROOT, load_source

convert = load_source(
    "qwen38_convert_under_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_pack/convert.py",
)


class TestConversionCoverage(unittest.TestCase):
    def test_all_projection_scales_and_weight_loader_mapping(self):
        index = {
            f"model.language_model.layers.0.mlp.experts.{e}.{projection}.{suffix}": "shard.safetensors"
            for e in range(3)
            for projection in convert.PROJECTIONS
            for suffix in convert.SUFFIXES
        }
        entries = convert.layer_checkpoint_entries(index, 0, 3)
        self.assertEqual(len(entries), 36)
        self.assertEqual(
            entries["model.language_model.layers.0.mlp.experts.2.up_proj.weight_scale"],
            ("shard.safetensors", "w13_weight_scale", "w3", 2),
        )
        del index["model.language_model.layers.0.mlp.experts.1.down_proj.input_scale"]
        with self.assertRaisesRegex(ValueError, "missing"):
            convert.layer_checkpoint_entries(index, 0, 3)

    def test_public_index_audit_covers_every_main_layer(self):
        report = json.loads(
            (
                REPO_ROOT / "flash-next-plan/model-metadata/checkpoint-index-audit.json"
            ).read_text()
        )
        self.assertEqual(len(report["main_layer_coverage"]), 48)
        self.assertTrue(
            all(
                item["found"] == 6144 for item in report["main_layer_coverage"].values()
            )
        )
        self.assertFalse(report["weights_downloaded"])
