from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from ._load_source import load_source

format_module = load_source(
    "qwen38_pack_format_for_builder_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_pack/format.py",
)
sys.modules["format"] = format_module
builder_module = load_source(
    "qwen38_pack_builder_under_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_pack/build.py",
)


class TestPackBuilder(unittest.TestCase):
    def test_build_fixture_pack(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "source"
            layer = source / "layer-000"
            layer.mkdir(parents=True)
            metadata = {}
            for index, name in enumerate(format_module.PACK_TENSORS):
                filename = f"{name}.bin"
                (layer / filename).write_bytes(
                    bytes([index + 1])
                    * (4 if name in format_module.LAYER_TENSORS else 8)
                )
                metadata[name] = {
                    "file": filename,
                    "shape": [1] if name in format_module.LAYER_TENSORS else [2, 4],
                    "dtype": (
                        "float32" if name in format_module.LAYER_TENSORS else "uint8"
                    ),
                }
            (layer / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            payload = root / "runtime.bin"
            manifest_path = root / "runtime.json"
            builder_module.build_pack(
                source,
                payload,
                manifest_path,
                checkpoint_revision="fixture",
                sglang_commit="baseline",
                num_layers=1,
                num_experts=2,
                top_k=2,
                hidden_size=2560,
                intermediate_size=640,
                alignment=16,
            )
            manifest = format_module.load_manifest(manifest_path)
            offsets = [
                manifest.tensors[0][name].offset for name in format_module.PACK_TENSORS
            ]
            self.assertTrue(all(offset % 16 == 0 for offset in offsets))
            with format_module.Qwen38ExpertPack(manifest) as pack:
                self.assertEqual(
                    bytes(pack.tensor_bytes(0, "g2_alphas")), bytes([6]) * 8
                )


if __name__ == "__main__":
    unittest.main()
