"""Remote ViT replay cache tests without CUDA or model dependencies."""

import ast
import unittest
from array import array
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

from ._load_source import REPO_ROOT, load_source

cache = load_source(
    "encoder_replay_cpu", "python/sglang/srt/disaggregation/encoder/replay_cache.py"
)


class Modality(Enum):
    IMAGE = 1
    VIDEO = 2


class Tensor:
    def __init__(self, size, device="cpu"):
        self.size = size
        self.device = SimpleNamespace(type=device)

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.size)


class TestEncoderReplay(unittest.TestCase):
    def item(self, url="data:image/png;base64,AAA", **kw):
        return dict(url=url, modality=Modality.IMAGE, detail="auto", **kw)

    def test_immutable_identity_and_namespace(self):
        key = cache.replay_key([self.item()], "tenant")
        self.assertEqual(key, cache.replay_key([self.item()], "tenant"))
        self.assertNotEqual(key, cache.replay_key([self.item()], "other"))
        self.assertNotEqual(
            key, cache.replay_key([self.item("data:image/png;base64,BBB")], "tenant")
        )
        item = self.item()
        item["detail"] = "high"
        self.assertNotEqual(key, cache.replay_key([item], "tenant"))
        for url in ["https://example.com/a.png", "/tmp/a.png"]:
            self.assertIsNone(cache.replay_key([self.item(url)]))
        self.assertIsNone(
            cache.replay_key(
                [dict(url="data:image/png;base64,AAA", modality=Modality.VIDEO)]
            )
        )

    def test_no_gpu_retention_and_bounded_cpu_lru(self):
        c = cache.EncoderReplayCache(100)
        c.put("gpu", {"image": Tensor(10, "cuda")}, {})
        self.assertIsNone(c.get("gpu"))
        c.put("a", {"image": Tensor(50)}, {})
        c.put("b", {"image": Tensor(50)}, {})
        self.assertIsNone(c.get("a"))
        self.assertIsNotNone(c.get("b"))
        self.assertLessEqual(c.bytes, 100)
        c.remove("b")
        self.assertEqual(c.bytes, 0)
        c.put("oversize", {"image": Tensor(101)}, {})
        self.assertEqual(c.bytes, 0)

    def test_hit_rebuilds_current_prompt_without_dispatch(self):
        source = REPO_ROOT / "python/sglang/srt/disaggregation/encoder/receiver.py"
        node = next(
            n
            for n in ast.walk(ast.parse(source.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_reuse_encoder_replay"
        )
        import sys
        from unittest.mock import patch

        c = cache.EncoderReplayCache(100)
        items = [self.item()]
        key = cache.replay_key(items)
        c.put(key, {"image": Tensor(10)}, {"grid": [1, 2]})
        processor = Mock()

        def rebuild(prompt, embeddings, **meta):
            self.assertEqual(prompt, "new conversation turn")
            meta["grid"][0] = 99
            return SimpleNamespace(input_ids=[3, 4, 5])

        processor.get_validated_mm_data.side_effect = rebuild
        r = SimpleNamespace(replay_cache=c, tp_size=1, mm_processor=processor)
        req = SimpleNamespace(
            deferred_encoder_items=items,
            cache_salt=None,
            rid="new-id",
            input_text="new conversation turn",
            need_wait_for_mm_inputs=True,
        )
        namespace = dict(
            array=array,
            _select_mm_processor_prompt=lambda obj, p: obj.input_text,
            logger=Mock(),
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        with patch.dict(
            sys.modules, {"sglang.srt.disaggregation.encoder.replay_cache": cache}
        ):
            result = namespace["_reuse_encoder_replay"](r, req)
        self.assertEqual(result, (key, True))
        self.assertEqual(list(req.input_ids), [3, 4, 5])
        self.assertFalse(req.need_wait_for_mm_inputs)
        self.assertEqual(c.get(key)[1]["grid"], [1, 2])
