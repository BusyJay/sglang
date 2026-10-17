"""Expert source file-cache retention policy regression."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source

pack = load_source(
    "expert_discard_policy_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_pack/format.py",
)


class ExpertCacheRetentionTests(unittest.TestCase):
    def test_expert_cache_discard_switch(self):
        obj = SimpleNamespace(
            _cuda_source_private=True, _stream=SimpleNamespace(fileno=lambda: 123)
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(os, "posix_fadvise") as advise,
        ):
            pack.Qwen38ExpertPack.discard_source_file_cache(obj)
            advise.assert_not_called()
        for flag, called in [("0", False), ("1", True)]:
            with (
                mock.patch.dict(
                    os.environ, {"SGLANG_QWEN38_DISCARD_SOURCE_FILE_CACHE": flag}
                ),
                mock.patch.object(os, "posix_fadvise") as advise,
            ):
                pack.Qwen38ExpertPack.discard_source_file_cache(obj)
                self.assertEqual(advise.called, called)
        with (
            mock.patch.dict(
                os.environ, {"SGLANG_QWEN38_DISCARD_SOURCE_FILE_CACHE": "bad"}
            ),
            self.assertRaises(ValueError),
        ):
            pack.Qwen38ExpertPack.discard_source_file_cache(obj)


if __name__ == "__main__":
    unittest.main()
