import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source

warmup = load_source("qwen38_startup_warmup_under_test",
    "python/sglang/srt/entrypoints/qwen38_warmup.py")


class Request:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestStartupWarmup(unittest.TestCase):
    def manager(self, context=4096, abort=False, empty=False):
        calls = []

        async def generate(req, _):
            calls.append(req)
            if empty:
                return
            yield {"meta_info": {"finish_reason": {"type": "abort" if abort else "length"}}}

        manager = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [7, 9, 11]),
            model_config=SimpleNamespace(context_len=context),
            generate_request=generate,
        )
        return manager, calls

    def run_warmup(self, manager, mode="null"):
        with mock.patch.dict("sys.modules", {
            "sglang.srt.managers.io_struct": SimpleNamespace(GenerateReqInput=Request)
        }):
            asyncio.run(warmup.warmup_qwen38_offload(mode, manager))

    def test_representative_shapes_force_decode_and_drain_responses(self):
        manager, calls = self.manager()
        self.run_warmup(manager)
        self.assertEqual([len(req.input_ids) for req in calls], [64, 512, 2048])
        for req in calls:
            self.assertTrue(set(req.input_ids) <= {7, 9, 11})
            self.assertEqual(req.sampling_params["max_new_tokens"], 16)
            self.assertTrue(req.sampling_params["ignore_eos"])
            self.assertFalse(req.stream)

    def test_small_context_is_bounded_and_deduplicated(self):
        manager, calls = self.manager(context=32)
        self.run_warmup(manager)
        self.assertEqual([len(req.input_ids) for req in calls], [16])
        self.assertEqual(len(calls[0].input_ids) + calls[0].sampling_params["max_new_tokens"], 32)

    def test_abort_or_empty_response_fails_startup(self):
        for kwargs in ({"abort": True}, {"empty": True}):
            manager, _ = self.manager(**kwargs)
            with self.assertRaises(RuntimeError):
                self.run_warmup(manager)

    def test_invalid_context_and_disaggregation_rejected(self):
        manager, _ = self.manager(context=1)
        with self.assertRaises(ValueError):
            self.run_warmup(manager)
        manager, _ = self.manager()
        with self.assertRaises(ValueError):
            self.run_warmup(manager, "prefill")
