"""Run production sampler methods with stand-ins; no torch/weights required."""

import ast
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT


class TestSamplerSync(unittest.TestCase):
    def setUp(self):
        source = (REPO_ROOT / "python/sglang/srt/layers/sampler.py").read_text()
        sampler = next(node for node in ast.parse(source).body
                       if isinstance(node, ast.ClassDef) and node.name == "Sampler")
        methods = {node.name: node for node in sampler.body
                   if isinstance(node, ast.FunctionDef)}
        self.reduce = mock.Mock()
        self.group = SimpleNamespace(device_group="tp-device", world_size=1)
        self.attn_group = SimpleNamespace(device_group="attn-device", world_size=1)
        self.parallel = SimpleNamespace(tp_group=self.group,
            attn_tp_group=self.attn_group, attn_cp_size=1)
        self.namespace = {
            "torch": SimpleNamespace(Tensor=object,
                distributed=SimpleNamespace(all_reduce=self.reduce)),
            "SamplingBatchInfo": object,
            "SYNC_TOKEN_IDS_ACROSS_TP": False,
            "dist": SimpleNamespace(ReduceOp=SimpleNamespace(MIN="min")),
        }
        node = methods["_sync_token_ids_across_tp"]
        exec(compile(ast.Module(body=[node], type_ignores=[]), source, "exec"),
             self.namespace)
        self.sync = self.namespace[node.name]
        # Execute the actual constructor's group-selection prefix, before the
        # unrelated sampling backend setup. This covers DP attention group width.
        setup = methods["__init__"].body[1:]
        stop = next(i for i, node in enumerate(setup)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Attribute)
                            and target.attr == "rl_on_policy_target"
                            for target in node.targets))
        self.setup = ast.Module(body=setup[:stop], type_ignores=[])

    def initialize(self, dp=False):
        instance = SimpleNamespace()
        namespace = {"self": instance, "get_parallel": lambda: self.parallel,
                     "is_dp_attention_enabled": lambda: dp}
        exec(compile(self.setup, "sampler-group-setup", "exec"), namespace)
        return instance

    def test_grammar_and_forced_sync_skip_single_rank(self):
        instance = self.initialize()
        self.sync(instance, object(), SimpleNamespace(grammars=[object()]))
        self.namespace["SYNC_TOKEN_IDS_ACROSS_TP"] = True
        self.sync(instance, object(), SimpleNamespace(grammars=None))
        self.reduce.assert_not_called()
        self.assertEqual(instance.tp_sync_group, "tp-device")

    def test_multi_rank_retains_collective_and_no_sync_fast_path(self):
        self.group.world_size = 2
        instance = self.initialize()
        ids = object()
        self.sync(instance, ids, SimpleNamespace(grammars=None))
        self.reduce.assert_not_called()
        self.sync(instance, ids, SimpleNamespace(grammars=[object()]))
        self.reduce.assert_called_once_with(ids, op="min", group="tp-device")

    def test_absent_device_group_never_uses_default_group(self):
        self.group.world_size = 2
        self.group.device_group = None
        self.sync(self.initialize(), object(), SimpleNamespace(grammars=[object()]))
        self.reduce.assert_not_called()

    def test_dp_attention_uses_attn_tp_width_not_global_tp(self):
        self.group.world_size = 8
        instance = self.initialize(dp=True)
        self.sync(instance, object(), SimpleNamespace(grammars=[object()]))
        self.reduce.assert_not_called()
        self.assertEqual(instance.tp_sync_group, "attn-device")
        self.attn_group.world_size = 2
        self.sync(self.initialize(dp=True), object(), SimpleNamespace(grammars=[object()]))
        self.assertEqual(self.reduce.call_args.kwargs["group"], "attn-device")
