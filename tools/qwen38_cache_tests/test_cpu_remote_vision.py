"""Remote encoder boundaries, no GPU/weights/HTTP services required."""

import ast
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT, load_source


class TestRemoteVision(unittest.TestCase):
    def test_offload_mode_selects_remote_vision_without_local_tower(self):
        def declare(args, source, **values):
            args.__dict__.update(values)
        with mock.patch.dict("sys.modules", {
            "sglang.srt.arg_groups.overrides": SimpleNamespace(
                declare_resolution=declare, resolving_view=lambda args: args),
        }):
            hook = load_source("qwen38_remote_vision_hook_test",
                "python/sglang/srt/arg_groups/qwen38_cache_hook.py")
        for remote in (False, True):
            args = SimpleNamespace(load_format="qwen38_nvfp4_cache", language_only=remote,
                encoder_only=False, language_model_only=False,
                enable_adaptive_dispatch_to_encoder=False, enable_multimodal=None,
                tp_size=1, dp_size=1, ep_size=1, pp_size=1,
                enable_waterfill=False, enforce_shared_experts_fusion=False,
                moe_a2a_backend="none", cpu_offload_gb=0, offload_group_size=0,
                speculative_algorithm=None)
            hook.handle_qwen38_nvfp4_cache(args)
            self.assertEqual(args.language_model_only, not remote)
            self.assertTrue(args.ple_offload_embedding)
            self.assertEqual(args.max_running_requests, 1)
            for field in ("encoder_only", "enable_adaptive_dispatch_to_encoder") if remote else ("encoder_only",):
                setattr(args, field, True)
                with self.assertRaises(ValueError):
                    hook.handle_qwen38_nvfp4_cache(args)
                setattr(args, field, False)

    def test_production_vision_constructor_does_not_allocate_on_remote_worker(self):
        source = (REPO_ROOT / "python/sglang/srt/models/qwen3_vl.py").read_text()
        cls = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLForConditionalGeneration")
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        start = next(i for i, node in enumerate(init.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Attribute) and target.attr == "language_model_only"
                             for target in node.targets))
        statements = ast.Module(body=init.body[start:start + 3], type_ignores=[])
        for remote in (False, True):
            visual = mock.Mock(return_value=object())
            model = SimpleNamespace(use_data_parallel=False)
            config = SimpleNamespace(language_model_only=False,
                skip_local_encoder_init=remote, vision_config=object())
            exec(compile(statements, "vision-constructor", "exec"), {
                "self": model, "config": config, "prefix": "",
                "Qwen3VLMoeVisionModel": visual, "add_prefix": lambda *args: "visual",
            })
            if remote:
                visual.assert_not_called()
                self.assertIsNone(model.visual)
            else:
                visual.assert_called_once()

    def test_raw_pixels_fail_on_remote_worker(self):
        source = (REPO_ROOT / "python/sglang/srt/models/qwen3_vl.py").read_text()
        cls = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLForConditionalGeneration")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_visual_feature")
        guard = ast.Module(body=[method.body[0]], type_ignores=[])
        with self.assertRaisesRegex(RuntimeError, "encoder server"):
            exec(compile(guard, "visual-feature-guard", "exec"), {
                "self": SimpleNamespace(skip_local_encoder_init=True)})
