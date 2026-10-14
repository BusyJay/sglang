"""MTP loading and opt-in boundaries without GPU/runtime dependencies."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT, load_source

common = load_source(
    "qwen38_mtp_common_test", "python/sglang/srt/model_loader/qwen38_mtp_common.py"
)


class TestMtpLoading(unittest.TestCase):
    def test_sharing_binds_complete_modules_without_mutating_target(self):
        try:
            import torch
        except ImportError:
            self.skipTest("module sharing test requires PyTorch")
        source = ast.parse(
            (REPO_ROOT / "python/sglang/srt/models/qwen4_exp_mtp.py").read_text()
        )
        cls = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpForCausalLMMTP"
        )
        methods = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("bind_qwen38_target_modules", "set_embed_and_head")
        ]
        namespace = {}
        exec(
            compile(
                ast.Module(body=methods, type_ignores=[]), "module-sharing", "exec"
            ),
            namespace,
        )
        target = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 4)),
            lm_head=torch.nn.Linear(4, 8, bias=False),
        )
        target.lm_head.quant_method = object()
        draft = SimpleNamespace(
            _deferred_shared_modules=True,
            fc_hidden=torch.nn.Linear(4, 4),
            model=SimpleNamespace(embed_tokens=None),
            lm_head=None,
        )
        with self.assertRaisesRegex(RuntimeError, "have not been bound"):
            namespace["set_embed_and_head"](draft, None, None)
        namespace["bind_qwen38_target_modules"](draft, target)
        self.assertIs(draft.model.embed_tokens, target.model.embed_tokens)
        self.assertIs(draft.lm_head, target.lm_head)
        self.assertIs(draft.lm_head.quant_method, target.lm_head.quant_method)
        embed, head = target.model.embed_tokens.weight, target.lm_head.weight
        namespace["set_embed_and_head"](draft, embed, head)
        self.assertIs(target.lm_head.weight, head)
        with self.assertRaisesRegex(ValueError, "complete target vocabulary"):
            namespace["set_embed_and_head"](draft, embed, head.clone())

    def test_index_selects_only_mtp_shards_and_requires_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mtp.safetensors").touch()
            mapping = {
                name: "mtp.safetensors" for name in common.required_expert_tensors()
            }
            mapping["model.layers.0.weight"] = "absent-target.safetensors"
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": mapping})
            )
            selected, files = common.mtp_checkpoint_files(root)
            self.assertEqual(len(selected), 3072)
            self.assertEqual(files, [str(root / "mtp.safetensors")])
            selected.pop("mtp.layers.0.mlp.experts.511.down_proj.weight_scale_inv")
            with mock.patch.dict(
                "sys.modules", {"safetensors": SimpleNamespace(safe_open=mock.Mock())}
            ):
                with self.assertRaisesRegex(ValueError, "missing MTP expert"):
                    list(common.mtp_weights_iterator(selected, files))

    def test_tensor_shape_dtype_contract(self):
        name = "mtp.layers.0.mlp.experts.3.gate_proj.weight"
        common.validate_mtp_tensor(name, [640, 2560], "torch.float8_e4m3fn")
        common.validate_mtp_tensor(name + "_scale_inv", [5, 20], "torch.bfloat16")
        for shape, dtype in (
            ([2560, 640], "torch.float8_e4m3fn"),
            ([640, 2560], "torch.bfloat16"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid MTP tensor"):
                common.validate_mtp_tensor(name, shape, dtype)

    def test_filter_does_not_read_target_or_ple_payload(self):
        name = "mtp.fc_hidden.weight"
        handle = mock.MagicMock()
        handle.__enter__.return_value = handle
        handle.keys.return_value = [
            name,
            "model.ple.weight",
            "model.embed_tokens.weight",
        ]
        handle.get_tensor.return_value = SimpleNamespace(
            shape=[2560, 2560], dtype="torch.bfloat16"
        )
        with (
            mock.patch.object(common, "required_expert_tensors", return_value=set()),
            mock.patch.dict(
                "sys.modules",
                {
                    "safetensors": SimpleNamespace(
                        safe_open=mock.Mock(return_value=handle)
                    )
                },
            ),
        ):
            result = list(
                common.mtp_weights_iterator(
                    {name: "mtp.safetensors"}, ["/fixture/mtp.safetensors"]
                )
            )
        handle.get_tensor.assert_called_once_with(name)
        self.assertEqual(result[0][0], name)

    def test_actual_writes_required_and_loader_restored(self):
        source = ast.parse(
            (
                REPO_ROOT / "python/sglang/srt/model_loader/qwen38_mtp_loader.py"
            ).read_text()
        )
        fn = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_mtp_weights_strict"
        )
        namespace = {"default_weight_loader": lambda p, tensor: None}
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "strict-loader", "exec"),
            namespace,
        )
        strict = namespace[fn.name]
        param = SimpleNamespace(weight_loader=mock.Mock())
        original = param.weight_loader
        model = SimpleNamespace(named_parameters=lambda: [("fc_hidden.weight", param)])

        def load(weights):
            for _, tensor in weights:
                param.weight_loader(param, tensor)

        model.load_weights = load
        strict(model, [("mtp.fc_hidden.weight", object())])
        self.assertIs(param.weight_loader, original)
        model.load_weights = lambda weights: list(weights)
        with self.assertRaisesRegex(ValueError, "was not loaded"):
            strict(model, [("mtp.fc_hidden.weight", object())])
        self.assertIs(param.weight_loader, original)
        with self.assertRaisesRegex(ValueError, "left uninitialized"):
            strict(model, [])

    def test_stacked_parameter_missing_shard_fails(self):
        source = ast.parse(
            (
                REPO_ROOT / "python/sglang/srt/model_loader/qwen38_mtp_loader.py"
            ).read_text()
        )
        fn = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_mtp_weights_strict"
        )
        namespace = {"default_weight_loader": lambda *args: None}
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "strict-loader", "exec"),
            namespace,
        )
        param = SimpleNamespace(weight_loader=mock.Mock())

        def load(weights):
            for _, tensor in weights:
                param.weight_loader(param, tensor, "q")

        model = SimpleNamespace(
            named_parameters=lambda: [("model.layers.0.qkv_proj.weight", param)],
            load_weights=load,
        )
        with self.assertRaisesRegex(ValueError, "slices left uninitialized"):
            namespace[fn.name](
                model, [("mtp.layers.0.self_attn.q_proj.weight", object())]
            )


class TestMtpConfig(unittest.TestCase):
    def setUp(self):
        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.arg_groups.overrides": SimpleNamespace(
                    resolving_view=lambda args: args,
                    declare_resolution=lambda args, source, **values: (
                        args.__dict__.update(values)
                    ),
                )
            },
        ):
            self.hook = load_source(
                "qwen38_mtp_hook_test",
                "python/sglang/srt/arg_groups/qwen38_cache_hook.py",
            )
        load_source(
            "sglang.srt.layers.moe.qwen38_cache_capacity",
            "python/sglang/srt/layers/moe/qwen38_cache_capacity.py",
        )

    def args(self, **values):
        args = SimpleNamespace(
            load_format="qwen38_nvfp4_cache",
            language_only=False,
            encoder_only=False,
            language_model_only=True,
            enable_adaptive_dispatch_to_encoder=False,
            enable_multimodal=None,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            pp_size=1,
            enable_waterfill=False,
            enforce_shared_experts_fusion=False,
            moe_a2a_backend="none",
            cpu_offload_gb=0,
            offload_group_size=0,
            speculative_algorithm="NEXTN",
            speculative_draft_load_format="qwen38_mtp",
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
            model_path="target",
            revision="fixed",
            model_loader_extra_config={"runtime_mode": "threaded"},
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=None,
            speculative_moe_runner_backend=None,
        )
        args.__dict__.update(values)
        return args

    def test_target_only_does_not_enable_mtp(self):
        args = self.args(speculative_algorithm=None, speculative_draft_load_format=None)
        self.hook.handle_qwen38_nvfp4_cache(args)
        self.assertIsNone(args.speculative_algorithm)
        self.assertIsNone(args.speculative_draft_model_path)

    def test_chain_defaults_and_target_identity(self):
        for steps in (None, 1, 2, 3):
            args = self.args(speculative_num_steps=steps)
            self.hook.handle_qwen38_nvfp4_cache(args)
            self.assertEqual(args.speculative_num_draft_tokens, (steps or 1) + 1)
            self.assertEqual(args.speculative_draft_model_path, "target")
            self.assertEqual(args.speculative_draft_model_revision, "fixed")
            self.assertEqual(args.speculative_moe_runner_backend, "triton")
            self.assertEqual(args.speculative_draft_kv_cache_dtype, "bfloat16")
            self.assertEqual(args.mamba_ssm_dtype, "bfloat16")

    def test_unsupported_combinations_fail_before_loading(self):
        for values in (
            {"tp_size": 2},
            {"speculative_algorithm": "EAGLE3"},
            {"speculative_draft_load_format": None},
            {"speculative_draft_model_path": "other"},
            {"speculative_draft_model_revision": "other"},
            {"speculative_num_steps": 4},
            {"speculative_eagle_topk": 2},
            {"speculative_num_draft_tokens": 4},
            {"speculative_moe_runner_backend": "deep_gemm"},
            {"disable_overlap_schedule": True},
            {
                "model_loader_extra_config": {
                    "runtime_mode": "threaded",
                    "adaptive_cache": True,
                    "cache_slots_total": 480,
                }
            },
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.hook.handle_qwen38_nvfp4_cache(self.args(**values))

    def test_mtp_adaptive_total_budget(self):
        for steps in (1, 2, 3):
            args = self.args(
                speculative_num_steps=steps,
                model_loader_extra_config={
                    "runtime_mode": "threaded",
                    "adaptive_cache": True,
                    "cache_slots_total": 3841,
                },
            )
            self.hook.handle_qwen38_nvfp4_cache(args)
            self.assertTrue(args.model_loader_extra_config["adaptive_cache"])
