"""Selective encoder loading regressions, without torch or model weights."""

import ast
import json
import logging
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT, load_source


weights = load_source(
    "qwen4_encoder_weights_test",
    "python/sglang/srt/model_loader/encoder_weights.py",
)


class TestEncoderWeights(unittest.TestCase):
    def source(self, encoder=True, architecture="Qwen4ExpForConditionalGeneration"):
        return SimpleNamespace(
            prefix="", model_config=SimpleNamespace(hf_config=SimpleNamespace(
                encoder_only=encoder, architectures=[architecture])))

    def reader(self, keys):
        reader = mock.MagicMock()
        reader.__enter__.return_value = reader
        reader.keys.return_value = keys
        reader.get_tensor.side_effect = lambda name: "tensor:" + name
        return reader

    def test_scope_is_only_qwen4_encoder(self):
        self.assertTrue(weights.is_qwen4_encoder_source(self.source()))
        self.assertFalse(weights.is_qwen4_encoder_source(self.source(False)))
        self.assertFalse(weights.is_qwen4_encoder_source(self.source(architecture="OtherVL")))
        self.assertFalse(weights.is_qwen4_encoder_source(SimpleNamespace()))

    def test_index_selects_only_visual_shards_without_opening_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            index = {"weight_map": {
                "model.visual.blocks.0.weight": "mixed.safetensors",
                "visual.merger.weight": "vision.safetensors",
                "model.language_model.layers.0.weight": "language.safetensors"}}
            with open(Path(folder) / "model.safetensors.index.json", "w") as stream:
                json.dump(index, stream)
            files = [os.path.join(folder, name) for name in (
                "language.safetensors", "vision.safetensors", "mixed.safetensors")]
            with mock.patch.dict("sys.modules", {"safetensors": None}):
                self.assertEqual(weights.select_encoder_safetensors(folder, files),
                                 [files[2], files[1]])
            with self.assertRaisesRegex(ValueError, "missing visual shards"):
                weights.select_encoder_safetensors(folder, files[:2])

    def test_unindexed_scans_headers_only(self):
        language = self.reader(["model.language_model.weight"])
        visual = self.reader(["model.visual.weight", "model.language_model.weight"])
        opener = mock.Mock(side_effect=[language, visual])
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
            "sys.modules", {"safetensors": SimpleNamespace(safe_open=opener)}
        ):
            self.assertEqual(weights.select_encoder_safetensors(folder, ["language", "mixed"]), ["mixed"])
        language.get_tensor.assert_not_called()
        visual.get_tensor.assert_not_called()

    def test_no_visual_checkpoint_fails_before_tensor_reads(self):
        reader = self.reader(["model.language_model.weight"])
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
            "sys.modules", {"safetensors": SimpleNamespace(safe_open=mock.Mock(return_value=reader))}
        ):
            with self.assertRaisesRegex(ValueError, "no visual"):
                weights.select_encoder_safetensors(folder, ["language"])
        reader.get_tensor.assert_not_called()

    def test_mixed_shard_reads_only_exact_visual_prefixes(self):
        names = ["model.language_model.weight", "model.visual.blocks.0.weight",
                 "visual.merger.weight", "model.language_model.visual.weight"]
        reader = self.reader(names)
        with mock.patch.dict("sys.modules", {
            "safetensors": SimpleNamespace(safe_open=mock.Mock(return_value=reader))
        }):
            result = list(weights.encoder_safetensors_weights_iterator(["mixed"]))
        self.assertEqual([name for name, _ in result], names[1:3])
        self.assertEqual(reader.get_tensor.call_args_list,
                         [mock.call(names[1]), mock.call(names[2])])

    def test_completeness_guard_reports_missing_visual_parameters(self):
        weights.validate_encoder_weights(["visual.a", "visual.b"], {"visual.a", "visual.b"})
        with self.assertRaisesRegex(ValueError, "1 visual parameters not loaded: visual.b"):
            weights.validate_encoder_weights(["visual.a", "visual.b"], {"visual.a"})

    def test_production_loader_routes_all_safetensor_modes_to_selective_reader(self):
        tree = ast.parse((REPO_ROOT / "python/sglang/srt/model_loader/loader.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "DefaultModelLoader")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_get_weights_iterator")
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args + method.args.kwonlyargs:
            arg.annotation = None
        for mode in ("auto", "safetensors", "fastsafetensors"):
            selective = mock.Mock(return_value=iter([("visual.a", "tensor")]))
            selector = mock.Mock(return_value=["visual-shard"])
            context = {
                "is_qwen4_encoder_source": weights.is_qwen4_encoder_source,
                "select_encoder_safetensors": selector,
                "encoder_safetensors_weights_iterator": selective,
                "logger": logging.getLogger(__name__), "time": time,
            }
            exec(compile(ast.Module(body=[method], type_ignores=[]), "loader", "exec"), context)
            loader = SimpleNamespace(load_config=SimpleNamespace(
                model_loader_extra_config={}, load_format=mode, draft_model_idx=None),
                counter_before_loading_weights=0.0)
            result = context["_get_weights_iterator"](loader, self.source(), resolved_source=SimpleNamespace(
                hf_folder="checkpoint", weight_files=["all-shards"], use_safetensors=True))
            self.assertEqual(list(result), [("visual.a", "tensor")])
            selector.assert_called_once_with("checkpoint", ["all-shards"])
            selective.assert_called_once_with(["visual-shard"])

    def test_production_model_invokes_completeness_guard_only_for_encoder(self):
        tree = ast.parse((REPO_ROOT / "python/sglang/srt/models/qwen4_exp.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "Qwen4ExpForConditionalGeneration")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "load_weights")
        guard = next(node for node in method.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "self.config.encoder_only")
        for encoder in (False, True):
            with mock.patch.dict("sys.modules", {
                "sglang.srt.model_loader.encoder_weights": weights
            }):
                context = {"self": SimpleNamespace(config=SimpleNamespace(encoder_only=encoder)),
                           "params_dict": {"visual.a": None, "model.b": None}, "loaded_params": set()}
                if encoder:
                    with self.assertRaisesRegex(ValueError, "visual.a"):
                        exec(compile(ast.Module(body=[guard], type_ignores=[]), "guard", "exec"), context)
                else:
                    exec(compile(ast.Module(body=[guard], type_ignores=[]), "guard", "exec"), context)

    def test_startup_prefetch_excludes_encoder_even_for_mixed_shards(self):
        tree = ast.parse((REPO_ROOT / "python/sglang/srt/model_loader/loader.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "DefaultModelLoader")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "start_checkpoint_prefetch")
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args + method.args.kwonlyargs:
            arg.annotation = None
        prefetch = mock.Mock()
        context = {"is_qwen4_encoder_source": weights.is_qwen4_encoder_source,
                   "_prefetch_all_checkpoints": prefetch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "prefetch", "exec"), context)
        encoder = SimpleNamespace(source=self.source(), use_safetensors=True,
                                  weight_files=["mixed-visual-language-shard"])
        language = SimpleNamespace(source=self.source(False), use_safetensors=True,
                                   weight_files=["language-shard"])
        context["start_checkpoint_prefetch"]((encoder, language), num_threads=4)
        prefetch.assert_called_once_with(["language-shard"], num_threads=4)
