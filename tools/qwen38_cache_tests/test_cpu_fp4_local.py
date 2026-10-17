"""Checkpoint format tests independent of a GPU and full model weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from ._load_source import load_source

adapter = load_source(
    "fp4_local_adapter_test", "python/sglang/srt/model_loader/qwen38_fp4_local.py"
)
ple = load_source(
    "fp4_local_ple_index_test", "python/sglang/srt/models/qwen4_exp_ple_lazy.py"
)
mtp = load_source(
    "fp4_local_mtp_contract_test", "python/sglang/srt/model_loader/qwen38_mtp_common.py"
)


class TestFp4Local(unittest.TestCase):
    def test_ple_global_scale_alias_preserves_fp32_and_missing_scale_fails(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
        from sglang.srt.models.qwen38_fp4_ple import Qwen38Fp4PleEmbedding

        prefix = "model.layers.1.ple.ple_embedding.ngram_embedding."
        target = torch.full((1,), float("nan"), dtype=torch.float32)
        value = torch.tensor([3.324236240587197e-5], dtype=torch.float32)
        loaded = set()
        self.assertTrue(
            Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer(
                None,
                prefix + "weight_scale_2",
                value,
                {prefix + "fp4_global_scale": target},
                loaded,
            )
        )
        self.assertTrue(torch.equal(target, value))
        self.assertEqual(loaded, {prefix + "fp4_global_scale"})
        with self.assertRaisesRegex(ValueError, "weight_scale_2"):
            Qwen38Fp4PleEmbedding.finalize_lazy(
                SimpleNamespace(fp4_global_scale=torch.tensor([float("nan")]))
            )

    def test_pinned_full_table_and_lazy_are_distinct_backends(self):
        from sglang.srt.models.qwen38_fp4_ple import Qwen38Fp4PleEmbedding
        from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
        def metadata(obj, embedding, **kwargs):
            torch.nn.Module.__init__(obj)
            obj.weight = torch.nn.Parameter(torch.empty(7, 160, device="meta"), requires_grad=False)
            obj.embedding_dim, obj.org_vocab_size = 160, 7
            obj._checkpoint_rows, obj._cache_tables = 7, 1
            obj._lazy_shards, obj._lazy_cache = [], None
            obj._file_prefetcher = obj._file_rss_trimmer = None
            obj.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=7)
        with mock.patch.object(Qwen4ExpPinnedHostEmbedding, "__init__", metadata):
            pinned = Qwen38Fp4PleEmbedding(SimpleNamespace(weight_scale=torch.ones(1)))
            lazy = Qwen38Fp4PleEmbedding(SimpleNamespace(weight_scale=torch.ones(1)), backend="lazy")
        self.assertFalse(pinned._lazy)
        self.assertTrue(lazy._lazy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ple.safetensors"
            weight = torch.arange(560).remainder(256).to(torch.uint8).reshape(7,80)
            scale = torch.full((7,10), .5, dtype=torch.float8_e4m3fn)
            prefix = "x.ngram_embedding.shard_0."
            save_file({prefix+"weight":weight, prefix+"weight_scale":scale},path)
            from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index
            desc = read_ple_checkpoint_index(str(path))
            pinned.load_checkpoint_part(0,"weight",desc[prefix+"weight"],1)
            pinned.load_checkpoint_part(0,"weight_scale",desc[prefix+"weight_scale"],1)
            pinned.fp4_global_scale.fill_(.25)
            with mock.patch("sglang.srt.models.qwen38_fp4_ple.allocate_ple_host_table", side_effect=lambda **kw: torch.empty(kw["shape"],dtype=kw["dtype"])):
                pinned.finalize_lazy()
            self.assertTrue(pinned._packed_ready)
            self.assertIsNone(pinned._lazy_cache)
            self.assertEqual(tuple(pinned.weight.shape),(7,90))
            self.assertTrue(torch.equal(pinned.weight[:,:80],weight))
            self.assertTrue(torch.equal(pinned.weight[:,80:],scale.view(torch.uint8)))

    def test_mtp_nested_shards_select_the_indexed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale"
            for parent, value in (("tensors", 0.5), ("other", 1.0)):
                (root / parent).mkdir()
                save_file(
                    {name: torch.full((640, 160), value, dtype=torch.float8_e4m3fn)},
                    root / parent / "mtp.safetensors",
                )
            selected = {name: "tensors/mtp.safetensors"}
            import json

            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": selected})
            )
            actual_selected, files = mtp.mtp_checkpoint_files(root)
            self.assertEqual(actual_selected, selected)
            self.assertEqual(files, [str(root / "tensors/mtp.safetensors")])
            with mock.patch.object(mtp, "required_expert_tensors", return_value=set()):
                loaded = list(
                    mtp.mtp_weights_iterator(
                        selected,
                        [str(root / "other/mtp.safetensors"), *files],
                        "W4A16_NVFP4",
                    )
                )
            self.assertEqual(len(loaded), 1)
            self.assertTrue(torch.all(loaded[0][1].float() == 0.5))

    def test_csf_restores_all_byte_values_and_exceptions(self):
        # Independent source byte matrix, encoded according to the public slab
        # contract, including exceptional values outside the sixteen-byte window.
        rows, cols = 32, 16
        source = (torch.arange(rows * cols).reshape(rows, cols) % 256).to(torch.uint8)
        bases = torch.arange(rows).remainder(16).mul(16).to(torch.uint8)
        source[0, 0], source[31, 15] = 200, 1
        delta = source.to(torch.int16) - bases[:, None]
        outside = (delta < 0) | (delta > 15)
        codes = torch.where(outside, 0, delta).to(torch.uint8)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        fixed = torch.cat((bases.reshape(2, 16), packed.reshape(2, -1)), 1)
        pos = outside.flatten().nonzero().flatten()
        exceptions = (pos | (source.flatten()[pos].long() << 24)).to(torch.uint32)
        decoded = adapter.decode_nvfp4_csf(fixed, exceptions, (rows, cols))
        self.assertTrue(torch.equal(decoded.view(torch.uint8), source))

    def test_csf_invalid_streams_are_rejected(self):
        fixed = torch.zeros((1, 144), dtype=torch.uint8)
        for records in ([256], [1, 1]):
            with self.assertRaises(ValueError):
                adapter.decode_nvfp4_csf(
                    fixed, torch.tensor(records, dtype=torch.uint32), (16, 16)
                )
        with self.assertRaises(ValueError):
            adapter.decode_nvfp4_csf(fixed, torch.empty(0, dtype=torch.int32), (16, 16))
        with self.assertRaises(ValueError):
            adapter.scale_sources({"s": "x", "s.nvfp4_csf_fixed": "x"}, "s")
        with self.assertRaises(ValueError):
            adapter.scale_sources({"s.nvfp4_csf_fixed": "x"}, "s")
        self.assertEqual(adapter.scale_sources({"s": "x"}, "s"), ("s",))
        self.assertEqual(
            len(
                adapter.scale_sources(
                    {"s.nvfp4_csf_fixed": "x", "s.nvfp4_csf_exceptions": "y"}, "s"
                )
            ),
            2,
        )

    def test_packed_ple_index_and_strided_row_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ple.safetensors"
            name = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0."
            weight = (
                torch.arange(32 * 80).remainder(256).to(torch.uint8).reshape(32, 80)
            )
            scales = torch.ones((32, 10), dtype=torch.float8_e4m3fn)
            save_file({name + "weight": weight, name + "weight_scale": scales}, path)
            index = ple.read_ple_checkpoint_index(str(path))
            self.assertEqual(set(index), {name + "weight", name + "weight_scale"})
            packed = torch.empty((3, 90), dtype=torch.uint8)
            caches = [
                ple.PleRowCache(
                    [(0, 32, index[name + kind])],
                    row_bytes=size,
                    capacity_rows=3,
                    pin_memory=False,
                )
                for kind, size in [("weight", 80), ("weight_scale", 10)]
            ]
            try:
                for ids in ([31, 0, 7], [0, 8, 9], [9, 31, 0]):
                    caches[0].fill(ids, packed[:, :80])
                    caches[1].fill(ids, packed[:, 80:])
                    self.assertTrue(torch.equal(packed[:, :80], weight[ids]))
                    self.assertTrue(
                        torch.equal(packed[:, 80:], scales.view(torch.uint8)[ids])
                    )
            finally:
                for cache in caches:
                    cache.close()

    def test_mxfp8_scale_names_only_change_for_mxfp8(self):
        q = SimpleNamespace(
            resolve_quant_algo=lambda p: "MXFP8" if p == "m" else "NVFP4"
        )
        self.assertEqual(
            adapter.canonicalize_mxfp8_scale_name("m.weight_scale", q),
            "m.weight_scale_inv",
        )
        self.assertEqual(
            adapter.canonicalize_mxfp8_scale_name("e.weight_scale", q), "e.weight_scale"
        )

    def test_mtp_accepts_nvfp4_contract_and_rejects_wrong_shapes(self):
        names = mtp.required_expert_tensors("W4A16_NVFP4")
        self.assertEqual(len(names), 4608)
        self.assertNotIn("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", names)
        for proj, rows, cols in [
            ("gate_proj", 640, 2560),
            ("up_proj", 640, 2560),
            ("down_proj", 2560, 640),
        ]:
            base = f"mtp.layers.0.mlp.experts.0.{proj}."
            for suffix, shape, dtype in [
                ("weight", (rows, cols // 2), torch.uint8),
                ("weight_scale", (rows, cols // 16), torch.float8_e4m3fn),
                ("weight_scale_2", (), torch.float32),
            ]:
                mtp.validate_mtp_tensor(base + suffix, shape, dtype, "W4A16_NVFP4")
            with self.assertRaises(ValueError):
                mtp.validate_mtp_tensor(
                    base + "weight", (rows, cols), torch.bfloat16, "W4A16_NVFP4"
                )

    def test_converter_restores_csf_pairs_across_files(self):
        from sglang.srt.model_loader.qwen38_nvfp4_pack.convert import (
            load_layer_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            files = [{}, {}]
            index = {}
            expected = {}
            for proj, rows, cols, target, shard in [
                ("gate_proj", 16, 256, "w13", "w1"),
                ("up_proj", 16, 256, "w13", "w3"),
                ("down_proj", 256, 16, "w2", "w2"),
            ]:
                base = "model.language_model.layers.0.mlp.experts.0." + proj + "."
                scales = torch.full((rows, cols // 16), 0.5, dtype=torch.float8_e4m3fn)
                for suffix, tensor in [
                    ("weight", torch.full((rows, cols // 2), 17, dtype=torch.uint8)),
                    ("weight_scale_2", torch.tensor(0.1)),
                    ("input_scale", torch.tensor(1.0)),
                ]:
                    files[0][base + suffix] = tensor
                    expected[(target + "_" + suffix, shard)] = tensor
                if proj == "down_proj":
                    files[0][base + "weight_scale"] = scales
                else:
                    # Every row uses the same E4M3 base with zero delta codes.
                    fixed = torch.zeros(
                        (rows // 16, 16 * (1 + cols // 32)), dtype=torch.uint8
                    )
                    fixed[:, :16] = scales.view(torch.uint8)[0, 0]
                    files[0][base + "weight_scale.nvfp4_csf_fixed"] = fixed
                    files[1][base + "weight_scale.nvfp4_csf_exceptions"] = torch.empty(
                        0, dtype=torch.uint32
                    )
                expected[(target + "_weight_scale", shard)] = scales
            for i, tensors in enumerate(files):
                filename = f"{i}.safetensors"
                save_file(tensors, Path(directory) / filename)
                index.update({name: filename for name in tensors})
            loaded = {}

            def loader(param, tensor, name, *, shard_id, expert_id):
                self.assertEqual(expert_id, 0)
                loaded[(name, shard_id)] = tensor.clone()

            layer = SimpleNamespace(
                num_experts=1,
                intermediate_size_per_partition=16,
                w2_weight=torch.empty(1, 256, 8),
                weight_loader=loader,
                w13_weight_scale_2=torch.tensor([[0.1, 0.1]]),
                quant_method=SimpleNamespace(
                    process_weights_after_loading=lambda _: None
                ),
            )
            for target, _ in expected:
                if not hasattr(layer, target):
                    setattr(layer, target, None)
            load_layer_checkpoint(layer, directory, index, 0)
            self.assertEqual(loaded.keys(), expected.keys())
            for key in expected:
                self.assertTrue(
                    torch.equal(
                        loaded[key].reshape(-1).view(torch.uint8),
                        expected[key].reshape(-1).view(torch.uint8),
                    ),
                    key,
                )

    def test_full_packed_table_interleaves_planes_with_bounded_chunks(self):
        from sglang.srt.models.qwen38_fp4_ple import copy_packed_ple_table

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ple.safetensors"
            prefix = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0."
            weight = torch.arange(7 * 80).remainder(256).to(torch.uint8).reshape(7, 80)
            scale = torch.full((7, 10), 0.5, dtype=torch.float8_e4m3fn)
            save_file({prefix + "weight": weight, prefix + "weight_scale": scale}, path)
            desc = ple.read_ple_checkpoint_index(str(path))
            target = torch.empty(7, 90, dtype=torch.uint8)
            copy_packed_ple_table(
                target,
                [(0, 7, desc[prefix + "weight"])],
                [(0, 7, desc[prefix + "weight_scale"])],
                7,
                160,
                chunk_rows=2,
            )
            self.assertTrue(torch.equal(target[:, :80], weight))
            self.assertTrue(torch.equal(target[:, 80:], scale.view(torch.uint8)))
