# SPDX-License-Identifier: Apache-2.0
"""Lossless pack storage, cache row relocation and the actual CPU miss ABI."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import compress_pack, decode, encode
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
    PACK_TENSORS,
    Qwen38ExpertPack,
    Qwen38ExpertPackManifest,
    TensorRecord,
    load_manifest,
    write_manifest,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, stage="base-a", runner_config="cpu")


def fixture(root, *, h=256, i=128, experts=4, native_fallback=True):
    rng = np.random.default_rng(52)
    chunks = {
        "w13_weight": rng.integers(0, 256, (experts, 2 * i, h // 2), dtype=np.uint8),
        "w2_weight": rng.integers(0, 256, (experts, h, i // 2), dtype=np.uint8),
        "w13_blockscale_swizzled": rng.integers(
            20, 36, (experts, 2 * i, h // 16), dtype=np.uint8
        ),
        "w2_blockscale_swizzled": rng.integers(
            20, 36, (experts, h, i // 16), dtype=np.uint8
        ),
        "g1_alphas": np.full(experts, 0.8, np.float32),
        "g2_alphas": np.full(experts, 1.2, np.float32),
        "w13_input_scale_quant": np.array([0.7], np.float32),
        "w2_input_scale_quant": np.array([1.1], np.float32),
    }
    chunks["w13_blockscale_swizzled"].reshape(-1)[::431] = 65
    # Poorly compressible FC2 deliberately exercises native plane fallback.
    if native_fallback:
        chunks["w2_blockscale_swizzled"] = rng.integers(
            0, 100, (experts, h, i // 16), dtype=np.uint8
        )
    records = {}
    payload = root / "native.bin"
    with payload.open("wb") as out:
        for name in PACK_TENSORS:
            a = chunks[name]
            records[name] = TensorRecord(
                out.tell(),
                a.nbytes,
                a.shape,
                "float8_e4m3fn" if "blockscale" in name else str(a.dtype),
            )
            out.write(a.tobytes())
    m = Qwen38ExpertPackManifest(
        payload,
        payload.stat().st_size,
        "synthetic",
        "test",
        "flashinfer_cutlass",
        "sm120",
        1,
        experts,
        3,
        h,
        i,
        {0: records},
    )
    write_manifest(m, root / "native.json")
    return m, chunks


class TestCsfPack(CustomTestCase):
    def test_tiered_banks_roundtrip_cpu_and_budget(self):
        """External records retain CPU results and count against GPU byte budgets."""
        from sglang.srt.layers.moe.qwen38_cache_capacity import slots_from_byte_budget
        from sglang.srt.model_loader.qwen38_nvfp4_pack.tiered import (
            fixed_bytes,
            restore_slab,
            tier_pack,
        )
        import mmap

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, chunks = fixture(root, h=2560, i=640, native_fallback=False)
            compress_pack(root / "native.json", root / "csf.bin", root / "csf.json")
            old = load_manifest(root / "csf.json")
            sizes = {
                n: fixed_bytes(s["rows"], s["columns"])
                for n, s in old.scale_encoding.items()
            }
            report = tier_pack(
                root / "csf.json",
                root / "tier.bin",
                root / "tier.json",
                base_strides=sizes,
            )
            tiered = load_manifest(root / "tier.json")
            banks = sum(v["resident_bytes"] for v in report["planes"].values())
            self.assertGreater(banks, 0)
            budget = banks + 3 * report["slot_bytes"]
            self.assertEqual(
                slots_from_byte_budget({"cache_budget_bytes": budget}, tiered)[
                    "cache_slots_total"
                ],
                3,
            )
            with (
                tiered.payload_path.open("rb") as stream,
                mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping,
            ):
                for name, spec in tiered.scale_encoding.items():
                    for expert in range(tiered.num_experts):
                        start = tiered.tensors[0][name].offset + expert * spec["stride"]
                        restored = restore_slab(
                            mapping, start, spec, spec["overflow"]["0"]
                        )
                        self.assertEqual(
                            decode(restored, spec["rows"], spec["columns"]),
                            chunks[name][expert].tobytes(),
                        )
            operators, packs = [], []
            try:
                for manifest in (native, tiered):
                    pack = Qwen38ExpertPack(manifest, source_mode="file_cow")
                    packs.append(pack)
                    operators.append(CpuNvfp4("", pack, (), gpu_backend="b12x"))
                torch.manual_seed(101)
                for nt in (1, 3):
                    x = (torch.randn(nt, 2560) * 0.05).bfloat16()
                    ids = torch.tensor(
                        [[0, 1, -1], [1, 3, 1], [2, 0, 3]][:nt], dtype=torch.int32
                    )
                    weights = torch.rand(nt, 3)
                    outputs = []
                    for op in operators:
                        out = torch.empty(nt, 2560)
                        op.run(0, x, ids, weights, out)
                        outputs.append(out)
                    torch.testing.assert_close(*outputs, rtol=0, atol=0)
            finally:
                for op in operators:
                    op.close()
                for pack in packs:
                    pack.close()
            # Reject an out-of-payload bank before GPU registration or CPU work.
            with tiered.payload_path.open("r+b") as stream:
                name = next(iter(tiered.scale_encoding))
                stream.seek(tiered.tensors[0][name].offset + 976)
                stream.write(struct.pack("<Q", tiered.payload_nbytes + 16))
            with Qwen38ExpertPack(tiered, source_mode="file_cow") as pack:
                with self.assertRaisesRegex(ValueError, "bank extent"):
                    pack.validate_scale_slabs()

    def test_roundtrip_padding_exceptions_and_invalid_extent(self):
        rng = np.random.default_rng(13)
        for rows, cols in ((128, 4), (1280, 160), (2560, 40)):
            raw = rng.integers(0, 256, rows * cols, dtype=np.uint8).tobytes()
            slab = encode(raw, rows, cols)
            self.assertEqual(decode(slab + bytes(16), rows, cols), raw)
            with self.assertRaisesRegex(ValueError, "extent"):
                decode(
                    slab[:1008] + struct.pack("<I", 0xFFFFFFFF) + slab[1012:],
                    rows,
                    cols,
                )
            with self.assertRaisesRegex(ValueError, "truncated"):
                decode(slab[:15], rows, cols)

    def test_pack_migration_and_actual_capacity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            native, chunks = fixture(root)
            report = compress_pack(
                root / "native.json", root / "csf.bin", root / "csf.json"
            )
            m = load_manifest(root / "csf.json")
            self.assertEqual(set(m.scale_encoding), {"w13_blockscale_swizzled"})
            self.assertLess(
                report["compressed_slot_bytes"], report["native_slot_bytes"]
            )
            from sglang.srt.layers.moe.qwen38_cache_capacity import (
                slots_from_byte_budget,
            )

            # Use a partial budget so the model's maximum does not clamp growth.
            budget = 3 * report["native_slot_bytes"]
            sized = slots_from_byte_budget({"cache_budget_bytes": budget}, m)
            self.assertEqual(
                sized["cache_slots_total"],
                min(budget // report["compressed_slot_bytes"], m.num_experts),
            )
            with self.assertRaisesRegex(ValueError, "combined"):
                slots_from_byte_budget(
                    {"cache_budget_bytes": budget, "cache_slots_total": 3}, m
                )
            with Qwen38ExpertPack(m, source_mode="file_cow") as pack:
                name = "w13_blockscale_swizzled"
                spec = m.scale_encoding[name]
                # Existing pool migration copies opaque rows with the new stride.
                pool = torch.zeros((7, spec["stride"]), dtype=torch.uint8)
                for e in range(m.num_experts):
                    view = pack.expert_tensor_bytes(0, e, name)
                    pool[e].copy_(torch.frombuffer(bytearray(view), dtype=torch.uint8))
                    view.release()
                relocated = pool[[3, 0, 2, 1]].clone()
                for slot, e in enumerate((3, 0, 2, 1)):
                    self.assertEqual(
                        decode(
                            relocated[slot].numpy().tobytes(),
                            spec["rows"],
                            spec["columns"],
                        ),
                        chunks[name][e].tobytes(),
                    )
            malformed = m.to_dict(manifest_path=root / "bad.json")
            malformed["scale_encoding"][name]["stride"] += 1
            import json

            (root / "bad.json").write_text(json.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "CSF"):
                load_manifest(root / "bad.json")

    def test_slab_validation_before_gpu_registration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            compress_pack(root / "native.json", root / "csf.bin", root / "csf.json")
            manifest = load_manifest(root / "csf.json")
            record = manifest.tensors[0]["w13_blockscale_swizzled"]
            with manifest.payload_path.open("r+b") as out:
                out.seek(record.offset)
                out.write(struct.pack("<I", 0xFFFFFFFF))
            with Qwen38ExpertPack(manifest, source_mode="file_cow") as pack:
                with self.assertRaisesRegex(ValueError, "CSF slab header"):
                    pack.validate_scale_slabs()

    @unittest.skipUnless(
        "avx512_bf16" in Path("/proc/cpuinfo").read_text(), "requires AVX512 BF16"
    )
    def test_cpu_miss_reads_compressed_pack(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, h=256, i=256, native_fallback=False)
            compress_pack(root / "native.json", root / "csf.bin", root / "csf.json")
            self.assertEqual(len(load_manifest(root / "csf.json").scale_encoding), 2)
            for backend in ("b12x", "triton"):
                operators = []
                packs = []
                try:
                    for path in ("native.json", "csf.json"):
                        pack = Qwen38ExpertPack(
                            load_manifest(root / path), source_mode="file_cow"
                        )
                        packs.append(pack)
                        operators.append(CpuNvfp4("", pack, (), gpu_backend=backend))
                    torch.manual_seed(5)
                    for nt in (1, 2, 3):
                        x = (torch.randn(nt, 256) * 0.002).to(torch.bfloat16)
                        ids = torch.tensor(
                            [[0, 1, -1], [1, 3, 1], [2, 0, 3]][:nt], dtype=torch.int32
                        )
                        weights = torch.rand(nt, 3)
                        outputs = []
                        for op in operators:
                            out = torch.empty(nt, 256)
                            op.run(0, x, ids, weights, out)
                            outputs.append(out)
                        torch.testing.assert_close(*outputs, rtol=0, atol=0)
                    operators[-1].release_worker()
                finally:
                    for op in operators:
                        op.close()
                    for pack in packs:
                        pack.close()


if __name__ == "__main__":
    unittest.main()
