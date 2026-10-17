# SPDX-License-Identifier: Apache-2.0
"""Execute the mapped checkpoint loader with CPU tensors and a fake CUDA registrar."""

import ast
import contextlib
import json
import logging
import mmap
import os
import struct
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from ._load_source import REPO_ROOT


class FakeEmbedding(nn.Module):
    def __init__(self, rows, width, **kwargs):
        super().__init__()
        self.org_vocab_size = rows
        self.embedding_dim = width
        self.weight = nn.Parameter(
            torch.empty(rows, width, dtype=torch.bfloat16), requires_grad=False
        )


def implementation():
    tree = ast.parse(
        (REPO_ROOT / "python/sglang/srt/models/qwen38_host_embedding.py").read_text()
    )
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    env = dict(
        torch=torch,
        nn=nn,
        VocabParallelEmbedding=FakeEmbedding,
        json=json,
        mmap=mmap,
        os=os,
        struct=struct,
        time=time,
        logger=logging.getLogger(__name__),
    )
    exec(compile(tree, "host_embedding", "exec"), env)
    return env


def checkpoint(path, tensor, dtype="BF16"):
    payload = tensor.view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {
            "model.language_model.embed_tokens.weight": {
                "shape": list(tensor.shape),
                "dtype": dtype,
                "data_offsets": [0, len(payload)],
            }
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


class Tests(unittest.TestCase):
    def test_mapping_byte_parity_no_checkpoint_writes_and_close(self):
        env = implementation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            source = torch.arange(28).reshape(7, 4).to(torch.bfloat16)
            checkpoint(path, source)
            original = path.read_bytes()
            descriptor = env["find_token_embedding"]([path])
            registered = []
            fake_module = types.SimpleNamespace(
                _module=lambda: types.SimpleNamespace(
                    mailbox_register=lambda raw: (
                        registered.append(raw.data_ptr()) or 0x100000
                    ),
                    mailbox_unregister=lambda raw: registered.append("unregister"),
                )
            )
            with (
                mock.patch.dict(
                    sys.modules,
                    {"sglang.kernels.ops.moe.qwen38_expert_cache": fake_module},
                ),
                mock.patch.object(torch.cuda, "current_device", return_value=0),
                mock.patch.object(
                    torch.cuda,
                    "device",
                    side_effect=lambda *_: contextlib.nullcontext(),
                ),
                mock.patch.object(torch.cuda, "synchronize"),
            ):
                model = env["Qwen38MappedTokenEmbedding"](7, 4, descriptor)
                self.assertEqual(model.weight.device.type, "meta")
                model.weight_loader(model.weight, source)
                self.assertEqual(model.weight.device.type, "cpu")
                self.assertTrue(torch.equal(model.weight, source))
                self.assertIsNone(model.quant_method)
                model.weight.data[0, 0] = -9
                self.assertEqual(path.read_bytes(), original)
                model.close()
                model.close()
                self.assertEqual(registered[-1], "unregister")

    def test_rejects_wrong_dtype_missing_duplicate_and_truncation(self):
        env = implementation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            source = torch.ones(7, 4, dtype=torch.bfloat16)
            checkpoint(path, source, dtype="F16")
            with self.assertRaises(ValueError):
                env["find_token_embedding"]([path])
            checkpoint(path, source)
            with self.assertRaises(ValueError):
                env["find_token_embedding"]([path, path])
            with self.assertRaises(ValueError):
                env["find_token_embedding"]([])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                env["find_token_embedding"]([path])


if __name__ == "__main__":
    unittest.main()
