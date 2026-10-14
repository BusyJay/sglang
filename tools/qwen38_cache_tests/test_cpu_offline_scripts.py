import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ._load_source import REPO_ROOT, load_source
from .test_cpu_loader_filter import common_module


class TestOfflineScripts(unittest.TestCase):
    def test_docker_repo_id_mounts_cache_and_retains_model_id(self):
        with tempfile.TemporaryDirectory() as root:
            script = REPO_ROOT / "tools/qwen38_cache_tests/docker_model.sh"
            # Fake the two launch commands; no Docker/GPU/server execution.
            bin_dir = Path(root) / "bin"
            bin_dir.mkdir()
            for name, body in (("docker", "exit 0"), ("python3", "printf '%s\\n' \"$@\"")):
                executable = bin_dir / name
                executable.write_text("#!/bin/sh\n" + body + "\n")
                executable.chmod(0o755)
            result = subprocess.run(["bash", str(script),
                "--image", "fixture", "--checkpoint", "nvidia/test-model",
                "--hf-cache", root, "--pack", root, "--results", root,
                "--", "canary", "--revision", "fixture-revision"],
                capture_output=True, text=True, env={**os.environ, "QWEN38_PROD_MOUNT": "",
                    "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"{root}:/workspace/hf-cache:ro", result.stdout)
            self.assertIn("HF_HUB_CACHE=/workspace/hf-cache", result.stdout)
            self.assertIn("HF_HUB_OFFLINE=1", result.stdout)
            self.assertIn("--network\nnone", result.stdout)
            self.assertIn("--checkpoint\nnvidia/test-model", result.stdout)
            self.assertIn("--revision\nfixture-revision", result.stdout)

    def test_canary_model_id_reaches_server_without_path_coercion(self, remote=False):
        canary = load_source("qwen38_offline_canary_test", "tools/qwen38_cache_tests/full_model_canary.py")
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.json"
            manifest.write_text('{"checkpoint_revision":"fixture"}')
            tokenizer = mock.Mock()
            tokenizer.encode.return_value = [1, 2]
            process = mock.Mock()
            process.pid = 12345
            process.poll.return_value = None
            process.wait.return_value = 0
            info = {}
            responses = [info, {"text": "answer", "meta_info": {}}]
            remote_flags = []
            if remote:
                image = Path(root) / "fixture.png"
                image.write_bytes(b"fixture-image")
                remote_flags = ["--encoder-urls", "http://encoder-fixture:31000",
                    "--image", str(image), "--image-answer-pattern", "red"]
                responses.append({"choices": [{"message": {"content": "red square"}}]})
            with mock.patch.dict("sys.modules", {
                "transformers": mock.Mock(AutoTokenizer=mock.Mock(from_pretrained=mock.Mock(return_value=tokenizer))),
                "sglang.srt.model_loader.qwen38_nvfp4_common": common_module,
            }), mock.patch("sys.argv", ["canary", "--checkpoint", "nvidia/test-model",
                "--manifest", str(manifest), "--output", root, "--revision", "fixture",
                "--cache-dir", root, "--contexts", "8", "--offload-startup-warmup",
                "--decode-graph", "--runtime-mode", "threaded"] + remote_flags +
                ([] if remote else ["--mtp-steps", "1", "--total-slots", "3840", "--adaptive-cache"])), \
                mock.patch.object(common_module, "resolve_cached_checkpoint", return_value=Path(root)) as resolve, \
                mock.patch.object(common_module, "cached_checkpoint_weight_files"), \
                mock.patch.object(canary.subprocess, "Popen", return_value=process) as launch, \
                mock.patch.object(canary, "ready", return_value=True), \
                mock.patch.object(canary, "request", side_effect=responses) as request_mock, \
                mock.patch.object(canary.os, "killpg"):
                self.assertEqual(canary.main(), 0)
            resolve.assert_called_once_with("nvidia/test-model", revision="fixture", cache_dir=root)
            args = launch.call_args.args[0]
            for flag, expected in (("--model-path", "nvidia/test-model"),
                ("--served-model-name", "nvidia/test-model"), ("--revision", "fixture"),
                ("--reasoning-parser", "qwen3"), ("--download-dir", root),
                ("--warmups", "qwen38_offload")):
                self.assertEqual(args[args.index(flag) + 1], expected)
            self.assertIn("--cuda-graph-backend-decode=breakable", args)
            self.assertIn("--cuda-graph-backend-prefill=disabled", args)
            if not remote:
                self.assertEqual(args[args.index("--speculative-draft-load-format") + 1], "qwen38_mtp")
                extra = json.loads(args[args.index("--model-loader-extra-config") + 1])
                self.assertEqual(extra["cache_slots_total"], 3840)
                self.assertNotIn("cache_slots_per_layer", extra)
                self.assertTrue(extra["adaptive_cache"])
            self.assertEqual("--language-model-only" in args, not remote)
            if remote:
                self.assertIn("--language-only", args)
                self.assertIn("http://encoder-fixture:31000", args)
                payload = request_mock.call_args.args[1]
                self.assertTrue(request_mock.call_args.args[0].endswith("/v1/chat/completions"))
                self.assertTrue(payload["messages"][0]["content"][0]["image_url"]["url"].startswith(
                    "data:image/png;base64,"))

    def test_canary_remote_vision_and_image_probe(self):
        self.test_canary_model_id_reaches_server_without_path_coercion(remote=True)
