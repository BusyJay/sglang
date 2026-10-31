# SPDX-License-Identifier: Apache-2.0
"""The diagnostic must report honest nesting and restore inference hooks."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "mlx_profile_tested", ROOT / "benchmark/mlx/mlx_profile.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class TestMlxProfile(CustomTestCase):
    def test_device_interval_union(self):
        spec = importlib.util.spec_from_file_location(
            "metal_profile_tested", ROOT / "benchmark/mlx/mlx_metal_trace.py"
        )
        metal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metal)
        self.assertEqual(metal.union_ns([(0, 10), (2, 8), (5, 15), (20, 21)]), 16)

    def test_device_xml_references_filter_process_and_merge_overlaps(self):
        spec = importlib.util.spec_from_file_location(
            "metal_profile_xml_tested", ROOT / "benchmark/mlx/mlx_metal_trace.py"
        )
        metal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metal)
        columns = (
            "process",
            "start",
            "duration",
            "event-label",
            "channel-name",
            "event-depth",
            "start-latency",
            "cmdbuffer-id",
        )
        schema = (
            '<schema name="metal-gpu-intervals">'
            + "".join(f"<col><mnemonic>{name}</mnemonic></col>" for name in columns)
            + "</schema>"
        )
        rows = (
            '<row><process id="p"><pid>77</pid></process><start>1000</start>'
            '<duration>500</duration><event-label id="e" fmt="Compute"/>'
            '<channel-name id="c">GPU 0</channel-name><event-depth>0</event-depth>'
            "<start-latency>250</start-latency><cmdbuffer-id>1</cmdbuffer-id></row>"
            '<row><process ref="p"/><start>1200</start><duration>700</duration>'
            '<event-label ref="e"/><channel-name ref="c"/><event-depth>1</event-depth>'
            "<start-latency>500</start-latency><cmdbuffer-id>2</cmdbuffer-id></row>"
            "<row><process><pid>88</pid></process><start>0</start><duration>9999</duration>"
            '<event-label ref="e"/><channel-name ref="c"/><event-depth>0</event-depth>'
            "<start-latency>0</start-latency><cmdbuffer-id>3</cmdbuffer-id></row>"
        )
        # Keep task/test artifacts inside the checkout, never system tmp.
        with tempfile.TemporaryDirectory(prefix="mlx-profile-test-", dir=ROOT) as work:
            work = Path(work)
            source = work / "gpu.xml"
            source.write_text(
                "<trace-query-result><node>"
                + schema
                + rows
                + "</node></trace-query-result>"
            )
            result = metal.analyze(source, work / "report", 77)
            self.assertEqual(result["event_count"], 2)
            self.assertAlmostEqual(result["interval_union_ms"], 0.0009)
            events = json.loads((work / "report/device.trace.json").read_text())[
                "traceEvents"
            ]
            self.assertEqual({e["pid"] for e in events}, {77})
            self.assertEqual({e["name"] for e in events}, {"Compute"})
            self.assertEqual({e["tid"] for e in events}, {"GPU 0"})

    def test_comparison_exposes_different_generation_budgets(self):
        spec = importlib.util.spec_from_file_location(
            "profile_cli_tested", ROOT / "benchmark/mlx/profile_glm_mlx.py"
        )
        cli = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"mlx_profile": PROFILE}):
            spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory(prefix="mlx-profile-test-", dir=ROOT) as work:
            directories = [Path(work) / name for name in ("before", "after")]
            for directory, count in zip(directories, (32, 64)):
                directory.mkdir()
                result = dict(
                    synthetic=False,
                    workload="generate",
                    revision="pinned",
                    steps=1,
                    verify_mode="ffn",
                    rows=2,
                    tokens=count,
                    prompt_tokens_sha256="same-prompt",
                    modes={
                        "baseline": dict(
                            median_ms=count,
                            median_decode_ms=count,
                            samples=[dict(tokens_sha256=str(count))],
                        )
                    },
                )
                (directory / "results.json").write_text(json.dumps(result))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                cli.compare(
                    types.SimpleNamespace(before=directories[0], after=directories[1])
                )
            result = json.loads(stream.getvalue())
            self.assertEqual(result["configuration_mismatch"]["tokens"], [32, 64])
            self.assertFalse(result["token_hash_equal"])

    def test_nested_exceptions_and_bounded_trace(self):
        rec = PROFILE.Recorder(None, max_events=2)
        with self.assertRaisesRegex(ValueError, "test"):
            with rec.span("parent"):
                with rec.span("child"):
                    raise ValueError("test")
        self.assertEqual(rec.stack, [])
        parent = next(e for e in rec.events if e["name"] == "parent")
        child = next(e for e in rec.events if e["name"] == "child")
        self.assertAlmostEqual(parent["args"]["self_us"], parent["dur"] - child["dur"])
        self.assertEqual(child["args"]["exception"], "ValueError")
        with rec.span("dropped"):
            pass
        self.assertEqual(rec.dropped, 1)
        self.assertEqual(len(rec.events), 2)
        json.dumps(rec.summary())

    def test_staticmethod_and_hooks_restored_after_error(self):
        # Only external model/MLX boundaries are substituted: the profiler's
        # actual instrument and invocation machinery is exercised.
        class Block:
            def __call__(self, *args):
                return None

            verify_ffn_rows = _branch_output = project_verify_rows = __call__
            _latent_matrices = forward_with_hidden = verify_mtp_rows = __call__

        class Engine:
            @staticmethod
            def _publish(state, cache):
                return state + cache

            prefill = decode = _flush_head = Block.__call__

        fake_mx = types.SimpleNamespace(array=type("Array", (), {}))
        for name in (
            "eval",
            "async_eval",
            "synchronize",
            "clear_cache",
            "quantized_matmul",
            "gather_qmm",
        ):
            setattr(fake_mx, name, lambda *args: None)
        module = types.SimpleNamespace(
            **{
                name: type(name, (Block,), {})
                for name in (
                    "MLP",
                    "MoE",
                    "ClampedSwitchGLU",
                    "KDA",
                    "MLA",
                    "DecoderLayer",
                    "MTPHead",
                    "Model",
                )
            }
        )
        model = types.SimpleNamespace(named_modules=lambda: [])
        before = Engine.__dict__["_publish"]
        original_eval = fake_mx.eval
        rec = PROFILE.Recorder(fake_mx)
        with self.assertRaises(ValueError):
            with rec.instrument(module, model, Engine):
                self.assertEqual(Engine()._publish(3, 4), 7)
                raise ValueError("abort replay")
        self.assertIs(Engine.__dict__["_publish"], before)
        self.assertIs(fake_mx.eval, original_eval)
        self.assertEqual(Engine()._publish(3, 4), 7)


if __name__ == "__main__":
    unittest.main()
