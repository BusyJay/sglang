# SPDX-License-Identifier: Apache-2.0
"""CPU-only global cache and measured branch selection invariants."""

import copy
import math
import unittest

from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu, HybridCostModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestHybridCostModel(CustomTestCase):
    METADATA = dict(
        backend="b12x",
        hidden=4096,
        width=1536,
        top_k=4,
        cores=32,
        gpu="NVIDIA GPU",
        cpu="CPU model",
    )

    def calibration(self, **overrides):
        result = dict(
            self.METADATA,
            schema_version=2,
            samples=[
                dict(
                    tokens=1,
                    hit_routes=2,
                    miss_routes=2,
                    miss_experts=1,
                    miss_tokens=[2],
                    cpu_us=80.0,
                    gpu_us=100.0,
                )
            ],
        )
        result.update(overrides)
        if isinstance(result["samples"], list):
            result["samples"] = [
                (
                    dict(
                        sample,
                        gpu_fill_experts=sample.get(
                            "gpu_fill_experts",
                            len(
                                sample.get(
                                    "rejected_tokens", sample.get("miss_tokens", [])
                                )
                                or []
                            )
                            - sample.get("miss_experts", 0),
                        ),
                        rejected_tokens=sample.get(
                            "rejected_tokens", copy.deepcopy(sample.get("miss_tokens"))
                        ),
                    )
                    if isinstance(sample, dict)
                    else sample
                )
                for sample in result["samples"]
            ]
        return result

    def model(self, calibration):
        return HybridCostModel(calibration, **self.METADATA)

    def test_fill_buckets_separate_opposite_branch_outcomes(self):
        samples = [
            dict(
                self.calibration()["samples"][0],
                gpu_fill_experts=fills,
                cpu_us=cpu_us,
                gpu_us=gpu_us,
            )
            for fills, cpu_us, gpu_us in [(0, 160, 100), (2, 80, 100)]
        ]
        model = self.model(self.calibration(samples=samples))
        self.assertEqual(model.choose(1, 2, 2, 1, [2], [2], 0), "gpu")
        self.assertEqual(model.choose(1, 2, 2, 1, [2], [2], 2), "cpu")
        self.assertEqual(model.choose(1, 2, 2, 1, [2], [2], 1), "gpu")
        rows = [[0, 1, 2, 2]]
        self.assertEqual(model.select_cpu(1, rows, {0: 0, 1: 1}, {2}), set())
        self.assertEqual(
            model.select_cpu(1, rows, {0: 0, 1: 1}, {2}, admitted_experts=2), {2}
        )

    def test_fill_counts_validate_schema_and_runtime(self):
        for value in [-1, 3, True, 1.0, None, "0"]:
            with self.subTest(value=value):
                table = self.calibration()
                table["samples"][0]["gpu_fill_experts"] = value
                with self.assertRaisesRegex(ValueError, "gpu_fill_experts"):
                    self.model(table)
                self.assertEqual(
                    self.model(self.calibration()).choose(1, 2, 2, 1, [2], [2], value),
                    "gpu",
                )
        table = self.calibration()
        del table["samples"][0]["gpu_fill_experts"]
        with self.assertRaisesRegex(ValueError, "gpu_fill_experts"):
            self.model(table)
        table = self.calibration()
        table["samples"][0].update(
            hit_routes=2, miss_tokens=[2], rejected_tokens=[1, 2], gpu_fill_experts=0
        )
        with self.assertRaisesRegex(ValueError, "gpu_fill_experts"):
            self.model(table)

    def test_device_rows_filter_and_sort_scalar_buckets(self):
        sample = self.calibration()["samples"][0]
        samples = [
            dict(sample, gpu_fill_experts=fills, cpu_us=cpu_us)
            for fills, cpu_us in [(2, 70), (0, 80), (1, 90)]
        ]
        model = self.model(self.calibration(samples=samples))
        rows = model.device_cost_rows(32)
        self.assertEqual([row[6] for row in rows], [0, 2])
        self.assertEqual([row[5] for row in rows], [80, 70])
        self.assertTrue(all(len(row) == 71 for row in rows))
        self.assertEqual(rows[0][7:9], [2, 0])
        self.assertEqual(rows[0][39:41], [2, 0])
        with self.assertRaisesRegex(ValueError, "route capacity"):
            model.device_cost_rows(0)

    def test_full_branch_cost_and_strict_default_margin(self):
        # End-to-end times, rather than worker/kernel time, decide the branch.
        cases = [
            (80.0, "cpu"),
            (89.999, "cpu"),
            (90.0, "gpu"),
            (90.001, "gpu"),
            (100.0, "gpu"),
            (120.0, "gpu"),
        ]
        for cpu_us, expected in cases:
            with self.subTest(cpu_us=cpu_us):
                calibration = self.calibration()
                calibration["samples"][0]["cpu_us"] = cpu_us
                self.assertEqual(
                    self.model(calibration).choose(1, 2, 2, 1, (2,)), expected
                )

    def test_zero_hits_can_use_measured_cpu(self):
        calibration = self.calibration()
        calibration["samples"][0].update(
            hit_routes=0,
            miss_routes=4,
            miss_experts=4,
            miss_tokens=[1, 1, 1, 1],
            rejected_tokens=[1, 1, 1, 1],
        )
        model = self.model(calibration)
        self.assertEqual(model.choose(1, 0, 4, 4, (1, 1, 1, 1)), "cpu")
        self.assertEqual(model.choose(1, 0, 4, 3, (1, 1, 2)), "gpu")

    def test_unmeasured_geometry_defaults_to_gpu_without_interpolation(self):
        model = self.model(self.calibration())
        for workload in [
            (2, 2, 2, 1),
            (1, 1, 2, 1),
            (1, 2, 1, 1),
            (1, 2, 2, 2),
            (4, 2, 2, 1),
            (1, 0, 4, 4),
        ]:
            with self.subTest(workload=workload):
                self.assertEqual(model.choose(*workload, (2,)), "gpu")

    def test_no_calibration_and_empty_table_default_to_gpu(self):
        for calibration in [None, self.calibration(samples=[])]:
            with self.subTest(calibration=calibration):
                model = self.model(calibration)
                self.assertEqual(model.choose(1, 0, 4, 4), "gpu")
                self.assertEqual(model.choose(1, 2, 2, 1), "gpu")
                self.assertEqual(model.choose(1, 2, 2, 1, (2,)), "gpu")
                self.assertEqual(model.choose(1, 0, 4, 4, (1, 1, 1, 1)), "gpu")
                self.assertEqual(model.choose(1, 2, 2, 1, iter([2])), "gpu")
                self.assertEqual(model.choose(1, 4, 0, 0), "hit")
                self.assertEqual(model.choose(0, 0, 0, 0), "gpu")
        self.assertEqual(self.model(self.calibration()).choose(1, 4, 0, 0), "hit")

    def test_custom_margin_boundaries(self):
        for margin, cpu_us, expected in [
            (0, 99.0, "cpu"),
            (0, 100.0, "gpu"),
            (0.25, 74.0, "cpu"),
            (0.25, 75.0, "gpu"),
            (0.999, 80.0, "gpu"),
        ]:
            with self.subTest(margin=margin, cpu_us=cpu_us):
                calibration = self.calibration(margin=margin)
                calibration["samples"][0]["cpu_us"] = cpu_us
                self.assertEqual(
                    self.model(calibration).choose(1, 2, 2, 1, (2,)), expected
                )

    def test_exact_metadata_is_required(self):
        for name, expected in self.METADATA.items():
            wrong = expected + 1 if type(expected) is int else expected + " other"
            for value in [None, wrong, str(expected) if type(expected) is int else 1]:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.model(self.calibration(**{name: value}))
            calibration = self.calibration()
            del calibration[name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, name):
                self.model(calibration)
        for name in ("hidden", "width", "top_k", "cores"):
            for value in [True, float(self.METADATA[name])]:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.model(self.calibration(**{name: value}))
        # A table measured with logical threads cannot match physical cores.
        with self.assertRaisesRegex(ValueError, "cores"):
            self.model(self.calibration(cores=64))

    def test_invalid_runtime_metadata_is_rejected_even_if_table_matches(self):
        for name, invalids in [
            ("cores", [0, -1, True, 32.0]),
            ("top_k", [0]),
            ("hidden", [0]),
            ("width", [0]),
            ("backend", ["", None]),
            ("gpu", [" "]),
            ("cpu", [1]),
        ]:
            for value in invalids:
                with self.subTest(name=name, value=value):
                    metadata = dict(self.METADATA, **{name: value})
                    with self.assertRaisesRegex(ValueError, name):
                        HybridCostModel(self.calibration(**{name: value}), **metadata)

    def test_transport_and_runtime_version_match_when_supplied(self):
        for name, expected in (
            ("transport", "pinned-d2h-h2d"),
            ("runtime_version", "worker-v1"),
        ):
            metadata = dict(self.METADATA, **{name: expected})
            model = HybridCostModel(self.calibration(**{name: expected}), **metadata)
            self.assertEqual(model.choose(1, 2, 2, 1, (2,)), "cpu")
            for value in [None, "", " ", 1, expected + "-other"]:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        HybridCostModel(self.calibration(**{name: value}), **metadata)
            with self.subTest(name=name, missing="calibration"):
                with self.assertRaisesRegex(ValueError, name):
                    HybridCostModel(self.calibration(), **metadata)
            with self.subTest(name=name, missing="runtime"):
                with self.assertRaisesRegex(ValueError, name):
                    self.model(self.calibration(**{name: expected}))
            for value in ["", " ", 1, True]:
                with self.subTest(name=name, invalid_runtime=value):
                    with self.assertRaisesRegex(ValueError, name):
                        HybridCostModel(
                            self.calibration(**{name: value}),
                            **dict(self.METADATA, **{name: value}),
                        )

    def test_invalid_schema_and_container_shapes(self):
        for calibration in [[], "{}", 1, True]:
            with self.subTest(calibration=calibration):
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    self.model(calibration)
        for version in [None, 0, 1, 3, True, 2.0, "2"]:
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    self.model(self.calibration(schema_version=version))
        for samples in [None, {}, (), "samples", [None], [[]]]:
            with self.subTest(samples=samples):
                with self.assertRaisesRegex(ValueError, "samples"):
                    self.model(self.calibration(samples=samples))
        for name in ("schema_version", "samples"):
            calibration = self.calibration()
            del calibration[name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, name):
                self.model(calibration)

    def test_margin_rejects_nonfinite_out_of_range_and_nonnumeric(self):
        for margin in [
            -0.001,
            1,
            1.1,
            math.nan,
            math.inf,
            -math.inf,
            True,
            "0.1",
            None,
            [],
            10**400,
        ]:
            with self.subTest(margin=margin):
                with self.assertRaisesRegex(ValueError, "margin"):
                    self.model(self.calibration(margin=margin))

    def test_sample_timings_must_be_positive_finite_numbers(self):
        for name in ("cpu_us", "gpu_us"):
            for value in [
                0,
                -1,
                math.nan,
                math.inf,
                -math.inf,
                True,
                "80.0",
                None,
                [],
                10**400,
            ]:
                with self.subTest(name=name, value=value):
                    calibration = self.calibration()
                    calibration["samples"][0][name] = value
                    with self.assertRaisesRegex(ValueError, "samples\\[0\\]"):
                        self.model(calibration)
            calibration = self.calibration()
            del calibration["samples"][0][name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, name):
                self.model(calibration)

    def test_route_counts_reject_invalid_types_and_ranges(self):
        invalid = dict(
            tokens=[0, 4],
            hit_routes=[-1, 3],
            miss_routes=[0, -1, 3],
            miss_experts=[0, -1, 3],
        )
        for name, values in invalid.items():
            for value in values + [True, 1.0, "1", None, []]:
                with self.subTest(name=name, value=value):
                    calibration = self.calibration()
                    calibration["samples"][0][name] = value
                    with self.assertRaisesRegex(ValueError, "samples\\[0\\]"):
                        self.model(calibration)
            calibration = self.calibration()
            del calibration["samples"][0][name]
            with self.subTest(missing=name):
                with self.assertRaisesRegex(ValueError, "counts"):
                    self.model(calibration)

    def test_valid_route_boundaries_and_json_integer_timings(self):
        workloads = [
            (1, 0, 1, 1),
            (1, 0, 4, 4),
            (2, 7, 1, 1),
            (3, 0, 12, 12),
            (3, 11, 1, 1),
            (3, 0, 1, 1),
        ]
        samples = [
            dict(
                tokens=t,
                hit_routes=h,
                miss_routes=m,
                miss_experts=e,
                miss_tokens=[m] if e == 1 else [1] * e,
                cpu_us=1,
                gpu_us=2,
            )
            for t, h, m, e in workloads
        ]
        model = self.model(self.calibration(samples=samples))
        for workload, sample in zip(workloads, samples):
            with self.subTest(workload=workload):
                self.assertEqual(model.choose(*workload, sample["miss_tokens"]), "cpu")

    def test_duplicate_workloads_rejected_even_with_different_timings(self):
        calibration = self.calibration()
        duplicate = dict(calibration["samples"][0], cpu_us=150.0)
        calibration["samples"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicates workload"):
            self.model(calibration)

    def test_same_route_totals_require_matching_expert_distribution(self):
        sample = dict(
            tokens=2,
            hit_routes=0,
            miss_routes=6,
            miss_experts=2,
            miss_tokens=[1, 5],
            cpu_us=80.0,
            gpu_us=100.0,
        )
        model = self.model(self.calibration(samples=[sample]))
        self.assertEqual(model.choose(2, 0, 6, 2, (1, 5)), "cpu")
        self.assertEqual(model.choose(2, 0, 6, 2, (5, 1)), "cpu")
        self.assertEqual(model.choose(2, 0, 6, 2, (2, 4)), "gpu")
        self.assertEqual(model.choose(2, 0, 6, 2, (3, 3)), "gpu")
        # Distinct distributions coexist and use their own full branch costs.
        other = dict(sample, miss_tokens=[2, 4], cpu_us=95.0)
        model = self.model(self.calibration(samples=[sample, other]))
        self.assertEqual(model.choose(2, 0, 6, 2, (1, 5)), "cpu")
        self.assertEqual(model.choose(2, 0, 6, 2, (2, 4)), "gpu")

    def test_missing_or_invalid_runtime_profile_defaults_to_gpu(self):
        model = self.model(self.calibration())
        for counts in (
            (True, 2, 2, True),
            (1.0, 2.0, 2.0, 1.0),
            (1, [], 2, 1),
            (1, -1, 2, 1),
        ):
            with self.subTest(counts=counts):
                self.assertEqual(model.choose(*counts, (2,)), "gpu")
        self.assertEqual(model.choose(1, 2, 2, 1), "gpu")
        for profile in [
            None,
            (),
            (1,),
            (1, 1),
            (0,),
            (5,),
            (True,),
            (2.0,),
            "2",
            2,
            (math.nan,),
            ([],),
        ]:
            with self.subTest(profile=profile):
                self.assertEqual(model.choose(1, 2, 2, 1, profile), "gpu")
        self.assertEqual(model.choose(1, 4, 0, 0, None), "hit")
        self.assertEqual(model.choose(1, 2, 2, 1, iter([2])), "cpu")

    def test_calibration_profile_is_required_sorted_and_consistent(self):
        for profile in [
            None,
            (),
            "2",
            2,
            [],
            [1],
            [1, 1],
            [0],
            [-1],
            [5],
            [True],
            [2.0],
            [math.nan],
            [math.inf],
            [[]],
        ]:
            with self.subTest(profile=profile):
                calibration = self.calibration()
                calibration["samples"][0]["miss_tokens"] = profile
                with self.assertRaisesRegex(ValueError, "miss_tokens"):
                    self.model(calibration)
        calibration = self.calibration()
        del calibration["samples"][0]["miss_tokens"]
        with self.assertRaisesRegex(ValueError, "miss_tokens"):
            self.model(calibration)
        calibration = self.calibration()
        calibration["samples"][0].update(
            hit_routes=0, miss_routes=4, miss_experts=2, miss_tokens=[3, 1]
        )
        with self.assertRaisesRegex(ValueError, "miss_tokens"):
            self.model(calibration)

    def test_calibration_is_not_mutated_or_retained_by_reference(self):
        calibration = self.calibration()
        original = copy.deepcopy(calibration)
        model = self.model(calibration)
        self.assertEqual(calibration, original)
        calibration["samples"][0]["cpu_us"] = 150.0
        calibration["samples"][0]["miss_tokens"][0] = 1
        calibration["samples"][0]["rejected_tokens"][0] = 1
        calibration["margin"] = 0.99
        self.assertEqual(model.choose(1, 2, 2, 1, (2,)), "cpu")

    def test_select_cpu_admission_examples_and_no_benefit(self):
        # Ten misses: admitted experts still contribute to GPU_routes.
        samples = [
            dict(
                tokens=3,
                hit_routes=10 - k,
                miss_routes=k,
                miss_experts=k,
                miss_tokens=[1] * k,
                rejected_tokens=[1] * 5,
                gpu_fill_experts=10 - k,
                cpu_us=80 if k == 1 else 95,
                gpu_us=100,
            )
            for k in range(1, 6)
        ]
        rows = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        model = self.model(self.calibration(samples=samples))
        # All ten admitted, even though they are not resident hits yet.
        self.assertEqual(model.select_cpu(3, rows, {}, set()), frozenset())
        # Five admitted + five rejected; only k=1 beats the strict margin.
        selected = model.select_cpu(3, rows, {}, {5, 6, 7, 8, 9}, admitted_experts=5)
        self.assertEqual(selected, frozenset({5}))
        self.assertEqual(sum(e not in selected for row in rows for e in row), 9)
        for sample in samples:
            sample["cpu_us"] = 90  # Equality at the margin is not eligible.
        model = self.model(self.calibration(samples=samples))
        self.assertEqual(
            model.select_cpu(3, rows, {}, {5, 6, 7, 8, 9}, admitted_experts=5),
            frozenset(),
        )

    def test_select_cpu_minimizes_full_branch_time_over_all_prefixes(self):
        samples = [
            dict(
                tokens=2,
                hit_routes=4 - k,
                miss_routes=k,
                miss_experts=k,
                miss_tokens=[1] * k,
                rejected_tokens=[1] * 3,
                cpu_us=cpu_us,
                gpu_us=gpu_us,
            )
            for k, cpu_us, gpu_us in [(1, 80, 100), (2, 70, 80), (3, 75, 200)]
        ]
        # k=1 is eligible; k=3 has the greatest relative benefit, but k=2
        # has the lowest sampled elapsed time of the complete mixed branch.
        rows = [[0, 1], [2, 3]]
        model = self.model(self.calibration(samples=samples))
        self.assertEqual(model.select_cpu(2, rows, {0: 0}, {1, 2, 3}), {1, 2})
        # Equal elapsed times keep the smaller prefix deterministically.
        samples[2]["cpu_us"] = 70
        model = self.model(self.calibration(samples=samples))
        self.assertEqual(model.select_cpu(2, rows, {0: 0}, {1, 2, 3}), {1, 2})

    def test_select_cpu_counts_duplicate_routes_and_deduplicates_rejected(self):
        samples = [
            dict(
                tokens=3,
                hit_routes=7 - sum(profile),
                miss_routes=sum(profile),
                miss_experts=len(profile),
                miss_tokens=profile,
                rejected_tokens=[1, 1, 3],
                cpu_us=cpu_us,
                gpu_us=100,
            )
            for profile, cpu_us in [([1], 85), ([1, 1], 80), ([1, 1, 3], 95)]
        ]
        model = self.model(self.calibration(samples=samples))
        rows = [[9, 9, 4, -1], [9, 2, 0, -1], [0, -1, -1, -1]]
        mapping, rejected = {0: 0}, [9, 4, 2, 4, 9]
        original = copy.deepcopy((rows, mapping, rejected))
        selected = model.select_cpu(3, rows, mapping, rejected)
        self.assertIsInstance(selected, frozenset)
        self.assertEqual(selected, {2, 4})
        self.assertEqual((rows, mapping, rejected), original)
        # Both one-route experts tie: expert id 2 must precede expert id 4.
        model = self.model(self.calibration(samples=samples[:1]))
        self.assertEqual(model.select_cpu(3, rows, mapping, rejected), {2})
        # Exact distributions are required even when route totals match.
        samples[2].update(miss_tokens=[1, 2, 2], rejected_tokens=[1, 2, 2], cpu_us=1)
        model = self.model(self.calibration(samples=samples))
        self.assertEqual(model.select_cpu(3, rows, mapping, rejected), {2, 4})

    def test_select_cpu_unmeasured_prefixes_and_no_calibration(self):
        rows = [[0, 1, 2, 3]]
        sample = dict(
            tokens=1,
            hit_routes=2,
            miss_routes=2,
            miss_experts=2,
            miss_tokens=[1, 1],
            rejected_tokens=[1, 1, 1],
            cpu_us=80,
            gpu_us=100,
        )
        model = self.model(self.calibration(samples=[sample]))
        self.assertEqual(model.select_cpu(1, rows, {}, {1, 2, 3}), {1, 2})
        for calibration in [None, self.calibration(samples=[]), self.calibration()]:
            with self.subTest(calibration=calibration):
                model = self.model(calibration)
                self.assertEqual(model.select_cpu(1, rows, {}, {1, 2, 3}), frozenset())

    def test_select_cpu_validates_rows_mapping_and_rejected(self):
        model = self.model(self.calibration())
        valid = (1, [[0, 0, 1, 1]], {0: 0}, {1})
        self.assertEqual(model.select_cpu(*valid), {1})
        cases = [
            (True, valid[1], {0: 0}, {1}),
            (4, valid[1], {0: 0}, {1}),
            (1, None, {0: 0}, {1}),
            (1, [None], {0: 0}, {1}),
            (1, [], {0: 0}, {1}),
            (1, valid[1] * 2, {0: 0}, {1}),
            (1, [[0, 0, 1, 1, 1]], {0: 0}, {1}),
            (1, [[0, 0, 1, True]], {0: 0}, {1}),
            (1, [[0, 0, 1, 1.0]], {0: 0}, {1}),
        ]
        for mapping in [None, [], {False: 0}, {-1: 0}, {0: True}, {0: -1}]:
            cases.append((1, valid[1], mapping, {1}))
        for rejected in [None, 1, [True], [1.0], [-1], [2], [[1]], {0, 1}]:
            cases.append((1, valid[1], {0: 0}, rejected))
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(model.select_cpu(*args), frozenset())
        # Unrequested resident mappings are harmless; hits cannot be CPU.
        self.assertEqual(model.select_cpu(1, valid[1], {0: 0, 9: 1}, {1}), {1})
        self.assertEqual(model.select_cpu(1, valid[1], {0: 0, 1: 1}, set()), set())

    def test_full_rejected_profile_prevents_reusing_gpu_transfer_costs(self):
        sample = dict(
            tokens=3,
            hit_routes=9,
            miss_routes=1,
            miss_experts=1,
            miss_tokens=[1],
            rejected_tokens=[1],
            cpu_us=80,
            gpu_us=100,
        )
        rows = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        model = self.model(self.calibration(samples=[sample]))
        self.assertEqual(model.select_cpu(3, rows, dict(enumerate(range(9))), {9}), {9})
        # Same 9 GPU routes: five admitted plus four rejected GPU transfers
        # cannot reuse a sample with nine cached GPU routes.
        self.assertEqual(model.select_cpu(3, rows, {}, {5, 6, 7, 8, 9}), set())
        self.assertEqual(model.choose(3, 9, 1, 1, [1], [1] * 5), "gpu")
        other = dict(sample, rejected_tokens=[1] * 5, cpu_us=95)
        model = self.model(self.calibration(samples=[sample, other]))
        self.assertEqual(model.choose(3, 9, 1, 1, [1], [1]), "cpu")
        self.assertEqual(model.choose(3, 9, 1, 1, [1], [1] * 5), "gpu")
        # Equal rejected totals with different distributions also stay distinct.
        self.assertEqual(model.choose(3, 9, 1, 1, [1], [1, 4]), "gpu")

    def test_rejected_profile_validation_and_multiset_subset(self):
        invalid = [
            None,
            (),
            "2",
            2,
            [],
            [0],
            [-1],
            [True],
            [2.0],
            [math.nan],
            [math.inf],
            [[]],
            [1],
            [1, 1],
            [3, 2],
            [2, 3],
        ]
        for profile in invalid:
            with self.subTest(profile=profile):
                calibration = self.calibration()
                calibration["samples"][0]["rejected_tokens"] = profile
                with self.assertRaisesRegex(ValueError, "rejected_tokens"):
                    self.model(calibration)
        calibration = self.calibration()
        del calibration["samples"][0]["rejected_tokens"]
        with self.assertRaisesRegex(ValueError, "rejected_tokens"):
            self.model(calibration)
        model = self.model(self.calibration())
        for profile in invalid[1:]:
            with self.subTest(runtime_profile=profile):
                self.assertEqual(model.choose(1, 2, 2, 1, [2], profile), "gpu")
        self.assertEqual(model.choose(1, 2, 2, 1, [2], iter([2])), "cpu")
        # Set inclusion is insufficient: two selected one-route experts need
        # two occurrences of 1 in the full rejected profile.
        sample = dict(
            tokens=1,
            hit_routes=2,
            miss_routes=2,
            miss_experts=2,
            miss_tokens=[1, 1],
            rejected_tokens=[1, 2],
            cpu_us=80,
            gpu_us=100,
        )
        with self.assertRaisesRegex(ValueError, "multiset subset"):
            self.model(self.calibration(samples=[sample]))
        self.assertEqual(model.choose(1, 2, 2, 2, [1, 1], [1, 2]), "gpu")


class TestHybridPolicy(CustomTestCase):
    def test_eviction_precedes_invisible_fill(self):
        p = GlobalLfu(2, 8, [1, 1], admission=0)
        p.seed(0, {0: 0})
        p.frequency[0][0] = 0.8
        epoch, snapshot = p.begin(0)
        with self.assertRaises(RuntimeError):
            p.plan(0, epoch, [1])
        p.release(0, epoch)
        evicted, fills = p.plan(0, epoch, [1])
        self.assertEqual(evicted, [(0, 0)])
        self.assertEqual(fills, [(1, 0)])
        self.assertEqual(snapshot, {0: 0})
        self.assertEqual(p.layers[0].mapping, {})
        with self.assertRaises(RuntimeError):
            p.begin(0)
        p.commit(0, epoch, fills)
        self.assertEqual(p.begin(0)[1], {1: 0})

    def test_global_replacement_uses_colder_other_layer_after_reader_retirement(self):
        p = GlobalLfu(2, 8, [1, 1], admission=0)
        p.seed(0, {0: 0})
        p.seed(1, {2: 1})
        p.frequency[0][0] = 10
        p.frequency[1][2] = 0.8
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        evicted, fills = p.plan(0, epoch, [3], safe_layers={0, 1})
        self.assertEqual(evicted, [(2, 1)])
        self.assertEqual(fills, [(3, 1)])
        p.commit(0, epoch, fills)
        self.assertEqual(p.layers[0].mapping, {0: 0, 3: 1})
        self.assertFalse(p.layers[1].mapping)
        self.assertEqual(p.owner, [0, 0])

    def test_ten_global_candidates_replace_ten_colder_other_layer_experts(self):
        p = GlobalLfu(2, 16, [1, 10], admission=0)
        p.seed(0, {0: 0})
        p.seed(1, {e: e for e in range(10)})
        p.frequency[0][0] = 20
        p.frequency[1][:10] = [0.8] * 10
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        _, fills = p.plan(0, epoch, range(1, 11), safe_layers={0, 1})
        p.commit(0, epoch, fills)
        self.assertEqual(len(fills), 10)
        self.assertFalse(p.layers[1].mapping)
        self.assertEqual(len(p.layers[0].mapping), 11)
        self.assertEqual(p.owner, [0] * 11)

    def test_global_replacement_rejects_unretired_other_layer(self):
        p = GlobalLfu(2, 8, [1, 1], admission=0)
        p.seed(1, {2: 1})
        p.begin(1)
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        with self.assertRaisesRegex(RuntimeError, "still computing"):
            p.plan(0, epoch, [3], safe_layers={0, 1})
        self.assertEqual(p.layers[1].mapping, {2: 1})

    def test_locked_layer_cannot_be_stolen(self):
        p = GlobalLfu(2, 8, [1, 1], admission=0)
        p.seed(1, {2: 1})
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        _, fills = p.plan(0, epoch, [0, 1, 3])
        p.commit(0, epoch, fills)
        self.assertEqual(len(fills), 1)
        self.assertEqual(p.layers[1].mapping, {2: 1})
        self.assertEqual(p.owner[1], 1)

    def test_stale_epoch_and_noop_clean(self):
        p = GlobalLfu(1, 4, [0])
        epoch, _ = p.begin(0)
        with self.assertRaises(RuntimeError):
            p.release(0, epoch - 1)
        p.release(0, epoch)
        _, fills = p.plan(0, epoch, [])
        with self.assertRaises(RuntimeError):
            p.commit(0, epoch - 1, fills)
        p.commit(0, epoch, fills)
        self.assertEqual(p.begin(0)[0], epoch + 1)

    def test_cpu_misses_drive_lfu_and_admission(self):
        p = GlobalLfu(1, 4, [1])
        p.seed(0, {0: 0})
        p.frequency[0][0] = 1.0
        for iteration in range(2):
            epoch, _ = p.begin(0)
            p.release(0, epoch)
            _, fills = p.plan(0, epoch, [2, 2, 2])
            self.assertEqual(len(fills), iteration)
            p.commit(0, epoch, fills)
        self.assertAlmostEqual(p.frequency[0][2], 1.95)
        self.assertIn(2, p.layers[0].mapping)

    def test_first_touch_uses_empty_slot_without_probation(self):
        p = GlobalLfu(2, 8, [1, 0])
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        self.assertEqual(p.plan(0, epoch, [3]), ([], [(3, 0)]))
        self.assertNotIn(3, p.layers[0].mapping)
        p.commit(0, epoch, [(3, 0)])
        self.assertEqual(p.layers[0].mapping, {3: 0})

    def test_new_arena_reservation_relocates_residents_even_inside_prefix(self):
        p = GlobalLfu(2, 8, [4, 4])
        p.seed(1, {0: 0, 1: 1, 2: 2, 3: 3})
        p.frequency[1][:4] = [1, 2, 3, 4]
        plan = p.plan_resize(6, reserved={0, 3, 4, 5})
        self.assertEqual(set(key for key in plan.keys if key), {(1, 2), (1, 3)})
        self.assertTrue(all(plan.keys[slot] is None for slot in plan.reserved))
        payload = {slot: key for slot, key in enumerate(p.keys)}
        before = dict(payload)
        for old, new in plan.moves:
            payload[new] = before[old]
        p.commit_resize(plan)
        self.assertEqual(p.reserved, {0, 3, 4, 5})
        for expert, slot in p.layers[1].mapping.items():
            self.assertEqual(payload[slot], (1, expert))

    def test_only_released_empty_ownership_can_move(self):
        p = GlobalLfu(2, 8, [2, 0], admission=0)
        epoch, _ = p.begin(1)
        p.release(1, epoch)
        _, fills = p.plan(1, epoch, [3])
        self.assertEqual(fills, [])
        p.commit(1, epoch, fills)
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        _, fills = p.plan(0, epoch, [])
        p.commit(0, epoch, fills)
        self.assertEqual(p.owner, [None, None])
        epoch, _ = p.begin(1)
        p.release(1, epoch)
        _, fills = p.plan(1, epoch, [3])
        p.commit(1, epoch, fills)
        self.assertEqual(p.owner[0], 1)

    def test_cold_eviction_releases_global_slot(self):
        p = GlobalLfu(1, 4, [1])
        p.seed(0, {0: 0})
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        evicted, fills = p.plan(0, epoch, [2])
        self.assertEqual(evicted, [(0, 0)])
        self.assertEqual(fills, [])
        self.assertEqual(p.owner, [None])
        p.commit(0, epoch, fills)

    def test_commit_rejects_wrong_reservation(self):
        p = GlobalLfu(2, 4, [1, 1])
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        with self.assertRaises(RuntimeError):
            p.commit(0, epoch, [(1, 1)])

    def test_resize_selects_global_heat_across_layer_boundaries(self):
        p = GlobalLfu(3, 8, [2, 2, 2])
        for layer in range(3):
            p.seed(layer, {0: 0, 1: 1})
            p.frequency[layer][:2] = [layer * 10 + 1, layer * 10 + 2]
        plan = p.plan_resize(3)
        self.assertEqual(set(plan.keys), {(1, 1), (2, 0), (2, 1)})
        self.assertEqual(len(plan.moves), 3)
        # No policy change before copies finish.
        self.assertEqual(len(p.keys), 6)
        p.commit_resize(plan)
        self.assertEqual(p.layers[0].mapping, {})
        self.assertEqual(len(p.layers[2].mapping), 2)
        self.assertEqual(len(p.keys), 3)

    def test_growth_preserves_payload_addresses_and_creates_unowned_slots(self):
        p = GlobalLfu(2, 8, [1, 1])
        p.seed(0, {2: 0})
        p.seed(1, {3: 1})
        before = [dict(state.mapping) for state in p.layers]
        plan = p.plan_resize(7)
        self.assertEqual(plan.moves, ())
        self.assertEqual(plan.evicted, ())
        p.commit_resize(plan)
        self.assertEqual([state.mapping for state in p.layers], before)
        self.assertEqual(p.owner[2:], [None] * 5)

    def test_resize_preserves_surviving_prefix_without_moves(self):
        p = GlobalLfu(2, 8, [2, 2])
        p.seed(0, {0: 0, 1: 1})
        p.seed(1, {0: 2, 1: 3})
        p.frequency[0][:2] = [9, 8]
        p.frequency[1][:2] = [7, 6]
        plan = p.plan_resize(2)
        self.assertEqual(plan.moves, ())
        p.commit_resize(plan)
        self.assertEqual(p.layers[0].mapping, {0: 0, 1: 1})
        self.assertEqual(p.layers[1].mapping, {})

    def test_resize_rejects_computing_managing_and_stale_plan(self):
        p = GlobalLfu(1, 4, [2])
        epoch, _ = p.begin(0)
        with self.assertRaises(RuntimeError):
            p.plan_resize(1)
        p.release(0, epoch)
        with self.assertRaises(RuntimeError):
            p.plan_resize(1)
        p.commit(0, epoch, [])
        plan = p.plan_resize(1)
        p.commit_resize(p.plan_resize(3))
        with self.assertRaises(RuntimeError):
            p.commit_resize(plan)

    def test_resize_moves_keep_all_weight_bytes_through_stage_cycles(self):
        p = GlobalLfu(4, 8, [4] * 4)
        for layer in range(4):
            p.seed(layer, {e: e for e in range(4)})
            p.frequency[layer][:4] = [((layer + 1) * 13 + e * 7) % 19 for e in range(4)]
        payload = [bytes([key[0], key[1]]) * 32 for key in p.keys]
        expected = {key: value for key, value in zip(p.keys, payload)}
        for capacity in (11, 16, 6, 14, 4):
            old = list(payload)
            plan = p.plan_resize(capacity)
            payload += [None] * max(0, capacity - len(payload))
            for source, destination in plan.moves:
                self.assertGreaterEqual(source, capacity)
                self.assertLess(destination, capacity)
                payload[destination] = payload[source]
            for slot, key in enumerate(plan.keys):
                if key is not None:
                    self.assertEqual(payload[slot], expected[key])
            p.commit_resize(plan)
            payload = payload[:capacity]
            for state in p.layers:
                for expert, slot in state.mapping.items():
                    self.assertLess(slot, capacity)
            self.assertEqual(
                len(old) < capacity, not plan.evicted and capacity > len(old)
            )

    def test_b12x_zero_slot_is_not_admitted_or_resize_destination(self):
        p = GlobalLfu(2, 8, [2, 2], admission=0)
        p.reserved.add(0)
        p.set_owner(0, -1)
        p.seed(1, {1: 2, 2: 3})
        p.frequency[1][1:3] = [8, 9]
        plan = p.plan_resize(2)
        self.assertIsNone(plan.keys[0])
        self.assertEqual(plan.keys[1], (1, 2))
        self.assertEqual(plan.moves, ((3, 1),))
        p.commit_resize(plan)
        self.assertEqual(p.owner[0], -1)
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        self.assertEqual(p.plan(0, epoch, [3, 4]), ([], []))
        p.commit(0, epoch, [])
        self.assertEqual(p.owner[0], -1)

    def test_deterministic_pressure_preserves_ownership(self):
        p = GlobalLfu(4, 16, [4] * 4, admission=0)
        for step in range(120):
            layer = step % 4
            epoch, _ = p.begin(layer)
            before = [dict(s.mapping) for s in p.layers]
            p.release(layer, epoch)
            _, fills = p.plan(layer, epoch, [(step * 7) % 16, (step * 11 + 3) % 16])
            p.commit(layer, epoch, fills)
            slots = []
            for idx, state in enumerate(p.layers):
                if idx != layer:
                    self.assertEqual(state.mapping, before[idx])
                for expert, slot in state.mapping.items():
                    self.assertEqual(p.owner[slot], idx)
                    self.assertEqual(p.keys[slot], (idx, expert))
                    slots.append(slot)
            self.assertEqual(len(slots), len(set(slots)))
            self.assertEqual(
                p.free_slots, {s for s, owner in enumerate(p.owner) if owner is None}
            )
            for layer in range(4):
                self.assertEqual(
                    p.owned_slots[layer],
                    {s for s, owner in enumerate(p.owner) if owner == layer},
                )


if __name__ == "__main__":
    unittest.main()
