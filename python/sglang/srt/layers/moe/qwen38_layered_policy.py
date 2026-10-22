# SPDX-License-Identifier: Apache-2.0
"""Measured workload buckets for the per-layer LFU CPU bypass.

Schema 3 retains exact route masks. Schema 4 follows the global cost policy:
expert identities, layer numbers and cache capacities do not identify work.
It retains route counts, distinct-token counts (native CPU work), GPU expert
count and shared-expert overlap. Every authorized bucket must win on multiple
representatives; source/config fingerprints still bind the execution path.
"""

import hashlib
import json
from pathlib import Path

COLUMNS = 47


def workload_bucket(tokens, top_k, active, valid_routes, overlap, profiles):
    """Canonical cost key, also used by the serving calibration harness.

    Prefix order is (distinct tokens, route count, expert ID). The first two
    determine cost; the final tie breaker only determines expert identity.
    Five bits hold route multiplicity, bounded by tokens * top_k <= 30.
    """
    work = sorted(
        (sum(bool((p >> (4 * t)) & 15) for t in range(tokens)) << 5)
        | sum((p >> (4 * t)) & 15 for t in range(tokens))
        for p in profiles
    )
    return (tokens, top_k, active, valid_routes, bool(overlap), tuple(work))


def signature(
    tokens,
    top_k,
    misses,
    active,
    valid_routes,
    layer,
    slots,
    overlap,
    valid_counts,
    hit_counts,
    profiles,
):
    state = 1469598103934665603
    for value in (
        tokens,
        top_k,
        misses,
        active,
        valid_routes,
        layer,
        slots,
        overlap,
        *valid_counts,
        *hit_counts,
        *profiles,
    ):
        state = ((state ^ (value & 0xFFFFFFFF)) * 1099511628211) & ((1 << 64) - 1)
    return state


def calibration_rows(source, metadata):
    """Only accept measurements from this path and this exact binary/config."""
    if source is None:
        return []
    data = (
        json.loads(Path(source).read_text())
        if isinstance(source, (str, Path))
        else source
    )
    schema = data.get("schema_version")
    if schema not in (3, 4) or data.get("metadata") != metadata:
        raise ValueError("layered CPU calibration metadata mismatch")
    rows = []
    seen = set()
    for sample in data.get("samples", []):
        tokens = sample["tokens"]
        top_k = sample["top_k"]
        profiles = tuple(sample["candidate_route_profiles"])
        valid_counts = tuple(sample["valid_route_counts"])
        hit_counts = tuple(sample["hit_route_counts"])
        token_counts = [
            sum((profile >> (4 * t)) & 15 != 0 for t in range(sample["tokens"]))
            for profile in profiles
        ]
        if token_counts != sorted(token_counts):
            raise ValueError("candidate profiles must follow token-count order")
        misses = len(profiles)
        active = sample["active_experts"]
        valid_routes = sample["valid_routes"]
        selected = sample["cpu_prefix"]
        layer = sample["layer"]
        slots = sample["cache_slots"]
        overlap = sample["shared_fill_overlap"]
        if (
            not 1 <= tokens <= 3
            or not 1 <= top_k <= 10
            or not 0 <= misses <= min(30, tokens * top_k)
            or not misses <= active <= valid_routes <= tokens * top_k
            or len(valid_counts) != tokens
            or len(hit_counts) != tokens
            or any(
                not 0 <= hit <= valid <= top_k
                for valid, hit in zip(valid_counts, hit_counts)
            )
            or sum(valid_counts) != valid_routes
            or sum(hit_counts) < active - misses
            or any(
                sum((profile >> (4 * t)) & 15 for profile in profiles)
                != valid_counts[t] - hit_counts[t]
                for t in range(tokens)
            )
            or len(profiles) != misses
            or any(
                not 1 <= profile < (1 << (4 * tokens))
                or any((profile >> (4 * t)) & 15 > top_k for t in range(tokens))
                for profile in profiles
            )
            or not 0 <= selected <= misses
            or not 0 <= layer < metadata["layers"]
            or type(overlap) is not bool
            or not active <= slots <= metadata["experts"]
            or (
                misses
                and sum(
                    sum((profile >> (4 * t)) & 15 for t in range(tokens))
                    for profile in profiles
                )
                == 0
            )
        ):
            raise ValueError("invalid layered CPU calibration sample")
        if selected:
            baseline = sample["baseline_us"]
            measured = sample["selected_us"]
            absolute = sample["minimum_saving_us"]
            relative = sample["minimum_saving_fraction"]
            if not (
                baseline > 0
                and measured > 0
                and absolute >= 0
                and 0 <= relative < 1
                and baseline - measured > absolute
                and (baseline - measured) / baseline > relative
                and sample["selected_p95_us"] < sample["baseline_p95_us"]
            ):
                raise ValueError("unprofitable layered CPU calibration sample")
        if schema == 4:
            if (
                type(sample.get("representatives")) is not int
                or sample["representatives"] < 3
            ):
                raise ValueError(
                    "workload buckets require at least three measured representatives"
                )
            # Sentinel ownership fields distinguish schema 4 on device. The
            # full normalized key is compared after hash lookup as in schema 3.
            *_, profiles = workload_bucket(
                tokens, top_k, active, valid_routes, overlap, profiles
            )
            layer = slots = -1
            valid_counts = hit_counts = (0,) * tokens
        key = (
            tokens,
            top_k,
            active,
            valid_routes,
            layer,
            slots,
            overlap,
            valid_counts,
            hit_counts,
            profiles,
        )
        if key in seen:
            raise ValueError("duplicate layered CPU calibration key")
        seen.add(key)
        padded_valid = valid_counts + (0,) * (3 - tokens)
        padded_hit = hit_counts + (0,) * (3 - tokens)
        sig = signature(
            tokens,
            top_k,
            misses,
            active,
            valid_routes,
            layer,
            slots,
            int(overlap),
            valid_counts,
            hit_counts,
            profiles,
        )
        rows.append(
            [
                sig & 0xFFFFFFFF,
                sig >> 32,
                misses,
                selected,
                layer,
                slots,
                *padded_valid,
                *padded_hit,
                *profiles,
                *([0] * (30 - misses)),
                tokens,
                top_k,
                active,
                valid_routes,
                int(overlap),
            ]
        )
    if len(rows) > 4096:
        raise ValueError("layered CPU calibration exceeds bounded table")
    rows.sort(key=lambda row: ((row[1] & 0xFFFFFFFF), (row[0] & 0xFFFFFFFF)))
    return [[v - (1 << 32) if v >= (1 << 31) else v for v in row] for row in rows]


def fingerprint(*sources):
    digest = hashlib.sha256()
    for source in sources:
        digest.update(Path(source).read_bytes())
    return digest.hexdigest()


def execution_metadata(cpu, runtime, parallel_miss_plan):
    """Bind a cost table to the loaded native code and full-branch schedule."""
    return dict(
        cpu_library_fingerprint=(
            fingerprint(cpu.library_path) if cpu is not None else None
        ),
        fill_cta=runtime.miss_mailbox.fill_cta,
        cache_overlap_policy=runtime.cache_overlap_policy,
        shared_fill_overlap_mode="per_call_key",
        resident_alphas=runtime.resident_alphas,
        parallel_miss_plan=parallel_miss_plan,
    )
