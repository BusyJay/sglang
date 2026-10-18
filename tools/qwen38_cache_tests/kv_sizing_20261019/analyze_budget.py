"""Separate real VMM backing from ordinary allocator slack at KV sizing."""

import gzip
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    rows = json.loads((root / "expert-row-bytes.json").read_text())
    granularity = 2 << 20  # Both MINIMUM and RECOMMENDED queried on both GPUs.
    result = {}
    for name, slots, staged, tiny in (
        ("unstaged", 6382, False, False),
        ("current", 6382, True, False),
        ("aux6388", 6388, True, True),
    ):
        events = json.loads((root / (name + "-budget-events.json")).read_text())
        event = next(r for r in events if r["stage"] == "kv.budget.after_empty_cache")
        with gzip.open(root / (name + "-budget-segments.json.gz"), "rt") as stream:
            segments = json.load(stream)
        segments = [s for s in segments if s["device"] == event["device_index"]]
        ordinary = [s for s in segments if s["segment_pool_id"] == [0, 0]]
        auxiliary = (
            {n for n, row in rows.items() if row * 48 * 512 < granularity}
            if tiny
            else set()
        )
        vmm = (
            sum(
                (row * slots + granularity - 1) // granularity * granularity
                for n, row in rows.items()
                if n not in auxiliary
            )
            if staged
            else 0
        )
        v = event["values"]
        reserved = sum(s["total_size"] for s in ordinary)
        active = sum(s["allocated_size"] for s in ordinary)
        inactive = sum(
            b["size"] for s in ordinary for b in s["blocks"] if b["state"] == "inactive"
        )
        pending = sum(
            b["size"]
            for s in ordinary
            for b in s["blocks"]
            if b["state"] == "active_awaiting_free"
        )
        numbers = dict(
            driver_free=v["driver_free_bytes"],
            ordinary_reserved=reserved,
            ordinary_allocated=active,
            ordinary_inactive=inactive,
            ordinary_pending_free=pending,
            vmm_physical=vmm,
            vmm_virtual_reserved=sum(
                s["total_size"] for s in segments if s["segment_pool_id"] != [0, 0]
            ),
            vmm_rounding=vmm
            - sum(row * slots for n, row in rows.items() if n not in auxiliary)
            if staged
            else 0,
            auxiliary_allocated=sum(rows[n] * 48 * 512 for n in auxiliary),
            outside_allocator=v["driver_total_bytes"]
            - v["driver_free_bytes"]
            - reserved
            - vmm,
        )
        result[name] = dict(
            base_slots=slots,
            bytes=numbers,
            MiB={k: n / 2**20 for k, n in numbers.items()},
        )
    before = result["current"]["bytes"]
    baseline = result["unstaged"]["bytes"]
    result["difference_bytes"] = dict(
        driver_free_lost=baseline["driver_free"] - before["driver_free"],
        vmm_rounding=before["vmm_rounding"],
        ordinary_inactive_increase=before["ordinary_inactive"]
        - baseline["ordinary_inactive"],
        outside_allocator_increase=before["outside_allocator"]
        - baseline["outside_allocator"],
    )
    (root / "budget-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
