#!/usr/bin/env python3
"""Summarize an nsys SQLite export using only Python's standard library."""

import argparse
import collections
import json
import sqlite3
from pathlib import Path


def union(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if right is None or start > right:
            total += end - start
            right = end
        elif end > right:
            total += end - right
            right = end
    return total


def summarize(connection):
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute("select id,value from StringIds"))
    kernels = [
        dict(r)
        for r in connection.execute(
            "select start,end,shortName,demangledName,streamId from "
            "CUPTI_ACTIVITY_KIND_KERNEL order by start"
        )
    ]
    copies = [
        dict(r)
        for r in connection.execute(
            "select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start"
        )
    ]
    activities = kernels + copies
    lo = min(x["start"] for x in activities)
    hi = max(x["end"] for x in activities)
    plans = [k for k in kernels if strings[k["shortName"]] == "decode_plan_kernel"]

    def window(start, end):
        selected = [x for x in kernels if start <= x["start"] < end]
        transfers = [x for x in copies if start <= x["start"] < end]
        groups = collections.defaultdict(lambda: dict(calls=0, total_ms=0))
        for k in selected:
            group = groups[strings[k["demangledName"]]]
            group["calls"] += 1
            group["total_ms"] += (min(k["end"], end) - k["start"]) / 1e6
        copy_groups = collections.defaultdict(
            lambda: dict(calls=0, bytes=0, total_ms=0)
        )
        for k in transfers:
            group = copy_groups[k["copyKind"]]
            group["calls"] += 1
            group["bytes"] += k["bytes"]
            group["total_ms"] += (min(k["end"], end) - k["start"]) / 1e6
        active = union((x["start"], min(x["end"], end)) for x in selected + transfers)
        return dict(
            start_ns=start,
            end_ns=end,
            span_ms=(end - start) / 1e6,
            idle_ms=(end - start - active) / 1e6,
            copy_kind_totals=dict(copy_groups),
            kernels=[
                dict(name=k, **v)
                for k, v in sorted(
                    groups.items(), key=lambda item: -item[1]["total_ms"]
                )
            ],
        )

    result = dict(
        capture_local_time=connection.execute(
            "select localTime from TARGET_INFO_SESSION_START_TIME"
        ).fetchone()[0],
        planner_calls=len(plans),
        full_gpu_activity_window=window(lo, hi),
    )
    if plans and plans[0]["start"] > lo:
        result["before_first_decode_planner"] = window(lo, plans[0]["start"])
    steps = (len(plans) - 1) // 48
    if steps > 0:
        steady = window(plans[0]["start"], plans[steps * 48]["start"])
        steady.update(complete_steps=steps, ms_per_step=steady["span_ms"] / steps)
        result["complete_decode_steps"] = steady
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with sqlite3.connect(
        args.sqlite.resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        result = summarize(connection)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
