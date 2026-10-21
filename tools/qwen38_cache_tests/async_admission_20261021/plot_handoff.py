"""Render one auditable decode timeline from the detailed Nsight SQLite export."""

import argparse
import json
import re
import sqlite3
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    scopes = [
        (a, b, text or strings.get(text_id, ""))
        for a, b, text, text_id in db.execute(
            "SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
        )
    ]
    kernels = [
        (a, b, strings[name])
        for a, b, name in db.execute(
            "SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    copies = list(db.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
    db.close()
    origin = min(a for a, _, _ in kernels)
    last_prefill = max(b for a, b, n in scopes if n.endswith("HybridCache.prefill"))
    candidates = []
    for a, b, name in scopes:
        match = re.fullmatch(r"hybrid.cpu/M=(\d+)/E=(\d+)", name)
        if match and a > last_prefill:
            candidates.append((a, b, int(match[2])))
    anchor = min((row for row in candidates if row[2] >= 8), default=min(candidates))
    start = anchor[0] - 3_000_000
    end = start + 35_000_000

    def selected(suffix):
        return [(a, b) for a, b, name in scopes if name.endswith(suffix)]

    tracks = [
        ("Main: remap", selected("HybridCache._remap"), "#007f86"),
        ("Main: GPU dispatch", selected("HybridCache._gpu"), "#52b0a8"),
        (
            "Main: pending result join",
            selected("hybrid.result_join.pending"),
            "#ca3d49",
        ),
        ("CPU native call (wall)", selected("hybrid.cpu.native"), "#df9238"),
        ("Manager: LFU plan", selected("GlobalLfu.plan"), "#9569bc"),
        (
            "GPU: cache-fill kernels",
            [(a, b) for a, b, n in kernels if "mailbox_fill" in n],
            "#88604c",
        ),
        (
            "GPU: other kernels",
            [(a, b) for a, b, n in kernels if "mailbox_fill" not in n],
            "#3677b4",
        ),
        ("GPU: memcpy", copies, "#40a86a"),
    ]
    plt.rcParams["svg.hashsalt"] = "qwen38-handoff"
    fig, ax = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
    for i, (name, spans, color) in enumerate(tracks):
        clipped = [
            ((max(a, start) - origin) / 1e6, (min(b, end) - max(a, start)) / 1e6)
            for a, b in spans
            if a < end and b > start
        ]
        ax.broken_barh(clipped, (i - 0.32, 0.64), facecolors=color)
    for a, b, experts in candidates:
        if start < a < end:
            ax.text((a - origin) / 1e6, 2.64, str(experts), fontsize=6, va="bottom")
    ax.axvline((anchor[0] - origin) / 1e6, color="#333333", ls="--", lw=0.6)
    ax.set_yticks(range(len(tracks)), [name for name, _, _ in tracks])
    ax.set_xlim((start - origin) / 1e6, (end - origin) / 1e6)
    ax.set_ylim(len(tracks) - 0.4, -0.65)
    ax.set_xlabel(
        "Milliseconds since first recorded GPU kernel; blank GPU tracks mean no recorded work"
    )
    ax.set_title(
        "Decode sample after the 1K prefill; numbers above CPU calls are distinct miss experts"
    )
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(args.output, metadata={"Date": None})
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "selection": "first CPU call with >=8 miss experts after the final prefill scope",
                "origin_ns": origin,
                "window_start_ns": start,
                "window_end_ns": end,
                "anchor_start_ns": anchor[0],
                "anchor_experts": anchor[2],
                "note": "CPU/native spans are wall time, not sampled active CPU cycles.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
