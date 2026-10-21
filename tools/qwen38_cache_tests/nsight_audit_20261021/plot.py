"""Plot a deterministic median E=8 example from the independent audit."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.audit.read_text())
    candidates = sorted(
        (r for r in data["layers"] if r["miss_scope"] == "hybrid.cpu/M=3/E=8"),
        key=lambda r: r["host_us"],
    )
    row = candidates[len(candidates) // 2]
    origin = row["start_ns"]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")

    def bar(spans, y, color, label=None, height=0.62):
        for a, b, *_ in spans:
            ax.broken_barh(
                [((a - origin) / 1000, (b - a) / 1000)],
                (y - height / 2, height),
                facecolors=color,
                label=label,
            )
            label = None

    bar(row["remap"], 5, "#4788ba", "Host remap")
    bar(row["gpu_dispatch"], 5, "#8f70bb", "Host GPU dispatch")
    bar(row["join"], 5, "#d65a52", "Host result wait")
    bar(row["native"], 4, "#de9b36", "CPU native wall time")
    bar(
        row["native_cond_waits"],
        3,
        "#bf7422",
        "Coordinator waits (includes worker compute)",
    )
    for a, b, label, stream in row["gpu_activities"]:
        y = 2 if label == "MEMCPY" else 1
        bar([(a, b)], y, "#29a47a" if y == 1 else "#638fa6")
    bar(row["resident_kernel"], 1, "#075e43", "Resident expert GPU kernel")
    gap = row["before_resident_launch_gap"]
    bar([gap], 0, "#aaa", "Host gap before resident launch")
    ax.set_yticks(
        range(6),
        [
            "Pre-launch gap",
            "GPU kernels",
            "GPU copies",
            "CPU cond waits",
            "CPU native",
            "Main thread",
        ],
    )
    ax.set_xlim(0, row["host_us"])
    ax.set_xlabel("Microseconds from HybridCache.run entry")
    ax.set_title(
        f"Nsight raw trace: M=3, E=8, layer invocation {row['index']}\n"
        f"Absolute start: {origin / 1e9:.9f} s; correlated main-thread GPU work"
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(args.output, dpi=170)
    args.output.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
