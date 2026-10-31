# SPDX-License-Identifier: Apache-2.0
"""Decode the actual xctrace Metal interval schema, including XML references.

Metal intervals can be nested and overlap across channels. Report interval
union, not summed durations, as observed device busy time. These intervals are
encoder/device activity; do not relabel them as individual shader kernels.
"""

import collections
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def union_ns(intervals):
    end = None
    total = 0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def analyze(xml_path, output, pid):
    root = ET.parse(xml_path).getroot()
    refs = {e.attrib["id"]: e for e in root.iter() if "id" in e.attrib}

    def resolve(element):
        while "ref" in element.attrib:
            element = refs[element.attrib["ref"]]
        return element

    def scalar(element):
        element = resolve(element)
        return element.text or element.attrib.get("fmt", "")

    events = []
    for node in root.findall("node"):
        schema = node.find("schema")
        if schema is None or schema.attrib.get("name") != "metal-gpu-intervals":
            continue
        columns = [e.findtext("mnemonic") for e in schema.findall("col")]
        for row in node.findall("row"):
            values = dict(zip(columns, list(row)))
            process = resolve(values["process"])
            pid_element = process.find("pid")
            if pid_element is None or int(scalar(pid_element)) != pid:
                continue
            start, duration = (
                int(scalar(values["start"])),
                int(scalar(values["duration"])),
            )
            label = resolve(values["event-label"]).attrib.get("fmt", "")
            events.append(
                dict(
                    name=label,
                    cat="metal_device_interval",
                    ph="X",
                    pid=pid,
                    tid=scalar(values["channel-name"]),
                    ts=start / 1000,
                    dur=duration / 1000,
                    args=dict(
                        depth=int(scalar(values["event-depth"])),
                        cpu_to_gpu_us=int(scalar(values["start-latency"])) / 1000,
                        command_buffer=scalar(values["cmdbuffer-id"]),
                    ),
                )
            )
    if not events:
        raise ValueError(
            f"No metal-gpu-intervals for PID {pid}; check TOC/schema and recording permissions"
        )
    groups = collections.defaultdict(list)
    for e in events:
        groups[(e["name"], e["tid"])].append(e)
    start = min(e["ts"] for e in events)
    stop = max(e["ts"] + e["dur"] for e in events)
    summary = dict(
        pid=pid,
        event_count=len(events),
        observed_span_ms=(stop - start) / 1000,
        interval_union_ms=union_ns([(e["ts"], e["ts"] + e["dur"]) for e in events])
        / 1000,
        evidence="device/encoder intervals, NOT per-shader kernel timing; includes startup and warmup",
        labels=[
            dict(
                label=k[0],
                channel=k[1],
                count=len(v),
                inclusive_ms=sum(e["dur"] for e in v) / 1000,
                median_cpu_to_gpu_us=statistics.median(
                    e["args"]["cpu_to_gpu_us"] for e in v
                ),
            )
            for k, v in groups.items()
        ],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "device.trace.json").write_text(
        json.dumps(dict(traceEvents=events, displayTimeUnit="ms")) + "\n"
    )
    (output / "device-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
