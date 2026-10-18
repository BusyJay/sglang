#!/usr/bin/env python3
"""Summarize live Kineto traces without treating overlapping GPU work as serial.

python3 analyze_trace.py live-24steps.trace.json.gz --output live-analysis.json
All times are milliseconds. Kernel groups are name-based, not hardware counters.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersect(left, right):
    left, right = merged(left), merged(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def kernel_group(name):
    name = name.lower()
    if 'mailbox_fill_kernel' in name:
        return 'expert_fill_from_mapped_host'
    if 'decode_plan_kernel' in name:
        return 'expert_cache_planner'
    if 'b12x' in name and 'smallm' in name:
        return 'b12x_expert_compute'
    if 'quantize' in name and 'mxfp8' in name:
        return 'mxfp8_quantize'
    if 'gemm' in name and 'mxfp8' in name:
        return 'mxfp8_gemm'
    if 'hc_mix' in name or 'hc_combine' in name:
        return 'hyperconnection'
    if any(s in name for s in ('kernel_mha', 'qsa', 'sparse_gqa')):
        return 'qsa_attention_and_indexer'
    if 'delta_rule' in name or 'causal_conv' in name:
        return 'gdn_and_short_conv'
    return 'other_kernels'


def analyze(trace):
    opener = gzip.open if str(trace).endswith('.gz') else open
    with opener(trace, 'rt') as stream:
        events = [e for e in json.load(stream)['traceEvents']
                  if e.get('ph') == 'X' and 'dur' in e]
    cpu_steps = [e for e in events if e.get('cat') == 'user_annotation'
                 and e.get('name', '').startswith('step[VERIFY')]
    if not cpu_steps:
        raise ValueError('No target VERIFY annotations')
    cpu_steps.sort(key=lambda e: e['ts'])
    gpu_annotations = defaultdict(list)
    for e in events:
        if e.get('cat') == 'gpu_user_annotation':
            gpu_annotations[e.get('args', {}).get('External id')].append(e)
    gpu_windows, step_rows = [], []
    for step in cpu_steps:
        spans = gpu_annotations[step.get('args', {}).get('External id')]
        if not spans:
            raise ValueError('No correlated GPU annotations; refusing CPU-window fallback')
        start = min(e['ts'] for e in spans)
        end = max(e['ts'] + e['dur'] for e in spans)
        gpu_windows.append((start, end))
        step_rows.append({'name': step['name'], 'cpu_ms': step['dur'] / 1000,
                          'gpu_envelope_ms': (end - start) / 1000})
    cpu_windows = [(e['ts'], e['ts'] + e['dur']) for e in cpu_steps]
    gpu_windows = merged(gpu_windows)
    cpu_windows = merged(cpu_windows)
    groups, rows = defaultdict(list), defaultdict(lambda: [0, 0.0])
    gpu_work, runtime, cpu_ops = [], defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    for e in events:
        cat = e.get('cat')
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            clipped = intersect([(e['ts'], e['ts'] + e['dur'])], gpu_windows)
            elapsed = duration(clipped)
            if not elapsed:
                continue
            group = kernel_group(e['name']) if cat == 'kernel' else cat
            groups[group].extend(clipped)
            gpu_work.extend(clipped)
            row = rows[(group, e['name'])]
            row[0] += 1
            row[1] += elapsed / 1000
        elif cat in ('cpu_op', 'cuda_runtime', 'cuda_driver'):
            elapsed = duration(intersect([(e['ts'], e['ts'] + e['dur'])], cpu_windows))
            if elapsed:
                table = cpu_ops if cat == 'cpu_op' else runtime
                row = table[e['name']]
                row[0] += 1
                row[1] += elapsed / 1000
    count = len(cpu_steps)
    gpu_us = duration(gpu_windows)
    fill = groups['expert_fill_from_mapped_host']
    others = [interval for group, intervals in groups.items()
              if group != 'expert_fill_from_mapped_host' for interval in intervals]
    fill_union = duration(fill)
    fill_overlap = duration(intersect(fill, others))
    phases = {}
    for phase in ('draft', 'draft_extend'):
        spans = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name') == phase]
        gpu_durations = []
        for e in spans:
            gpu = gpu_annotations[e.get('args', {}).get('External id')]
            if gpu:
                gpu_durations.append(max(g['ts'] + g['dur'] for g in gpu) - min(g['ts'] for g in gpu))
        phases[phase] = {'calls': len(spans), 'mean_cpu_ms': statistics.mean(e['dur'] for e in spans) / 1000 if spans else None,
                         'mean_gpu_envelope_ms': statistics.mean(gpu_durations) / 1000 if gpu_durations else None}
    def top(table):
        return sorted([{'name': name, 'calls': values[0], 'inclusive_ms_per_verify': values[1] / count}
                       for name, values in table.items()], key=lambda r: r['inclusive_ms_per_verify'], reverse=True)[:20]
    return {
        'trace': Path(trace).name, 'target_verify_steps': count,
        'mean_cpu_verify_ms': statistics.mean(e['cpu_ms'] for e in step_rows),
        'mean_gpu_verify_envelope_ms': statistics.mean(e['gpu_envelope_ms'] for e in step_rows),
        'mean_verify_start_interval_ms': statistics.mean(b['ts'] - a['ts'] for a, b in zip(cpu_steps, cpu_steps[1:])) / 1000 if count > 1 else None,
        'target_gpu_envelope_union_ms': gpu_us / 1000,
        'target_gpu_active_union_ms': duration(gpu_work) / 1000,
        'target_gpu_active_fraction': duration(gpu_work) / gpu_us,
        'fill_union_ms_per_verify': fill_union / count / 1000,
        'fill_overlap_with_other_gpu_ms_per_verify': fill_overlap / count / 1000,
        'fill_without_other_traced_gpu_work_ms_per_verify': (fill_union - fill_overlap) / count / 1000,
        'groups': sorted([{'name': group, 'calls': sum(v[0] for (g, _), v in rows.items() if g == group),
                          'summed_ms_per_verify': sum(v[1] for (g, _), v in rows.items() if g == group) / count,
                          'union_ms_per_verify': duration(intervals) / count / 1000,
                          'fraction_of_gpu_envelope': duration(intervals) / gpu_us}
                         for group, intervals in groups.items()], key=lambda r: r['union_ms_per_verify'], reverse=True),
        'top_kernels': sorted([{'group': group, 'name': name, 'calls': values[0],
                               'summed_ms_per_verify': values[1] / count}
                              for (group, name), values in rows.items()], key=lambda r: r['summed_ms_per_verify'], reverse=True)[:30],
        'cpu_ops_inclusive': top(cpu_ops), 'cuda_api_inclusive': top(runtime),
        'draft_phases': phases, 'steps': step_rows,
        'limits': ['CPU operations and CUDA APIs are inclusive and nested; do not add their times.',
                   'Different GPU groups overlap; their percentages are not additive.',
                   'No-overlap fill is a timeline observation, not a counterfactual speedup or critical-path proof.',
                   'GPU active union includes mapped-host load latency; it is not achieved compute utilization.',
                   'No PCIe throughput, DRAM bandwidth, achieved occupancy, or cache miss count is inferred from duration alone.',
                   'Profiler overhead and live requests prevent direct throughput comparison with the earlier controlled benchmark.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ('top_kernels', 'cpu_ops_inclusive', 'cuda_api_inclusive', 'steps', 'limits')}, indent=2))


if __name__ == '__main__':
    main()
