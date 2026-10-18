#!/usr/bin/env python3
"""Extract inclusive B12X Python frame counts from a with_stack Kineto trace."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('trace', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
with gzip.open(args.trace, 'rt') as stream:
    events = json.load(stream)['traceEvents']
steps = sum(e.get('cat') == 'user_annotation' and e.get('name', '').startswith('step[VERIFY')
            for e in events)
if not steps:
    raise ValueError('No target VERIFY steps')
rows = defaultdict(lambda: [0, 0.0])
for event in events:
    if event.get('ph') == 'X' and event.get('cat') == 'python_function' and any(
        name in event['name'] for name in ('qwen38_cached_b12x', 'w4a16/kernel.py', 'compile_plan', '_lib/')
    ):
        row = rows[event['name']]
        row[0] += 1
        row[1] += event.get('dur', 0) / 1000
result = sorted([{'name': name, 'calls': values[0], 'inclusive_ms_per_verify': values[1] / steps}
                 for name, values in rows.items()], key=lambda r: r['inclusive_ms_per_verify'], reverse=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
