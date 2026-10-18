"""Summarize the fixed-prompt migration measurements."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
summary = {}
for name in ('old', 'new-triton', 'new-b12x'):
    files = sorted(root.glob(f'{name}-round*.json'))
    rows = [row for path in files for row in json.loads(path.read_text())['cases']]
    if not rows:
        continue
    summary[name] = {
        'samples': len(rows),
        'median_decode_tokens_per_second': statistics.median(r['decode_tokens_per_second'] for r in rows),
        'median_e2e_tokens_per_second': statistics.median(r['e2e_tokens_per_second'] for r in rows),
        'median_accept_length': statistics.median(r['meta_info']['spec_accept_length'] for r in rows),
        'median_ttft_s': statistics.median(r['ttft_s'] for r in rows),
        'cases': [{'case': r['case'], 'decode_tokens_per_second': r['decode_tokens_per_second'],
                   'accept_length': r['meta_info']['spec_accept_length']} for r in rows],
    }
if 'old' in summary:
    for row in summary.values():
        row['decode_change_vs_old_percent'] = 100 * (row['median_decode_tokens_per_second'] / summary['old']['median_decode_tokens_per_second'] - 1)
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for name, row in summary.items():
    print(name, {k: v for k, v in row.items() if k != 'cases'})
