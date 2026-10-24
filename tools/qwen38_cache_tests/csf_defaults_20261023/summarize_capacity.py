"""Summarize retained serving evidence without treating aborted runs as passes."""
import csv
import json
import re
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for folder in sorted(root.glob('capacity-*')):
    path = folder / 'performance.json'
    if not path.is_file():
        continue
    data = json.loads(path.read_text())
    command = data.get('command', [])
    if '--model-loader-extra-config' not in command:
        continue
    loader = json.loads(command[command.index('--model-loader-extra-config') + 1])
    log = (folder / 'default.log').read_text(errors='replace')
    slots = re.search(r'"expert_loan_budgets":\s*(\[[^]]+\])', log)
    cases = data.get('modes', {}).get('default', {}).get('cases', [])
    status = data.get('status')
    if status != 'passed' and ('OutOfMemoryError' in log or 'CUDA out of memory' in log):
        status = 'failed_oom'
    qa = folder / 'chat-correctness.json'
    row = dict(run=folder.name, status=status,
               slots=json.loads(slots[1]) if slots else None,
               static=command[command.index('--mem-fraction-static') + 1],
               cache_budget_bytes=loader.get('cache_budget_bytes'),
               completed_cases=len(cases),
               verify_calls=sum(c.get('meta_info', {}).get('spec_verify_ct', 0) for c in cases),
               chat_status=json.loads(qa.read_text()).get('status') if qa.exists() else None,
               cases=[{k: c.get(k) for k in ('name','prompt_tokens','output_tokens','ttft_seconds','decode_tokens_per_second')} for c in cases])
    short = [c['decode_tokens_per_second'] for c in cases if c.get('name','').startswith('short_')]
    if short:
        row['median_short_tps'] = statistics.median(short)
    memory = root / (folder.name + '-memory.csv')
    if memory.exists():
        samples = [x for x in csv.reader(memory.open()) if len(x) >= 3 and x[1].strip().isdigit() and x[2].strip().isdigit()]
        if not samples:
            rows.append(row)
            continue
        row['sampled_peak_gpu_used_mib'] = max(int(x[1]) for x in samples)
        row['sampled_min_gpu_free_mib'] = min(int(x[2]) for x in samples)
    rows.append(row)
(root / 'capacity-summary.json').write_text(json.dumps(rows, indent=2) + '\n')
for row in rows:
    if '-half-' in row['run']:
        print(json.dumps(row, indent=2))
