#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--report', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
report = json.loads(Path(args.report).read_text())
passed = (report.get('status') == 'PASS'
          and report.get('saved_gpu_bytes', 0) >= report.get('host_weight_bytes', 1) * .99
          and all(c['parity'] and c['graph_parity'] for c in report.get('cases', [])))
# 0.1ms per small embedding invocation, 5% ratio at prefill sizes.
latency = all(c['added_us'] <= 100 if c['tokens'] <= 3
              else c['mapped_us'] <= c['gpu_us'] * 1.05
              for c in report.get('cases', []))
result = dict(status='PASS' if passed and latency else ('GPU_REQUIRED' if not report.get('cuda_available') else 'FAIL'),
              memory_and_parity_pass=passed, embedding_latency_pass=latency and passed,
              production_ready=False,
              conclusion='GPU microchecks only; compare real service TTFT/TPOT/throughput and startup memory before rollout.',
              report=report)
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
raise SystemExit(0 if passed and latency else 2)
