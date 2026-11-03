"""Measure 300-token short requests on an already healthy single server."""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--tokens", type=int, default=300)
args = parser.parse_args()
root = Path('/Users/jay/workspace/sglang')
out = root / args.output
out.parent.mkdir(parents=True, exist_ok=True)
base = 'http://127.0.0.1:30007'


def request(route, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = response.read()
    return json.loads(raw) if raw and route == '/get_server_info' else raw.decode()


target = Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
from mlx_lm.utils import load_tokenizer
tokenizer = load_tokenizer(target)
texts = ['Explain why the sky is blue.',
         'Write a Python function that computes the first n Fibonacci numbers, with an explanation.',
         '请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
result = dict(label=args.label, tokens=args.tokens, cases=[], server_info=request('/get_server_info'),
              swap_before=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True))
out.write_text(json.dumps(result, indent=2))
for text_id, text in enumerate(texts):
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)
    request('/flush_cache')
    payload = dict(rid='validation-' + uuid.uuid4().hex, input_ids=prompt, stream=True,
                   sampling_params=dict(temperature=0, max_new_tokens=args.tokens, ignore_eos=True))
    req = urllib.request.Request(base + '/generate', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            raw = line[5:].strip()
            if raw == b'[DONE]':
                break
            event = json.loads(raw)
            count = event['meta_info']['completion_tokens']
            if count and (not arrivals or count != arrivals[-1][1]):
                arrivals.append([time.perf_counter() - start, count])
    elapsed = time.perf_counter() - start
    assert event['meta_info']['completion_tokens'] == args.tokens, event
    row = dict(text_id=text_id, input_tokens=len(prompt), elapsed_seconds=elapsed,
               ttft_seconds=arrivals[0][0], arrivals=arrivals, event=event,
               decode_tok_s=(arrivals[-1][1] - arrivals[0][1]) / (arrivals[-1][0] - arrivals[0][0]),
               token_hash=hashlib.sha256(json.dumps(event['output_ids']).encode()).hexdigest())
    result['cases'].append(row)
    out.write_text(json.dumps(result, indent=2))
    print('HTTP300', args.label, text_id, 'decode_tok_s', round(row['decode_tok_s'], 3),
          'ttft', round(row['ttft_seconds'], 3), 'e2e_tok_s', round(args.tokens / elapsed, 3), flush=True)
result['swap_after'] = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)
result['complete'] = True
out.write_text(json.dumps(result, indent=2))
