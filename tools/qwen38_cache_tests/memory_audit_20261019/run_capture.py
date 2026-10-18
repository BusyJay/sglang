"""Capture MEM snapshots around short, long and repeated stage transitions."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://192.0.2.1:8001')
    parser.add_argument('--output', required=True)
    parser.add_argument('--remote', default='/prod/homelab-llm/models/startup-memory/full-audit-20261019')
    args = parser.parse_args()
    local = Path(args.output)
    local.mkdir(parents=True, exist_ok=True)
    remote = args.remote
    records = []

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.url + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=240) as response:
            text = response.read().decode()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    for attempt in range(90):
        try:
            request('/health')
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('Server did not become healthy')
    (local/'server.json').write_text(json.dumps(request('/get_server_info'), indent=2))
    prompt = [1000 + i % 16 for i in range(170000)]
    for name, size in [('short-first', 0), ('long-74000', 74000),
                       ('long-170000', 170000), ('short-after-flush', 0),
                       ('repeat-74000', 74000), ('repeat-170000', 170000),
                       ('short-after-second-flush', 0), ('short-repeat', 0)]:
        if name.startswith('short'):
            request('/flush_cache', {})
        start = request('/start_profile', {'activities': ['MEM'],
                       'output_dir': remote + '/' + name})
        payload = {'sampling_params': {'temperature': 0, 'max_new_tokens': 32,
                                      'ignore_eos': True}}
        if size:
            payload['input_ids'] = prompt[:size]
        else:
            payload['text'] = 'Explain briefly why the sky is blue.'
        then = time.monotonic()
        result = request('/generate', payload)
        stop = request('/stop_profile', {})
        records.append(dict(name=name, elapsed=time.monotonic()-then,
                            start=start, stop=stop, response=result))
        (local/'requests.json').write_text(json.dumps(records, indent=2)+'\n')
        print(name, result.get('meta_info', {}).get('prompt_tokens'), flush=True)
    request('/flush_cache', {})
    request('/start_profile', {'activities': ['MEM'], 'output_dir': remote+'/idle-final'})
    request('/stop_profile', {})


if __name__ == '__main__':
    main()
