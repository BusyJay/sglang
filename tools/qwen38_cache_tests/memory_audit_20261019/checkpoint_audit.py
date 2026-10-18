"""Reconcile loaded GPU parameter shapes with local safetensors headers (no load)."""

import argparse
import gzip
import json
import math
import re
import struct
from pathlib import Path

DTYPE_BYTES = {'BF16': 2, 'F32': 4, 'F16': 2, 'I64': 8, 'I32': 4,
               'F8_E4M3': 1, 'U8': 1, 'U32': 4}
TORCH_BYTES = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float16': 2,
               'torch.int64': 8, 'torch.int32': 4, 'torch.uint8': 1,
               'torch.float8_e4m3fn': 1}


def read_headers(checkpoint):
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text())
    headers = {}
    for filename in sorted(set(index['weight_map'].values())):
        with (checkpoint/filename).open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name != '__metadata__':
                entry['file'] = filename
                headers[name] = entry
                assert name in index['weight_map']
                assert index['weight_map'][name] == filename
                assert entry['data_offsets'][1]-entry['data_offsets'][0] == (
                    math.prod(entry['shape'])*DTYPE_BYTES[entry['dtype']]
                ), name
    assert headers.keys() == index['weight_map'].keys()
    return headers


def sources(name, kind, headers):
    if kind == 'target':
        name = name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else name
    else:
        name = name.replace('model.', 'mtp.', 1) if name.startswith('model.') else ('mtp.'+name if name != 'lm_head.weight' else name)
        name = name.replace('.gemma_weight', '.weight')
    name = name.replace('.weight_scale_inv', '.weight_scale')
    if '.layers.' in name and '.linear_attn.' not in name:
        name = re.sub(r'(\.layers\.\d+\.)(qkv_proj|q_proj|k_proj|v_proj|o_proj|indexer|q_norm|k_norm)', r'\1self_attn.\2', name)
    name = name.replace('.linear_attn.attn.', '.linear_attn.')
    if name in headers:
        return [name]
    merges = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
              'gate_up_proj': ['gate_proj', 'up_proj'],
              'in_proj_qkvz': ['in_proj_qkv', 'in_proj_z'],
              'in_proj_ba': ['in_proj_b', 'in_proj_a']}
    for dst, src in merges.items():
        if '.'+dst+'.' in name:
            candidates = [name.replace('.'+dst+'.', '.'+part+'.') for part in src]
            if all(key in headers for key in candidates):
                return candidates
    if kind == 'draft' and '.experts.w' in name:
        prefix, suffix = name.split('.experts.w')
        group, suffix = suffix.split('_', 1)
        projections = ['gate_proj', 'up_proj'] if group == '13' else ['down_proj']
        candidates = [f'{prefix}.experts.{i}.{p}.{suffix}'
                      for i in range(512) for p in projections]
        if all(key in headers for key in candidates):
            return candidates
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--startup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    headers = read_headers(args.checkpoint)
    report = dict(checkpoint=str(args.checkpoint), header_count=len(headers),
                  checkpoint_bytes=sum(e['data_offsets'][1]-e['data_offsets'][0] for e in headers.values()),
                  parameters=[], derived=[], unmatched=[], casts=[], buffers=[])
    unique, used_headers = {}, set()
    for kind, pattern in [('target', '0878*.inventory.json'), ('draft', '0903*.inventory.json')]:
        inv = json.loads(next(args.startup.glob(pattern)).read_text())
        for row in inv['storages']:
            if not row['device'].startswith('cuda'):
                continue
            key = (row['device'], row['pointer'], row['bytes'])
            unique[key] = row['bytes']
            for alias in row['aliases']:
                name = alias['name']
                entry = dict(model=kind, **alias, pointer=row['pointer'], storage_bytes=row['bytes'])
                if alias['kind'] == 'buffer':
                    report['buffers'].append(entry)
                    continue
                if name.endswith(('_swizzled', '_padded', '.g2_alphas')):
                    report['derived'].append(entry)
                    continue
                src = sources(name, kind, headers)
                if not src:
                    report['unmatched'].append(entry)
                    continue
                expected = sum(headers[n]['data_offsets'][1]-headers[n]['data_offsets'][0] for n in src)
                logical = math.prod(alias['shape'])*TORCH_BYTES[alias['dtype']]
                entry.update(sources=src, source_bytes=expected, logical_bytes=logical)
                if logical != expected:
                    # A_log converts checkpoint BF16 to FP32; no arbitrary tolerance.
                    assert name.endswith('.A_log') and logical == expected*2, entry
                    report['casts'].append(entry)
                else:
                    # Merged tensors must preserve all non-row dimensions.
                    if len(src) == 1:
                        assert alias['shape'] == headers[src[0]]['shape'], entry
                    elif '.experts.w' not in name:
                        assert alias['shape'][0] == sum(headers[n]['shape'][0] for n in src), entry
                        assert all(alias['shape'][1:] == headers[n]['shape'][1:] for n in src), entry
                    report['parameters'].append(entry)
                used_headers.update(src)
    report['unique_gpu_named_storage_bytes'] = sum(unique.values())
    report['covered_checkpoint_tensor_count'] = len(used_headers)
    report['summary'] = {k: len(report[k]) for k in ('parameters', 'casts', 'derived', 'buffers', 'unmatched')}
    args.output.mkdir(parents=True, exist_ok=True)
    # Only relevant headers; full tensor bytes remain in the existing checkpoint.
    relevant = {n: headers[n] for n in used_headers}
    with gzip.open(args.output/'checkpoint-headers.json.gz', 'wt') as stream:
        json.dump(relevant, stream)
    with gzip.open(args.output/'checkpoint-reconciliation.json.gz', 'wt') as stream:
        json.dump(report, stream)
    print(json.dumps(report['summary']))
    print('unmatched:', report['unmatched'])
    print('unique GPU registered storage MiB:', sum(unique.values())/2**20)


if __name__ == '__main__':
    main()
