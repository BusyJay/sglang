"""Reconcile allocator segments with Python owners and real CUDA VMM backing.

Uses snapshot bytes, never adds virtual tensor capacities to physical usage.
Raw capture files are trusted local diagnostic output, not uploaded pickles.
"""

import argparse
import collections
import gzip
import json
import pickle
import re
from pathlib import Path

MIB = 2**20


def owner_name(name):
    name = re.sub(r'stack\.(self|scheduler)\.Scheduler\[\d+\]\.', '', name)
    name = name.replace('._modules.', '.').replace('._parameters.', '.').replace('._buffers.', '.')
    return name


def stack_name(block):
    frames = block.get('frames', [])
    for frame in frames:
        filename = frame['filename']
        if '/python/sglang/' in filename:
            return filename.split('/python/sglang/', 1)[1] + ':' + str(frame['line']) + ':' + frame['name']
    if frames:
        frame = frames[0]
        return f"{frame['filename']}:{frame['line']}:{frame['name']}"
    return 'history_not_enabled_at_allocation'


def union_bytes(ranges):
    total, end = 0, -1
    for start, stop in sorted(ranges):
        if stop > end:
            total += stop-max(end, start)
            end = stop
    return total


def summarize(path, output):
    snapshot = pickle.load(path.open('rb'))
    owners = json.loads(Path(str(path)+'.owners.json').read_text())
    device = 0
    segments = [s for s in snapshot['segments'] if s['device'] == device]
    arenas = {tuple(a['pool_id']): a for a in owners['arenas'] if not a['closed']}
    storages = [s for s in owners['storages'] if s['device'] == 'cuda:0']
    ordinary = [s for s in segments if tuple(s['segment_pool_id']) == (0, 0)]
    physical = [s for s in segments if tuple(s['segment_pool_id']) not in arenas]
    # CUDA Graph borrow pools can alias another pool's addresses. Count the
    # union of physical extents, even if the allocator reports them twice.
    reserved_union = union_bytes([(s['address'], s['address']+s['total_size']) for s in physical])
    rows, by_stack, states, pools = [], collections.Counter(), collections.Counter(), {}
    for seg in physical:
        pool = str(tuple(seg['segment_pool_id']))
        pools.setdefault(pool, dict(reserved=0, allocated=0, inactive=0, pending=0))
        pools[pool]['reserved'] += seg['total_size']
        for block in seg['blocks']:
            state = block['state']
            states[state] += block['size']
            pools[pool][{'inactive': 'inactive', 'active_allocated': 'allocated',
                         'active_awaiting_free': 'pending', 'active_pending_free': 'pending'}[state]] += block['size']
            if state == 'inactive':
                continue
            aliases = []
            for storage in storages:
                if (block['address'] <= storage['pointer'] and
                    storage['pointer']+storage['bytes'] <= block['address']+block['size']):
                    aliases += [owner_name(a['name']) for a in storage['aliases']]
            aliases = sorted(set(aliases), key=lambda n: (n.startswith('gc_'), len(n), n))
            stack = stack_name(block)
            by_stack[stack] += block['size']
            rows.append(dict(address=block['address'], size=block['size'],
                             requested_size=block.get('requested_size'), state=state,
                             pool=pool, owners=aliases, stack=stack))
    traces = snapshot.get('device_traces', [[]])[device]
    trace_actions = collections.Counter()
    trace_bytes = collections.Counter()
    for event in traces:
        trace_actions[event['action']] += 1
        trace_bytes[event['action']] += event.get('size', 0)
    vmm_backed = sum(a['backed'] for a in arenas.values())
    residual = owners['driver_total']-owners['driver_free']-reserved_union-vmm_backed
    result = dict(source=str(path), driver_free=owners['driver_free'],
                  driver_total=owners['driver_total'], physical_reserved_union=reserved_union,
                  physical_segment_sum=sum(s['total_size'] for s in physical),
                  ordinary_reserved=sum(s['total_size'] for s in ordinary),
                  vmm_backed=vmm_backed, outside_allocator_and_vmm=residual,
                  physical_states=dict(states), pools=pools,
                  vmm_arenas=list(arenas.values()),
                  unknown_owner_bytes=sum(r['size'] for r in rows if not r['owners']),
                  unknown_owner_count=sum(not r['owners'] for r in rows),
                  stack_totals=dict(by_stack.most_common()),
                  trace_actions=dict(trace_actions), trace_action_bytes=dict(trace_bytes),
                  blocks=rows, skipped_tensor_types=owners.get('skipped', []),
                  nvidia_smi=owners['nvidia_smi'])
    output.mkdir(parents=True, exist_ok=True)
    # Preserve complete segment stacks and ownership evidence in reviewable,
    # compressed files; raw binary histories remain at source paths.
    for name, data in [('segments', snapshot['segments']), ('owners', owners), ('ledger', result)]:
        with gzip.open(output/(name+'.json.gz'), 'wt') as stream:
            json.dump(data, stream)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for directory in sorted(p for p in args.captures.iterdir() if p.is_dir() and p.name != 'startup'):
        files = [p for p in directory.glob('*.pickle') if Path(str(p)+'.owners.json').exists()]
        if not files:
            continue
        path = max(files, key=lambda p: p.stat().st_mtime)
        row = summarize(path, args.output/directory.name)
        result[directory.name] = {k:v for k,v in row.items() if k not in ('blocks', 'stack_totals', 'skipped_tensor_types')}
        print(directory.name, {k:round(row[k]/MIB, 4) for k in (
            'driver_free', 'physical_reserved_union', 'vmm_backed',
            'outside_allocator_and_vmm', 'unknown_owner_bytes')})
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
