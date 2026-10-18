"""Extract native CUDA allocation evidence from the local Nsight SQLite export."""

import argparse
import collections
import gzip
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--idle-ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    con = sqlite3.connect(f'file:{args.sqlite}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    events = [dict(r) for r in con.execute('SELECT * FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start')]
    apis = [dict(r) for r in con.execute('''
        SELECT r.*, s.value AS api FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id=r.nameId
        WHERE (s.value LIKE '%Mem%' AND s.value NOT LIKE '%Memcpy%' AND s.value NOT LIKE 'cudaMemset%')
           OR s.value LIKE '%Malloc%' OR s.value LIKE '%Alloc%' OR s.value LIKE '%Free%'
           OR s.value LIKE '%Module%' OR s.value LIKE '%Context%'
        ORDER BY r.start
    ''')]
    lookup = {(r['globalTid'] & -16777216, r['correlationId']): r['api'] for r in apis}
    for row in events:
        row['api'] = lookup.get((row['globalPid'], row['correlationId']))
    totals, counts, returns, live = collections.Counter(), collections.Counter(), collections.Counter(), {}
    for row in apis:
        returns[(row['api'], row['returnValue'])] += 1
    for row in events:
        if row['memKind'] != 2:
            continue
        key = (row['api'], row['memoryOperationType'])
        totals[key] += row['bytes']
        counts[key] += 1
        address = (row['globalPid'], row['address'])
        if row['memoryOperationType'] == 0:
            assert address not in live, address
            live[address] = row
        else:
            previous = live.pop(address)
            assert previous['bytes'] == row['bytes'], row
    with gzip.open(args.idle_ledger, 'rt') as stream:
        idle = json.load(stream)
    mapped = sum(r['bytes'] for r in live.values() if r['api'] == 'cuMemSetAccess')
    physical = idle['physical_reserved_union']+idle['vmm_backed']
    # Successful Create/Release and Map/Unmap counts close too. Failed API
    # attempts do not become live physical handles.
    success = lambda api: returns[(api, 0)]
    outstanding_handles = success('cuMemCreate')-success('cuMemRelease')
    outstanding_maps = success('cuMemMap')-success('cuMemUnmap')
    assert outstanding_handles == outstanding_maps
    assert mapped == physical, (mapped, physical)
    result = dict(sqlite=str(args.sqlite),
        mapped_live_bytes=mapped, allocator_plus_vmm_bytes=physical,
        unexplained_mapped_bytes=mapped-physical,
        outstanding_handles=outstanding_handles, outstanding_maps=outstanding_maps,
        cuda_malloc_live=[r for r in live.values() if r['api'] != 'cuMemSetAccess'],
        allocation_totals=[dict(api=a, operation=o, count=counts[(a,o)], bytes=n)
                           for (a,o),n in totals.items()],
        api_results=[dict(api=a, result=r, count=n) for (a,r),n in sorted(returns.items())])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in [('native-memory-events', events), ('native-memory-apis', apis)]:
        with gzip.open(args.output/(name+'.json.gz'), 'wt') as stream:
            json.dump(obj, stream)
    (args.output/'native-reconciliation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('api_results','cuda_malloc_live')},indent=2))


if __name__ == '__main__':
    main()
