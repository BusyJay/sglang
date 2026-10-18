"""Check complete request histories against start/end requested storage bytes."""

import argparse
import collections
import json
from pathlib import Path
import pickle


def identity(event):
    """Event fields that must be stable across two dumps of the same history.

    PyTorch re-derives ``time_us`` at dump time, so the same startup entry can
    come back one microsecond apart in two snapshots. Everything else, order
    included, has to match for the incremental history to be trustworthy.
    """
    return {key: value for key, value in event.items() if key != 'time_us'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Filenames carry the capture clock: start snapshots are
    # "start-<pid>-<time_ns>.pickle" and stop snapshots are
    # "<epoch>.<fraction>-TP-<n>-memory.pickle". mtime ties within one second,
    # so pair stages with their start snapshot by those timestamps instead.
    def start_stamp(path):
        return int(path.name.removesuffix('.pickle').rsplit('-', 1)[1]) / 1e9

    starts = sorted((start_stamp(path), path) for path in args.captures.glob('start-*.pickle'))
    result = {}
    claimed = set()
    for directory in sorted(p for p in args.captures.iterdir() if p.is_dir() and p.name != 'startup'):
        end = max(directory.glob('*.pickle'), key=lambda p: float(p.name.split('-')[0]))
        end_epoch = float(end.name.split('-')[0])
        candidates = [path for stamp, path in starts if stamp <= end_epoch]
        assert candidates, f'No start snapshot before {directory.name}'
        start = candidates[-1]
        assert start not in claimed, f'{directory.name} reuses {start.name}'
        claimed.add(start)
        before, after = pickle.load(start.open('rb')), pickle.load(end.open('rb'))
        prior = before['device_traces'][0]
        trace = after['device_traces'][0]
        assert len(trace) < 2000000, 'History ring was truncated'
        assert [identity(e) for e in trace[:len(prior)]] == [identity(e) for e in prior], \
            'Start history prefix was lost'
        trace = trace[len(prior):]
        counts, sizes = collections.Counter(), collections.Counter()
        for event in trace:
            counts[event['action']] += 1
            sizes[event['action']] += event.get('size', 0)
        def requested(snapshot):
            return sum(b['requested_size'] for s in snapshot['segments']
                       for b in s['blocks'] if b['state'] != 'inactive')
        measured = requested(after)-requested(before)
        traced = sizes['alloc']-sizes['free_completed']
        assert measured == traced, (directory.name, measured, traced)
        result[directory.name] = dict(start=str(start), end=str(end),
            events=len(trace), initial_history_events=len(prior),
            actions=dict(counts), action_bytes=dict(sizes),
            requested_storage_delta=measured, alloc_minus_completed_free=traced,
            unexplained_bytes=measured-traced)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name:r['unexplained_bytes'] for name,r in result.items()}))


if __name__ == '__main__':
    main()
