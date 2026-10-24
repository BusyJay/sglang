"""Record exact response comparisons; full serving passes do not imply determinism."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent
first, second = [root / name for name in ('capacity-native-half-final', 'capacity-native-half-repeat')]
data = [json.loads((p / 'performance.json').read_text()) for p in (first, second)]
cases = [d['modes']['default']['cases'] for d in data]
rows = []
for a, b in zip(*cases):
    assert a['name'] == b['name']
    x, y = a['response_text'], b['response_text']
    offset = len(os.path.commonprefix([x, y]))
    rows.append(dict(name=a['name'], equal=x == y, shared_prefix_characters=offset,
                     first_excerpt=x[offset:offset + 100], second_excerpt=y[offset:offset + 100]))
environments = [json.loads((p / 'environment.json').read_text()) for p in (first, second)]
result = dict(
    statuses=[d['status'] for d in data],
    commands_equal=json.loads((first / 'command.json').read_text()) == json.loads((second / 'command.json').read_text()),
    source_hashes_equal=json.loads((first / 'source-manifest.json').read_text()) == json.loads((second / 'source-manifest.json').read_text()),
    environment_differences={k: [e.get(k) for e in environments] for k in set(environments[0]) | set(environments[1]) if environments[0].get(k) != environments[1].get(k)},
    cases=rows,
    note='Greedy generation with identical source and command. Short no-prefix prefills bypass the modified paged QSA path. This records run-to-run variation, not its cause and not a model quality evaluation.')
(root / 'native-identical-config-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
