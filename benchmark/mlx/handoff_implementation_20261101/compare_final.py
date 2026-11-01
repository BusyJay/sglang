#!/usr/bin/env python3
"""Sequential isolated-process final ABBA; one checkpoint resident at a time."""
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/Users/jay/workspace/sglang'));p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--groups',type=int,default=5);p.add_argument('--output',type=Path,required=True);p.add_argument('--harness',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
script=a.harness or a.root/'benchmark/mlx/handoff_implementation_20261101/isolated_model.py'
report=dict(scope='full model sequential independent processes, fixed AR/MTP2, cold5 steady91, output96',groups=a.groups,order=[],source={},samples=[])
report['harness_sha256']=hashlib.sha256(script.read_bytes()).hexdigest()
for name,armroot in [('baseline',a.baseline),('candidate',a.candidate)]:
 report['source'][name]={str(f.relative_to(armroot)):hashlib.sha256(f.read_bytes()).hexdigest() for group in ('python/sglang/kernels/ops','python/sglang/srt/hardware_backend/mlx') for f in (armroot/group).rglob('*.py')}
(a.output/'manifest.json').write_text(json.dumps(report['source'],indent=2))
for group in range(a.groups):
 order=('baseline','candidate','candidate','baseline') if group%2==0 else ('candidate','baseline','baseline','candidate')
 for pos,arm in enumerate(order):
  root=a.baseline if arm=='baseline' else a.candidate
  path=a.output/f'g{group}_p{pos}_{arm}.json';log=a.output/f'g{group}_p{pos}_{arm}.log'
  record=dict(group=group,position=pos,arm=arm,output=str(path),started=time.time());report['order'].append(record)
  with log.open('w') as handle:
   run=subprocess.run([sys.executable,str(script),'--arm',arm,'--group',str(group),'--output',str(path)],cwd=a.root,env=dict(os.environ,PYTHONPATH=str(root/'python'),MLX_ENABLE_TF32='0'),stdout=handle,stderr=subprocess.STDOUT)
  record.update(exit_code=run.returncode,finished=time.time())
  if run.returncode:raise RuntimeError(f'failed arm {arm} see {log}')
  data=json.loads(path.read_text())
  for rel,digest in report['source'][arm].items():
   if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:
    raise RuntimeError(f'source changed during benchmark: {arm} {rel}')
  if not Path(data['source']).resolve().is_relative_to((root/'python').resolve()):
   raise RuntimeError(f'wrong model import for {arm}: {data["source"]}')
  report['samples'].append(data)
  (a.output/'raw.json').write_text(json.dumps(report,indent=2));print('FINISHED',group,pos,arm,flush=True)
semantic_stat_keys=('verify_ct','num_proposed_drafts','num_reuse_tokens','step_histogram',
 'proposed_by_budget_depth','accepted_by_budget_depth','cycles_by_budget',
 'committed_tokens_by_budget','proposed_by_depth','eligible_by_depth','correct_by_depth',
 'eligible_by_budget_depth','correct_by_budget_depth')
summary=[]
for prompt in (32,2050):
 for steps in (0,2):
  values={arm:[s for data in report['samples'] if data['arm']==arm for s in data['samples'] if s['prompt']==prompt and s['steps']==steps] for arm in ('baseline','candidate')}
  rate={arm:statistics.median(s['tok_s'] for s in rows) for arm,rows in values.items()}
  hashes={arm:sorted({s['hash'] for s in rows}) for arm,rows in values.items()}
  signatures={arm:sorted({json.dumps({k:s['stats'].get(k) for k in semantic_stat_keys},sort_keys=True)
             for s in rows}) for arm,rows in values.items()}
  details={}
  for arm,rows in values.items():
   details[arm]=dict(samples=len(rows),steady_tok_s=[s['tok_s'] for s in rows],
    all_tok_s=[s['all_tok_s'] for s in rows],cold_seconds=[s['cold'] for s in rows],
    prefill_seconds=[s['prefill_seconds'] for s in rows],
    active_bytes=[s['active'] for s in rows],peak_bytes=[s['peak'] for s in rows],
    semantics=[json.loads(v) for v in signatures[arm]])
  group_changes=[]
  for group in range(a.groups):
   grouped={arm:[s['tok_s'] for data in report['samples'] if data['arm']==arm and data['group']==group
                 for s in data['samples'] if s['prompt']==prompt and s['steps']==steps]
            for arm in ('baseline','candidate')}
   if all(grouped.values()):
    group_changes.append(100*(statistics.median(grouped['candidate'])/statistics.median(grouped['baseline'])-1))
  summary.append(dict(prompt=prompt,steps=steps,tok_s=rate,
   change_percent=100*(rate['candidate']/rate['baseline']-1),group_change_percent=group_changes,
   hashes=hashes,identical_tokens=hashes['baseline']==hashes['candidate'],
   identical_acceptance=signatures['baseline']==signatures['candidate'],details=details))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
