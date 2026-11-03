"""Summarize sealed JSON without mixing macro, model and HTTP metrics."""
import json,statistics
from pathlib import Path
O=Path(__file__).parent
files=['fullmodel_ab','fullmodel_mechanism','fullmodel_half_v2','observation_ab']
report={}
for name in files:
 p=O/'results'/(name+'.json')
 d=json.loads(p.read_text());rows=[]
 for mode in dict.fromkeys(r['mode'] for r in d['samples']):
  for arm in ('ar','dflash3'):
   samples=[r for r in d['samples'] if r['mode']==mode and r['arm']==arm]
   rows.append(dict(mode=mode,arm=arm,count=len(samples),mean_tok_s=statistics.mean(r['tok_s'] for r in samples),range_tok_s=[min(r['tok_s'] for r in samples),max(r['tok_s'] for r in samples)],all_exact=all(r['exact'] for r in samples)))
 report[name]=dict(complete=d.get('complete'),checkpoint=d.get('checkpoint'),summaries=rows)
(O/'results'/'model_summary.json').write_text(json.dumps(report,indent=2))
for name,d in report.items():print(name,[(r['mode'],r['arm'],round(r['mean_tok_s'],3)) for r in d['summaries']])
