from pathlib import Path
import json,hashlib,subprocess
root=Path('benchmark/mlx/fp4_optimization_20261103');r=json.loads((root/'final_ab.json').read_text());rows=[]
for j,name in enumerate(('text','code','chinese')):
 groups={arm:[v for v in r['samples'] if v['text_id']==j and v['arm']==arm] for arm in ('baseline','optimized')}
 speeds={arm:sum(len(v['raw_s']) for v in values)/sum(sum(v['raw_s']) for v in values) for arm,values in groups.items()}
 rows.append({'task':name,**speeds,'improvement_pct':(speeds['optimized']/speeds['baseline']-1)*100,'tokens_equal':len({tuple(v['ids']) for v in r['samples'] if v['text_id']==j})==1})
rsummary={'rows':rows,'peak_gib':max(v['peak'] for v in r['samples'])/2**30,'swap_before':r['swap_before'],'swap_after':r['swap_after']};(root/'summary.json').write_text(json.dumps(rsummary,indent=2));print(json.dumps(rsummary,indent=2))
