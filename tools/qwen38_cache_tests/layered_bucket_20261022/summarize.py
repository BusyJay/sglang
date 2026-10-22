import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
out={}
for directory in root.iterdir():
 path=directory/'performance.json'
 if not path.is_file():continue
 d=json.loads(path.read_text()); cases=d['modes'].get('default',{}).get('cases',[]); result={'status':d['status'],'cases':{}}
 for c in cases:
  events=c['stream_events']; gaps=[(b[0]-a[0])*1000/(b[1]-a[1]) for a,b in zip(events,events[1:]) if b[1]>a[1] and a[1]>=64]
  sweeps=[(b[0]-a[0])*1000 for a,b in zip(events,events[1:]) if b[1]>a[1] and a[1]>=64]
  result['cases'][c['name']]=dict(decode_tok_s=c['decode_tokens_per_second'],after64_tok_s=c['after_64_tokens_per_second'],ttft_s=c['ttft_seconds'],e2e_s=c['e2e_seconds'],accept_length=c['meta_info'].get('spec_accept_length'),accept_rate=c['meta_info'].get('spec_accept_rate'),mean_sweep_ms=statistics.mean(sweeps) if sweeps else None,p95_sweep_ms=sorted(sweeps)[int((len(sweeps)-1)*.95)] if sweeps else None,p95_normalized_tpot_ms=sorted(gaps)[int((len(gaps)-1)*.95)] if gaps else None)
 short=[c for n,c in result['cases'].items() if n.startswith('short_')]
 if short: result['short_median_tok_s']=statistics.median(c['decode_tok_s'] for c in short)
 out[directory.name]=result
(root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for n,r in sorted(out.items()): print(n,r['status'],round(r.get('short_median_tok_s',0),2),[(k,round(c['decode_tok_s'],2),round(c['ttft_s'],3)) for k,c in r['cases'].items() if k in ('natural_chat','long_170k','post_resize_short')])
