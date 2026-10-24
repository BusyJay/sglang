import bisect, collections, gzip, json, statistics, sys
from pathlib import Path
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent

def union(events):
 out=[]
 for a,b in sorted((e['ts'],e['ts']+e['dur']) for e in events):
  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
  else:out.append([a,b])
 return out

def intersect(a,b):
 i=j=0;t=0
 while i<len(a) and j<len(b):
  t+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return t

def kind(n):
 for k,needle in [('planner','decode_plan_kernel'),('fill','mailbox_fill_kernel'),('routed','MoEMicroKernelW4A16SmallMDirect'),('mxfp8','DeviceGemmMxfp8Gemm'),('gemv','gemvx::kernel'),('bf16_gemm','cutlass_80_wmma_tensorop_bf16'),('hc_mix','_hc_mix_persistent_kernel'),('qsa','_sparse_gqa'),('gdn','delta_rule'),('alpha','alpha')]:
  if needle in n:return k
 return 'other'
def stats(es,total,denom):
 d=sorted(e['dur'] for e in es)
 return dict(count=len(d),total_ms=sum(d)/1000,share_pct=100*sum(d)/total,us_per_layer=sum(d)/denom,median_us=statistics.median(d),p95_us=d[min(len(d)-1,int(len(d)*.95))],max_us=max(d))
report={}
for label in (sys.argv[2:] or ['short','natural','prefill8k']):
 root=R/label
 p=next(p for p in root.rglob('*.json.gz') if p.name!='window.json.gz')
 with gzip.open(p,'rt') as f:trace=json.load(f)
 events=trace['traceEvents'];gpu=[e for e in events if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and e.get('dur',0)>0]
 plans=sorted([e for e in gpu if kind(e['name'])=='planner'],key=lambda e:e['ts'])
 if plans:
  lo=plans[48]['ts'];hi=plans[(len(plans)//48-1)*48]['ts']
 else:lo=min(e['ts'] for e in gpu);hi=max(e['ts']+e['dur'] for e in gpu)
 gpu=[e for e in gpu if lo<=e['ts'] and e['ts']+e['dur']<=hi]
 groups=collections.defaultdict(list);names=collections.defaultdict(list)
 for e in gpu:groups[kind(e['name'])].append(e);names[e['name']].append(e)
 total=sum(e['dur'] for e in gpu);denom=len(groups['planner']) or 48
 summary={k:stats(es,total,denom) for k,es in groups.items() if es}
 busy=sum(b-a for a,b in union(gpu))
 # Intersection with any other GPU activity, excluding fill itself.
 fills=union(groups['fill']);others=union([e for e in gpu if kind(e['name'])!='fill'])
 filltime=sum(b-a for a,b in fills);ov=intersect(fills,others)
 byname=sorted([dict(name=n,**stats(es,total,denom)) for n,es in names.items()],key=lambda e:-e['total_ms'])
 cpu=collections.defaultdict(list)
 for e in events:
  if e.get('cat') in ['cuda_runtime','cuda_driver'] and lo<=e.get('ts',-1)<hi and e.get('dur',0)>0:cpu[e['name']].append(e)
 gaps=[];u=union(gpu)
 for a,b in zip(u,u[1:]):
  if b[0]>a[1]:gaps.append(b[0]-a[1])
 report[label]=dict(trace=str(p),span_ms=(hi-lo)/1000,gpu_union_ms=busy/1000,gpu_idle_pct=100*(hi-lo-busy)/(hi-lo),total_gpu_activity_ms=total/1000,decode_layers=len(groups['planner']),groups=summary,fill_union_ms=filltime/1000,fill_overlap_any_ms=ov/1000,fill_exclusive_ms=(filltime-ov)/1000,fill_overlap_pct=100*ov/filltime if filltime else 0,top_kernels=byname[:40],cpu_api=sorted([dict(name=n,count=len(es),total_ms=sum(e['dur'] for e in es)/1000,max_us=max(e['dur'] for e in es)) for n,es in cpu.items()],key=lambda e:-e['total_ms'])[:20],idle_gap_p95_us=sorted(gaps)[int(.95*len(gaps))] if gaps else 0,idle_gap_max_us=max(gaps) if gaps else 0)
 if plans:
  a=plans[min(48*24,(len(plans)//48//3)*48)]['ts'];b=plans[min(len(plans)//48-1,48)*48]['ts']
 else:a=lo;b=hi
 trace['traceEvents']=[e for e in events if e.get('ph')=='M' or a<=e.get('ts',-1)<b]
 with gzip.open(root/'window.json.gz','wt') as f:json.dump(trace,f)
 print(label,json.dumps({k:v for k,v in report[label].items() if k not in ['top_kernels','cpu_api']}),flush=True)
 (R/'summary.json').write_text(json.dumps(report,indent=2))
