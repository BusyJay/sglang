# Bounded targeted probe: preserve full projection parallelism, head1/3 and gate4bit.
import json,time,statistics
from pathlib import Path
import mlx.core as mx
import candidates as C
OUT=Path(__file__).parent
records=[]
def measure(label,fs,args):
 ref=fs['baseline'](*args);mx.eval(ref)
 for name,f in fs.items():assert mx.array_equal(ref,f(*args)).item(),(label,name)
 raw=[]
 for group in range(3):
  for name in fs:
   if name=='baseline':continue
   for arm in ('baseline',name,name,'baseline'):
    t=time.perf_counter()
    for _ in range(6):mx.eval(fs[arm](*args))
    raw.append(dict(group=group,arm=arm,ms=(time.perf_counter()-t)*1000/6))
 records.append(dict(label=label,exact=True,raw=raw));(OUT/'refine.json').write_text(json.dumps(records,indent=2));print(label,{n:round(statistics.median(r['ms'] for r in raw if r['arm']==n),4) for n in fs},flush=True)
mx.random.seed(20261101)
for bits in (4,5,6):
 w,s,b=mx.quantize(mx.random.normal((32,4096,2048)).astype(mx.float16),group_size=64,bits=bits);mx.eval(w,s,b)
 a=mx.random.normal((1,3,8,2048)).astype(mx.float16);sc=mx.random.uniform(shape=(1,3,8));sh=mx.random.normal((1,3,4096)).astype(mx.bfloat16)
 i=mx.array([[[0,1,2,3,4,5,6,7],[0,1,2,3,8,9,10,11],[0,1,4,5,12,13,14,15]]],mx.uint32)
 fs={'baseline':mx.compile(lambda *a:C.D.mlx_glm_down_combine(*a,bits=bits,group_size=64))}
 for kind,tile in [('tg',2),('tg',3)]+([('word',1),('word',2)] if bits==4 else []):fs[f'{kind}{tile}']=mx.compile(lambda *a,kind=kind,tile=tile:C.parallel_down(*a,bits=bits,kind=kind,tile=tile))
 measure('parallel_down'+str(bits),fs,(a,i,sc,w,s,b,sh));del w,s,b,fs;mx.synchronize();mx.clear_cache()
w=mx.random.normal((154880,4096)).astype(mx.bfloat16);mx.eval(w)
for count in (1,3):
 x=mx.random.normal((1,count,4096)).astype(mx.bfloat16)
 fs={'baseline':mx.compile(lambda x,w:C.H.mlx_dense_argmax_rows(x,w))}
 for bm in (4,8):fs[f'vector{bm}']=mx.compile(lambda x,w,bm=bm:C.head(x,w,bm=bm,rps=4,vector=True))
 measure('head'+str(count),fs,(x,w))
del w,fs;mx.synchronize();mx.clear_cache()
ga=mx.quantize(mx.random.normal((32,2048,4096)).astype(mx.float16),group_size=64,bits=4);ua=mx.quantize(mx.random.normal((32,2048,4096)).astype(mx.float16),group_size=64,bits=4);x=mx.random.normal((1,3,4096)).astype(mx.float16);grouped=C.D._group_picks(i,3);mx.eval(ga,ua,x,grouped)
fs={'baseline':mx.compile(lambda x,g,u,i:C.A.mlx_affine_routed_swiglu(x,g,u,i,bits=4,group_size=64,limit=10.,grouped=grouped))}
for kind in ('word','masks','coeff'):fs[kind]=mx.compile(lambda x,g,u,i,kind=kind:C.gate(x,g,u,i,kind=kind,grouped=grouped,limit=10.))
measure('gate4bit',fs,(x,ga,ua,i))
