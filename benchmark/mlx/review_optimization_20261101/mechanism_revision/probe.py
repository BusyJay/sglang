import json,time,statistics
from pathlib import Path
import mlx.core as mx
import candidates as C
OUT=Path(__file__).parent
mx.random.seed(20261101)
records=[]
def compare(label,fs,args,metadata):
 ref=fs['baseline'](*args);mx.eval(ref)
 for name,f in fs.items():
  out=f(*args);mx.eval(out);assert mx.array_equal(ref,out).item(),(label,name,mx.max(mx.abs(ref-out)).item())
 raw=[]
 for group in range(3):
  for name in fs:
   if name=='baseline':continue
   for arm in ('baseline',name,name,'baseline'):
    mx.synchronize();t=time.perf_counter()
    for _ in range(8):mx.eval(fs[arm](*args))
    raw.append(dict(group=group,arm=arm,ms=(time.perf_counter()-t)*1000/8))
 row=dict(label=label,**metadata,exact=True,raw=raw)
 records.append(row);(OUT/'probe.json').write_text(json.dumps(records,indent=2))
 print(label,metadata,{n:round(statistics.median(r['ms'] for r in raw if r['arm']==n),4) for n in fs},flush=True)
for bits in (4,5,6):
 e,n,k,count=32,4096,2048,3
 w,s,b=mx.quantize(mx.random.normal((e,n,k)).astype(mx.float16),group_size=64,bits=bits);mx.eval(w,s,b)
 act=mx.random.normal((1,count,8,k)).astype(mx.float16);scores=mx.random.uniform(shape=(1,count,8));shared=mx.random.normal((1,count,n)).astype(mx.bfloat16)
 for dist in ('spread','mixed','repeat'):
  ids=(mx.arange(24).reshape(1,3,8)%(32 if dist=='spread' else 8)).astype(mx.uint32) if dist!='mixed' else mx.array([[[0,1,2,3,4,5,6,7],[0,1,2,3,8,9,10,11],[0,1,4,5,12,13,14,15]]],mx.uint32)
  grouped=C.D._group_picks(ids,3);mx.eval(act,scores,shared,ids,grouped)
  fs={'baseline':mx.compile(lambda a,i,sc,w,s,b,sh:C.D.mlx_glm_down_combine(a,i,sc,w,s,b,sh,bits=bits,group_size=64))}
  for kind in ('byte','device','threadgroup')+ (('word',) if bits==4 else ()):
   for simds in (8,):
    fs[kind]=mx.compile(lambda a,i,sc,w,s,b,sh,kind=kind:C.down(a,i,sc,w,s,b,sh,bits=bits,kind=kind,grouped=grouped,simds=simds))
  compare('down',fs,(act,ids,scores,w,s,b,shared),dict(bits=bits,dist=dist))
 del w,s,b,fs,act,shared;mx.synchronize();mx.clear_cache()
e,n,k,count=32,2048,4096,3
ga=mx.quantize(mx.random.normal((e,n,k)).astype(mx.float16),group_size=64,bits=4);ua=mx.quantize(mx.random.normal((e,n,k)).astype(mx.float16),group_size=64,bits=4);x=mx.random.normal((1,count,k)).astype(mx.float16);mx.eval(ga,ua,x)
for dist in ('spread','mixed','repeat'):
 ids=(mx.arange(24).reshape(1,3,8)%(32 if dist=='spread' else 8)).astype(mx.uint32) if dist!='mixed' else mx.array([[[0,1,2,3,4,5,6,7],[0,1,2,3,8,9,10,11],[0,1,4,5,12,13,14,15]]],mx.uint32)
 grouped=C.D._group_picks(ids,3);mx.eval(ids,grouped)
 fs={'baseline':mx.compile(lambda x,g,u,i:C.A.mlx_affine_routed_swiglu(x,g,u,i,bits=4,group_size=64,limit=7.,grouped=grouped))}
 for kind in ('device','word'):
  for rps in (2,4):fs[kind+str(rps)]=mx.compile(lambda x,g,u,i,kind=kind,rps=rps:C.gate(x,g,u,i,kind=kind,grouped=grouped,rps=rps))
 compare('gate',fs,(x,ga,ua,ids),dict(bits=4,dist=dist))
del ga,ua,x,fs;mx.synchronize();mx.clear_cache()
n,k=154880,4096
w=mx.random.normal((n,k)).astype(mx.bfloat16);mx.eval(w)
for count in (1,3):
 x=mx.random.normal((1,count,k)).astype(mx.bfloat16);mx.eval(x)
 fs={'baseline':mx.compile(lambda x,w:C.H.mlx_dense_argmax_rows(x,w))}
 for bm,rps,vector in ((4,4,False),(4,8,False),(8,8,False),(8,4,True),(4,8,True)):
  fs[f'{bm}_{rps}_{vector}']=mx.compile(lambda x,w,bm=bm,rps=rps,vector=vector:C.head(x,w,bm=bm,rps=rps,vector=vector))
 compare('head',fs,(x,w),dict(count=count,n=n,k=k))
