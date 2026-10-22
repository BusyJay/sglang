"""Real NVFP4 weights, equal row pool, BF16 vs Strata Q8 row arithmetic.
Input conversions and cold-cache eviction are outside timing. Not a whole engine comparison.
"""
import ctypes as C,json,os,random,statistics,time
from pathlib import Path
import numpy as np
import torch
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import Qwen38ExpertPack,load_manifest
os.environ.setdefault('SGLANG_QWEN38_SOURCE_CACHE_DIR','/dev/shm/sglang-qwen38-startup')
R=Path('/results');rng=np.random.default_rng(42);order=random.Random(42);torch.set_num_threads(1)
m=load_manifest(Path('/root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json'));pack=Qwen38ExpertPack(m,source_mode='shared_thp');mapping=pack.cuda_source_mapping()
lib=C.CDLL('/results/row_bridge.so');lib.row_pool_create.argtypes=[C.c_int,C.POINTER(C.c_int)];lib.row_pool_create.restype=C.c_void_p;lib.row_pool_destroy.argtypes=[C.c_void_p]
lib.row_run.argtypes=[C.c_void_p,C.c_int,*([C.c_void_p]*6),C.c_int,C.c_int,C.c_int];lib.flush_range.argtypes=[C.c_void_p,C.c_size_t]
fixtures=[]
for layer,expert in [(0,7),(22,111)]:
 for kind,n,rows in [('w13',m.hidden_size,2*m.intermediate_size),('w2',m.intermediate_size,m.hidden_size)]:
  def read(name):
   rec=m.tensors[layer][name];stride=rec.nbytes//m.num_experts
   return np.frombuffer(mapping,dtype=np.uint8,count=stride,offset=rec.offset+expert*stride).copy()
  w=read(kind+'_weight').reshape(rows,n//2);s=read(kind+'_blockscale_swizzled');codes=np.empty((rows,n),np.uint8);codes[:,::2]=w&15;codes[:,1::2]=w>>4
  rr=np.arange(rows)[:,None];cc=np.arange(n//16)[None,:];pad=(n//16+3)//4*4;idx=(((rr//128*(pad//4)+cc//4)*32+rr%32)*4+(rr%128)//32)*4+cc%4;scales=s[idx]
  gg=np.empty((rows,n//64,36),np.uint8);gg[:,:,:4]=scales.reshape(rows,n//64,4);c=codes.reshape(rows,n//64,4,16);gg[:,:,4:]=(c[:,:,:,:8]|(c[:,:,:,8:]<<4)).reshape(rows,n//64,32)
  x=torch.from_numpy(rng.standard_normal((3,n),dtype=np.float32)).bfloat16();a=x.view(torch.uint16).numpy().copy();xf=x.float().numpy();blocks=xf.reshape(3,n//32,32);delta=np.abs(blocks).max(axis=2)/127;qr=np.sign(blocks)*np.floor(np.abs(blocks/delta[:,:,None])+.5);q=np.empty((3,n//32,34),np.uint8);q[:,:,:2]=delta.astype(np.float16).view(np.uint8).reshape(3,n//32,2);q[:,:,2:]=qr.astype(np.int8).view(np.uint8)
  sf=torch.from_numpy(scales.copy()).view(torch.float8_e4m3fn).float().numpy();values=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],np.float32);ref=(values[codes]*np.repeat(sf,16,axis=1))@xf.T
  fixtures.append(dict(layer=layer,expert=expert,kind=kind,n=n,rows=rows,w=w,s=s,g=gg,a=a,q=q,ref=ref.T.copy()))
report={'note':__doc__,'strata_commit':os.popen('git -C /workspace/Strata rev-parse HEAD').read().strip(),'results':[],'errors':[]}
os.sched_setaffinity(0,{0})
for threads in [1,4,8,15]:
 cores=list(range(2,2+threads)) if threads<15 else list(range(1,16));core_array=(C.c_int*threads)(*cores);pool=lib.row_pool_create(threads,core_array)
 try:
  for f in fixtures:
   outputs=[np.empty((3,f['rows']),np.float32) for _ in range(2)]
   def invoke(mode,nt):
    lib.row_run(pool,mode,f['w'].ctypes.data,f['s'].ctypes.data,f['g'].ctypes.data,f['a'].ctypes.data,f['q'].ctypes.data,outputs[mode].ctypes.data,f['rows'],f['n'],nt)
   for nt in [1,3]:
    for mode in [0,1]:
     invoke(mode,nt);ref=f['ref'][:nt];out=outputs[mode][:nt];err=float(np.linalg.norm(out-ref)/np.linalg.norm(ref));report['errors'].append(dict(layer=f['layer'],kind=f['kind'],threads=threads,tokens=nt,engine=mode,relative_l2=err));assert err < (1e-4 if mode==0 else .03),(err,f['kind'],mode)
    for cold in [False,True]:
     times=[[],[]]
     for iteration in range(39):
      modes=[0,1];order.shuffle(modes)
      for mode in modes:
       if cold:
        if mode==0:
         lib.flush_range(f['w'].ctypes.data,f['w'].nbytes);lib.flush_range(f['s'].ctypes.data,f['s'].nbytes)
        else:lib.flush_range(f['g'].ctypes.data,f['g'].nbytes)
       started=time.perf_counter_ns();invoke(mode,nt);us=(time.perf_counter_ns()-started)/1000
       if iteration>=8:times[mode].append(us)
     report['results'].append(dict(layer=f['layer'],expert=f['expert'],matrix=f['kind'],tokens=nt,threads=threads,cores=cores,cold=cold,sglang_median_us=statistics.median(times[0]),strata_median_us=statistics.median(times[1]),sglang_p95_us=sorted(times[0])[-2],strata_p95_us=sorted(times[1])[-2],samples_us=times))
     (R/'rows.json').write_text(json.dumps(report,indent=2)+'\n')
   print('rows',threads,f['layer'],f['kind'],'done',flush=True)
 finally:lib.row_pool_destroy(pool)
report['status']='passed';(R/'rows.json').write_text(json.dumps(report,indent=2)+'\n')
