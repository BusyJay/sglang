"""Diagnostic only: unchanged CPU math, sleeping vs bounded-spin pool, wider core limit.
No production source is modified. No-op gaps excluded from dispatch timing.
"""
import ctypes as C, json, os, statistics, time
from pathlib import Path
import torch
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import Qwen38ExpertPack, load_manifest
from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4
os.environ.setdefault('SGLANG_QWEN38_SOURCE_CACHE_DIR','/dev/shm/sglang-qwen38-startup')
R=Path('/results');torch.set_num_threads(1)
m=load_manifest(Path('/root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json'))
pack=Qwen38ExpertPack(m,source_mode='shared_thp')
ops={mode:CpuNvfp4(str(R/f'pool-{mode}/kernel.so'),pack,(2,3,4,5),gpu_backend='b12x') for mode in ['cv','spin']}
for op in ops.values():
 f=op.library.bench_batch;f.argtypes=op.native_run.argtypes;f.restype=C.c_int;op.native_run=f
 op.library.bench_noop.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double)]
flush=C.CDLL('/results/row_bridge.so').flush_range;flush.argtypes=[C.c_void_p,C.c_size_t]
os.sched_setaffinity(0,{0});torch.manual_seed(42)
report={'note':__doc__,'noop':[],'full':[]};references={}
def save(): (R/'pool.json').write_text(json.dumps(report,indent=2)+'\n')
def stats(values): return dict(median_us=statistics.median(values),p95_us=sorted(values)[int(len(values)*.95)],samples_us=values)
for threads in [4,8,15]:
 cores=list(range(2,2+threads)) if threads<15 else list(range(1,16))
 for mode in ['cv','spin']:
  op=ops[mode];op.cores=(C.c_int*threads)(*cores)
  for gap in [0,200,2000,25000]:
   times=(C.c_double*39)();op.library.bench_noop(op.cores,threads,39,gap,times)
   report['noop'].append(dict(mode=mode,threads=threads,gap_us=gap,**stats(list(times)[8:])))
  op.release_worker();save()
 for nt in [1,3]:
  x=torch.randn((nt,m.hidden_size),dtype=torch.bfloat16)
  for count in [1,3]:
   ids=torch.full((nt,m.top_k),-1,dtype=torch.int32);ids[:,:count]=torch.tensor([7,111,217][:count]);weights=torch.ones(ids.shape,dtype=torch.float32)/m.top_k
   for cold in [False,True]:
    for mode in (['cv','spin'] if (nt+count+threads)%2 else ['spin','cv']):
     op=ops[mode];op.cores=(C.c_int*threads)(*cores);out=torch.empty_like(x,dtype=torch.float32);times=[]
     for iteration in range(39):
      if cold:
       for name in ['w13_weight','w2_weight','w13_blockscale_swizzled','w2_blockscale_swizzled']:
        rec=m.tensors[22][name];stride=rec.nbytes//m.num_experts
        for expert in [7,111,217][:count]:flush(op.address+rec.offset+expert*stride,stride)
      t=time.perf_counter_ns();op.run(22,x,ids,weights,out);elapsed=(time.perf_counter_ns()-t)/1000
      if iteration>=8:times.append(elapsed)
     key=(threads,nt,count,cold)
     if key in references: assert torch.equal(out,references.pop(key)),key
     else: references[key]=out.clone()
     report['full'].append(dict(mode=mode,threads=threads,tokens=nt,experts=count,cold=cold,**stats(times)))
     op.release_worker();save()
   print('full',threads,nt,count,'done',flush=True)
report['status']='passed';save()
