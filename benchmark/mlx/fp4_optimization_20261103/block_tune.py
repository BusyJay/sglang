import json,time,types
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import IndexedWeights,FP8Linear
from sglang.kernels.ops.gemm import mlx_fp8_rows as unused
import importlib
op=importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows')
r=IndexedWeights('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5');mx.set_cache_limit(64*2**20);results=[];mx.random.seed(222)
mods={}
src=Path(op.__file__).read_text()
for rps in (1,2,4,8):
 mod=types.ModuleType('variant'+str(rps));code=src.replace('rps = 4 if n >= 4096 else 1','rps = '+str(rps)).replace('sglang_fp8_shared_rows','sglang_fp8_shared_rows_rps'+str(rps));exec(compile(code,'variant','exec'),mod.__dict__);mods[rps]=mod
for suffix in ('mlp.shared_experts.gate_proj','mlp.shared_experts.down_proj','self_attn.q_b_proj','self_attn.o_proj'):
 prefix='model.language_model.layers.3.'+suffix;w=r.get(prefix+'.weight');sc=r.get(prefix+'.weight_scale');mx.eval(w,sc)
 for count in (1,3):
  x=mx.random.normal((count,w.shape[-1])).astype(mx.bfloat16);mx.eval(x);q=op.mlx_fp8_dynamic_32(x);mx.eval(q);ref=op.mlx_fp8_rows(x,w,sc,block_size=128,quantized_input=q);mx.eval(ref);arms={rps:lambda rps=rps:mods[rps].mlx_fp8_rows(x,w,sc,block_size=128,quantized_input=q) for rps in mods};values={a:fn() for a,fn in arms.items()};mx.eval(values);diff={a:float(mx.max(mx.abs(v-ref)).item()) for a,v in values.items()};trials=[]
  for trial in range(6):
   row={}
   for a,fn in list(arms.items())[::1 if trial%2==0 else -1]:
    mx.eval(fn());t=time.perf_counter()
    for _ in range(10):mx.eval(fn())
    row[a]=(time.perf_counter()-t)*1000/10
   trials.append(row)
  row={'prefix':prefix,'shape':list(w.shape),'count':count,'maxdiff':diff,'trials_ms':trials};results.append(row);print(row,flush=True);Path('benchmark/mlx/fp4_optimization_20261103/block_tune.json').write_text(json.dumps(results,indent=2))
 del w,sc,x,q,ref,values;mx.clear_cache()
