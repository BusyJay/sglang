import json,time,types
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import IndexedWeights,FP8Linear
import importlib
op=importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows');r=IndexedWeights('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5');mx.set_cache_limit(1024*2**20);results=[];mx.random.seed(888)
key='model.language_model.layers.3.mlp.shared_experts.';wg=r.get(key+'gate_proj.weight');wu=r.get(key+'up_proj.weight');sg=r.get(key+'gate_proj.weight_scale');su=r.get(key+'up_proj.weight_scale');weight=mx.concatenate([wg,wu],0);scales=mx.concatenate([sg,su],0);mx.eval(weight,scales)
for count in (1,3,6):
 x=mx.random.normal((count,wg.shape[-1])).astype(mx.bfloat16);mx.eval(x);q=op.mlx_fp8_dynamic_32(x);mx.eval(q)
 def separate():return (op.mlx_fp8_rows(x,wg,sg,block_size=128,quantized_input=q),op.mlx_fp8_rows(x,wu,su,block_size=128,quantized_input=q))
 def bank():return op.mlx_fp8_pair_rows(x,wg,sg,wu,su,q)
 ref=separate();got=bank();mx.eval(ref,got);assert all(mx.array_equal(v,w).item() for v,w in zip(ref,got));trials=[]
 for trial in range(8):
  row={}
  for a,fn in ([('separate',separate),('bank',bank)] if trial%2==0 else [('bank',bank),('separate',separate)]):
   t=time.perf_counter()
   for _ in range(20):mx.eval(fn())
   row[a]=(time.perf_counter()-t)*1000/20
  trials.append(row)
 result={'count':count,'trials_ms':trials};results.append(result);print(result,flush=True)
Path('benchmark/mlx/fp4_optimization_20261103/pair_kernel.json').write_text(json.dumps(results,indent=2))
