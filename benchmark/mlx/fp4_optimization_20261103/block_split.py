import json,time,types,ast,copy
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import IndexedWeights,FP8Linear
import importlib
op=importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows');r=IndexedWeights('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5');mx.set_cache_limit(64*2**20);results=[];mx.random.seed(333)
src=Path(op.__file__).read_text();mods={}
# Partition K across SIMD groups, same lane-strided partial within each chunk.
for split in (1,2,4,8):
 for rps in (1,2):
  code=src.replace('rps = 2 if block_size == 128 and n >= 4096 else 4 if n >= 4096 else 1','rps = '+str(rps)).replace('sglang_fp8_shared_rows',f'sglang_fp8_split{split}_rps{rps}')
  code=code.replace('uint row = (threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup) * RPS;',f'uint row = (threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup) * RPS;')
  code=code.replace('for (uint k = lane; k < K; k += 32)',f'for (uint k = threadgroup_position_in_grid.y * (K / {split}) + lane; k < (threadgroup_position_in_grid.y + 1) * (K / {split}); k += 32)')
  code=code.replace('out[(start + t) * N + row + r] = O(acc[t][r]);',f'out[(threadgroup_position_in_grid.y * COUNT + start + t) * N + row + r] = O(acc[t][r]);')
  code=code.replace('("O", x.dtype),','("O", mx.float32),').replace('output_shapes=[(count, n)]','output_shapes=[('+str(split)+',count,n)]').replace('output_dtypes=[x.dtype]','output_dtypes=[mx.float32]').replace('            1,\n            (count +',f'            {split},\n            (count +')
  code=code.replace(')[0].reshape(*shape[:-1], n)',')[0].sum(0).astype(x.dtype).reshape(*shape[:-1], n)')
  mod=types.ModuleType('split');exec(compile(code,'split','exec'),mod.__dict__);mods[(split,rps)]=mod
for suffix in ('mlp.shared_experts.gate_proj','self_attn.o_proj'):
 prefix='model.language_model.layers.3.'+suffix;w=r.get(prefix+'.weight');sc=r.get(prefix+'.weight_scale');mx.eval(w,sc)
 for count in (1,3):
  x=mx.random.normal((count,w.shape[-1])).astype(mx.bfloat16);mx.eval(x);q=op.mlx_fp8_dynamic_32(x);mx.eval(q);ref=op.mlx_fp8_rows(x,w,sc,block_size=128,quantized_input=q);mx.eval(ref);arms={'current':lambda:op.mlx_fp8_rows(x,w,sc,block_size=128,quantized_input=q)}
  for key,mod in mods.items():arms[str(key)]=lambda mod=mod:mod.mlx_fp8_rows(x,w,sc,block_size=128,quantized_input=q)
  values={a:fn() for a,fn in arms.items()};mx.eval(values);diff={a:float(mx.max(mx.abs(v-ref)).item()) for a,v in values.items()};trials=[]
  for trial in range(4):
   row={}
   for a,fn in list(arms.items())[::1 if trial%2==0 else -1]:
    mx.eval(fn());t=time.perf_counter()
    for _ in range(5):mx.eval(fn())
    row[a]=(time.perf_counter()-t)*1000/5
   trials.append(row)
  row={'prefix':prefix,'shape':list(w.shape),'count':count,'maxdiff':diff,'trials_ms':trials};results.append(row);print(row,flush=True);Path('benchmark/mlx/fp4_optimization_20261103/block_split.json').write_text(json.dumps(results,indent=2))
 del w,sc,x,q,ref,values;mx.clear_cache()
