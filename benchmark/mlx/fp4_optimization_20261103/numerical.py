from pathlib import Path
import json,hashlib
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import IndexedWeights,mxfp8_project_rows
import importlib
op=importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows')
reader=IndexedWeights('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5');mx.set_cache_limit(64*2**20);mx.random.seed(912);result=[]
for prefix in ['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.o_proj','lm_head']:
 w=reader.get(prefix+'.weight');sc=reader.get(prefix+'.weight_scale');mx.eval(w,sc)
 for dtype in (mx.bfloat16,mx.float32):
  for count in (1,3,6,9):
   x=(mx.random.normal((count,w.shape[-1]))*.5).astype(dtype);mx.eval(x);y=mxfp8_project_rows(x,w,sc);ref=mx.concatenate([mxfp8_project_rows(x[i:i+1],w,sc) for i in range(count)],0);mx.eval(y,ref);delta=float(mx.max(mx.abs(y-ref)).item());assert delta==0,(prefix,dtype,count,delta);result.append({'prefix':prefix,'dtype':str(dtype),'count':count,'maxdiff':delta})
 del w,sc,x,y,ref;mx.clear_cache()
Path('benchmark/mlx/fp4_optimization_20261103/mxfp8_invariance.json').write_text(json.dumps(result,indent=2));print('REAL_MXFP8_BATCH_INVARIANCE',len(result),flush=True)
