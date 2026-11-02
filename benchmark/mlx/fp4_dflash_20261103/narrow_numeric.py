import sys,json
from pathlib import Path
import mlx.core as mx
sys.path.insert(0,'benchmark/mlx/ar_optimization_20261103')
from native_nvfp4 import project
mx.random.seed(83);results=[]
for dtype in (mx.float32,mx.bfloat16):
 for n,k in ((128,512),(512,4096),(4096,512)):
  g=mx.full((8,),2688*0.01,mx.float32)
  w,s=mx.quantize(mx.random.normal((8,n,k)),mode='nvfp4',group_size=16,bits=4,global_scale=mx.array(2688*0.01,mx.float32))
  x=mx.random.normal((1,4,1,1,k)).astype(dtype);ids=mx.array([[[0,2],[2,4],[4,7],[0,7]]],mx.uint32)
  native=mx.gather_qmm(x,w,s,global_scale=g,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4)
  row=project(x,w.view(mx.uint8),s,g/2688,ids,rps=1)
  singleton=mx.concatenate([mx.gather_qmm(x[:,i:i+1],w,s,global_scale=g,rhs_indices=ids[:,i:i+1],mode='nvfp4',group_size=16,bits=4) for i in range(4)],1)
  mx.eval(native,row,singleton)
  delta=row.astype(mx.float32)-native.astype(mx.float32)
  result=dict(dtype=str(dtype),n=n,k=k,row_exact=mx.array_equal(row,native).item(),native_batch_exact=mx.array_equal(native,singleton).item(),max_abs=mx.max(mx.abs(delta)).item(),rms=mx.sqrt(mx.mean(delta*delta)).item());results.append(result);print(result,flush=True)
Path('benchmark/mlx/fp4_dflash_20261103/narrow_numeric.json').write_text(json.dumps(results,indent=2))
