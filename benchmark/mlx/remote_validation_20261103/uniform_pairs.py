"""Diagnostic independent uniform pair/single kernels with functional merge."""
from functools import cache
import mlx.core as mx
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
@cache
def kernel(paired):
 source=nv._SOURCE
 if paired:
  source=source.replace('if(nmatch==0)return;','if(nmatch!=2)return;')
  start=source.index('if(SINGLES&&nmatch==1){');end=source.index('float a[RPS]={0},b[RPS]={0};')
  source=source[:start]+source[end:]
 else:
  source=source.replace('if(nmatch==0)return;','if(nmatch!=1)return;')
  start=source.index('if(SINGLES&&nmatch==1){');end=source.index('float a[RPS]={0},b[RPS]={0};')
  source=source[:start]+source[start:end].replace('if(SINGLES&&nmatch==1){','{')
 return mx.fast.metal_kernel(name='diagnostic_uniform_nvfp4_'+str(paired),input_names=['x','weight','scales','global_scale','ids','dims','partners','counts'],output_names=['out'],source=source,header=nv._HEADER,compile_options={'math_mode':'safe'})
def project(value,p,ids,work):
 w,s,gs=p;e,n,pk=w.shape;k=pk*2;count,picks=ids.size,ids.shape[-1]
 results=[]
 for paired,rps in ((False,4),(True,2)):
  results.append(kernel(paired)(inputs=[value.reshape(-1,k),w,s,gs,ids.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',rps),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0])
 # Pair outputs include suppressed second picks; singletons have count==1.
 return mx.where(work[1][:,None]==1,results[0],results[1]).reshape(*ids.shape,1,n)
