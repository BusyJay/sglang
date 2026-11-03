"""Diagnostic uniform active-pick compaction for native-order paired NVFP4."""
from functools import cache
import mlx.core as mx
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
_SOURCE=r'''
if(thread_position_in_grid.x!=0)return;
uint count=dims[0],used=0;
for(uint pick=0;pick<count;++pick){
 uint expert=ids[pick],prior=0,next=pick;
 for(uint i=0;i<pick;++i)prior+=ids[i]==expert;
 if(prior&1)continue;
 for(uint i=pick+1;i<count;++i)if(ids[i]==expert){next=i;break;}
 starts[used]=pick;partners[used]=next;counts[used]=next==pick?1:2;++used;
}
active[0]=used;
for(uint i=used;i<count;++i){starts[i]=0;partners[i]=0;counts[i]=0;}
'''
@cache
def work_kernel():
 return mx.fast.metal_kernel(name='diagnostic_expert_active_work',input_names=['ids','dims'],output_names=['starts','partners','counts','active'],source=_SOURCE)
def compact_work(ids):
 return tuple(work_kernel()(inputs=[ids.reshape(-1),mx.array([ids.size],mx.uint32)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(ids.size,)]*3+[(1,)],output_dtypes=[mx.uint32]*4))
@cache
def kernel(kind):
 source=nv._SOURCE
 old='uint pick=threadgroup_position_in_grid.z,nmatch=counts[pick];\nif(nmatch==0)return;\nuint expert=ids[pick],partner=partners[pick];'
 if kind=='mapped':
  source=source.replace(old,'uint slot=threadgroup_position_in_grid.z;\nif(slot>=active[0])return;\nuint pick=starts[slot],nmatch=counts[slot],partner=partners[slot],expert=ids[pick];')
 elif kind=='loop':
  source=source.replace(old,'uint slot=threadgroup_position_in_grid.z;\nfor(;slot<active[0];slot+=dims[4]){\nuint pick=starts[slot],nmatch=counts[slot],partner=partners[slot],expert=ids[pick];')
  # A singleton finishes this pick, then continues the persistent slot loop.
  source=source.replace('return;\n}\nfloat a[RPS]','continue;\n}\nfloat a[RPS]')+'\n}\n'
 else:raise ValueError(kind)
 return mx.fast.metal_kernel(name='diagnostic_nvfp4_compact_'+kind,input_names=['x','weight','scales','global_scale','ids','dims','starts','partners','counts','active'],output_names=['out'],source=source,header=nv._HEADER,compile_options={'math_mode':'safe'})
def project(value,p,ids,work,rps=4,slots=16):
 w,s,gs=p;e,n,pk=w.shape;k=pk*2;count,picks=ids.size,ids.shape[-1]
 kind='mapped' if slots==count else 'loop'
 return kernel(kind)(inputs=[value.reshape(-1,k),w,s,gs,ids.reshape(-1),mx.array([k,n,picks,count,slots],mx.int32),*work],template=[('O',value.dtype),('RPS',rps),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,2*((n+2*rps-1)//(2*rps)),slots),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
