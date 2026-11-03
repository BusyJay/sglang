"""Capture one actual M5 four-row FFN, keeping full-bank model out of memory."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.gemm import mlx_nvfp4_swiglu as sw
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
O=Path('benchmark/mlx/remote_validation_20261103');C=Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(8*2**30);mx.set_cache_limit(256*2**20)
a=mx.load(C/'layer-26.safetensors');inputs=mx.load(O/'actual_ffn_operands.npz');bank=[]
for name in ('gate_proj','up_proj','down_proj'):
 p='mlp.switch_mlp.'+name+'.';bank.append([a[p+n] for n in ('weight','scales','weight_scale_2')])
mx.eval(bank);del a
x=inputs['c4_l26_x'];ids=inputs['c4_l26_ids']
def project(value,p,ids,work):
 w,s,gs=p
 return nv.mlx_nvfp4_gather_rows(value,w,s,gs*2688,ids,worklist=work)
def native(value,p,ids):
 w,s,gs=p
 return mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4,global_scale=gs*2688)
def ffn(value,ids,mode):
 work=mlx_expert_pair_worklist(ids) if mode=='reuse' else None
 if mode=='fused':act=sw.mlx_nvfp4_swiglu(value[...,None,None,:],*(bank[0][:2]+[bank[0][2]*2688]),*(bank[1][:2]+[bank[1][2]*2688]),ids,limit=7)
 else:
  g,u=[project(value[...,None,None,:],p,ids,work) if mode=='reuse' else native(value[...,None,None,:],p,ids) for p in bank[:2]]
  act=nn.silu(mx.minimum(g,7.0))*mx.clip(u,-7.0,7.0)
 return (project(act,bank[2],ids,work) if mode=='reuse' else native(act,bank[2],ids)).squeeze(-2)
for mode in ('native','reuse','fused'):
 fn=mx.compile(lambda value,route:ffn(value,route,mode));ref=fn(x,ids);mx.eval(ref);mx.synchronize()
 path=O/f'm5_ffn26_{mode}.gputrace';mx.metal.start_capture(str(path));stream=mx.new_stream(mx.gpu)
 with mx.stream(stream):
  value=fn(x+0.00001,ids);mx.eval(value);mx.synchronize(stream)
 mx.metal.stop_capture();print('FFN_CAPTURE',mode,path,flush=True)
