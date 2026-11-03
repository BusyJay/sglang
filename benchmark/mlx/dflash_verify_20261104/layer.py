"""Bounded M5 KDA-layer verify profiling; 24 repeats, no full model load."""
import json,time,statistics,hashlib,copy
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.mlx_fp4_cache import RuntimeWeightCache
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache
R=Path('/Users/jay/workspace/sglang');O=R/'benchmark/mlx/dflash_verify_20261104';T=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
mx.set_memory_limit(12*2**30);mx.set_cache_limit(256*2**20)
conf=json.loads((T/'config.json').read_text());a=g.ModelArgs.from_dict(conf);layer=g.DecoderLayer(a,26);assert RuntimeWeightCache(T,conf).load('layer-26',layer)
layer.fuse_hc=True;layer.compile_residual=True;layer.preserve_residual_dtype=True;layer.self_attn.fuse_decode=True
layer.self_attn.prepare_for_inference();layer.mlp.fuse_decode=True;layer.mlp.prepare_for_inference();layer.mlp.switch_mlp.compile_decode=True;layer.mlp.switch_mlp.fuse_gate_up=True;layer.mlp.switch_mlp.prepare_for_inference();layer.mlp.shared_experts.compile_decode=True;layer.mlp.shared_experts.compile_activation=True
operand=mx.load(R/'benchmark/mlx/remote_validation_20261103/actual_ffn_operands.npz')['c4_l26_x'].astype(mx.bfloat16);x=mx.repeat(mx.concatenate([operand,operand],1)[...,None,:],4,-2);mx.eval(x,layer.parameters())
cache=g.NativeCache(4);warm=layer.verify_ffn_rows(x[:,:1],cache);mx.eval(warm);data={'device':mx.device_info(),'scope':__doc__,'samples':[],'checks':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
for count in (4,8):
 value=x[:,:count];prepared=layer._branch_inputs(value,'attn',layer.input_layernorm);normed,post,residual=prepared;proj=layer.self_attn.project_verify_rows(normed);mx.eval(proj,prepared)
 c=clone_cache(cache);branch,snapshots=layer.self_attn.verify_rows(normed,c,projected=proj);attended=layer._branch_output(branch,post,residual);ffn_inputs=layer._branch_inputs(attended,'ffn',layer.post_attention_layernorm);mx.eval(attended,ffn_inputs)
 def entire(v):
  local=clone_cache(cache);return layer.verify_ffn_rows(v,local)
 def attention(v):
  local=clone_cache(cache);return layer.self_attn.verify_rows(v,local)
 def ffn(v):
  y,p,r=layer._branch_inputs(v,'ffn',layer.post_attention_layernorm);return layer._branch_output(layer.mlp(y,batch_fusion=True),p,r)
 funcs={'entire':mx.compile(entire),'attention':mx.compile(attention),'ffn':mx.compile(ffn),'input_projection':mx.compile(layer.self_attn.project_verify_rows),'output_projection':mx.compile(lambda v:g._project_rows(layer.self_attn.o_proj,v)), 'snapshot_kernel':mx.compile(lambda p:g.mlx_kda_decode(p[0],tuple(getattr(layer.self_attn,n+'_conv').conv.weight for n in ('q','k','v')),tuple(cache[i] for i in range(3)),cache[3],p[1],p[2],layer.self_attn.A_log,layer.self_attn.dt_bias,layer.self_attn.o_norm.weight,heads=layer.self_attn.num_heads,norm_eps=layer.self_attn.o_norm.eps,lower_bound=layer.self_attn.lower_bound,snapshots=True)), 'no_snapshot_kernel':mx.compile(lambda p:g.mlx_kda_decode(p[0],tuple(getattr(layer.self_attn,n+'_conv').conv.weight for n in ('q','k','v')),tuple(cache[i] for i in range(3)),cache[3],p[1],p[2],layer.self_attn.A_log,layer.self_attn.dt_bias,layer.self_attn.o_norm.weight,heads=layer.self_attn.num_heads,norm_eps=layer.self_attn.o_norm.eps,lower_bound=layer.self_attn.lower_bound,snapshots=False))}
 kern=funcs['snapshot_kernel'](proj);mx.eval(kern)
 inputs={'entire':value,'attention':normed,'ffn':attended,'input_projection':normed,'output_projection':kern[0],'snapshot_kernel':proj,'no_snapshot_kernel':proj}
 for name,fn in funcs.items():
  mx.eval(fn(inputs[name]));mx.eval(fn(inputs[name]))
 raw={name:[] for name in funcs}
 for block in range(4):
  names=list(funcs) if block%2==0 else list(reversed(funcs))
  for name in names:
   for step in range(6):
    inp=inputs[name];bump=mx.array((step+block*6)*1e-8,mx.bfloat16);mx.eval(bump);v=tuple(t+bump for t in inp) if isinstance(inp,tuple) else inp+bump;mx.eval(v)
    start=time.perf_counter();y=funcs[name](v);mx.eval(y);raw[name].append(time.perf_counter()-start)
 for name,seconds in raw.items():
  row={'count':count,'phase':name,'raw_seconds':seconds,'mean_ms':statistics.mean(seconds)*1000};data['samples'].append(row);print('LAYER24',count,name,row['mean_ms'],flush=True)
 ref=funcs['snapshot_kernel'](proj);no=funcs['no_snapshot_kernel'](proj);mx.eval(ref,no);exact=mx.array_equal(ref[0],no[0]).item() and mx.array_equal(ref[2][-1:],no[2]).item();assert exact;data['checks'].append({'count':count,'snapshot_output_last_state_exact':exact})
 (O/'layer.json').write_text(json.dumps(data,indent=2))
data['complete']=True;data['peak_bytes']=mx.get_peak_memory();(O/'layer.json').write_text(json.dumps(data,indent=2));print('DONE_GIB',data['peak_bytes']/2**30,flush=True)
