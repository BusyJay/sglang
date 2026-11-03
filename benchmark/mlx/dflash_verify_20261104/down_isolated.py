"""M5 short verification: identify the >=64 expert-pick sorted dispatch boundary."""
import json,time,statistics,hashlib
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.mlx_fp4_cache import RuntimeWeightCache
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache
R=Path('/Users/jay/workspace/sglang');O=R/'benchmark/mlx/dflash_verify_20261104';T=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
mx.set_memory_limit(12*2**30);mx.set_cache_limit(256*2**20);conf=json.loads((T/'config.json').read_text());a=g.ModelArgs.from_dict(conf);layer=g.DecoderLayer(a,26);assert RuntimeWeightCache(T,conf).load('layer-26',layer)
layer.fuse_hc=True;layer.compile_residual=True;layer.preserve_residual_dtype=True;layer.self_attn.fuse_decode=True;layer.self_attn.prepare_for_inference();layer.mlp.fuse_decode=True;layer.mlp.prepare_for_inference();layer.mlp.switch_mlp.compile_decode=True;layer.mlp.switch_mlp.fuse_gate_up=True;layer.mlp.switch_mlp.prepare_for_inference();layer.mlp.shared_experts.compile_decode=True;layer.mlp.shared_experts.compile_activation=True
from down_probe import DownCombineMoE
operand=mx.load(R/'benchmark/mlx/remote_validation_20261103/actual_ffn_operands.npz')['c4_l26_x'].astype(mx.bfloat16);x=mx.repeat(mx.concatenate([operand,operand],1)[...,None,:],4,-2);mx.eval(x,layer.parameters());cache=g.NativeCache(4);mx.eval(layer.verify_ffn_rows(x[:,:1],cache))
from down_probe import fused_down
mx.set_wired_limit(12*2**30)
data={'samples':[],'checks':[]}
for count in (4,8):
 value=x[:,:count];normed=layer._branch_inputs(value,'ffn',layer.post_attention_layernorm)[0];mx.eval(normed)
 captured={}
 old=fused_down
 import down_probe
 def capture(act,ids,scores,p,shared,rps,tile):
  captured.update(act=act,ids=ids,scores=scores,p=p,shared=shared)
  return old(act,ids,scores,p,shared,rps,tile)
 # The probe callable lives in the method's independent globals, so install before tracing.
 DownCombineMoE.__call__.__globals__['fused_down']=capture
 module=g._clone_graph_modules(layer.mlp);module.__class__=DownCombineMoE;module.probe_rps=4;module.probe_tile=1
 result=module(normed,batch_fusion=True);mx.eval(result,*[v for v in captured.values() if isinstance(v,mx.array)])
 act,ids,scores,proj,shared=[captured[k] for k in ('act','ids','scores','p','shared')]
 DownCombineMoE.__call__.__globals__['fused_down']=old
 def native(a):
  output=proj(a,ids).squeeze(-2)
  return (output*scores[...,None]).sum(-2).astype(shared.dtype)+shared
 funcs={'native':mx.compile(native)}
 for rps in (1,4,8):
  for tile in (1,2):
   funcs[f'r{rps}t{tile}']=mx.compile(lambda a,rps=rps,tile=tile:old(a,ids,scores,proj,shared,rps,tile))
 refs=funcs['native'](act);mx.eval(refs)
 for name,fn in funcs.items():
  out=fn(act);mx.eval(out);exact=mx.array_equal(refs,out).item();assert exact,(count,name)
  data['checks'].append({'count':count,'mode':name,'exact':exact})
  for _ in range(3):mx.eval(fn(act))
 raw={k:[] for k in funcs}
 for block in range(4):
  for name in list(funcs) if block%2==0 else list(reversed(funcs)):
   for step in range(6):
    inp=act+mx.array((step+block*6)*1e-8,act.dtype);mx.eval(inp);t=time.perf_counter();out=funcs[name](inp);mx.eval(out);raw[name].append(time.perf_counter()-t)
 for name,seconds in raw.items():
  row={'count':count,'mode':name,'raw':seconds,'mean_ms':statistics.mean(seconds)*1000};data['samples'].append(row);print('DOWN_ISOLATE',count,name,row['mean_ms'],flush=True)
 (O/'down_isolated.json').write_text(json.dumps(data,indent=2))
