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
import inspect,textwrap,importlib
hc=importlib.import_module('sglang.kernels.ops.layernorm.mlx_hc')
original_hc=g.mlx_hc_collapse_norm
src=textwrap.dedent(inspect.getsource(original_hc));start=src.index('    residual = mx.concatenate(');stop=src.index('    return normed',start)
src=src[:start]+"    residual = comb.swapaxes(-1,-2) @ x\n"+src[stop:]
ns=dict(hc.__dict__);exec(src,ns);batched_hc=ns['mlx_hc_collapse_norm']
operand=mx.load(R/'benchmark/mlx/remote_validation_20261103/actual_ffn_operands.npz')['c4_l26_x'].astype(mx.bfloat16);x=mx.repeat(mx.concatenate([operand,operand],1)[...,None,:],4,-2);mx.eval(x,layer.parameters());cache=g.NativeCache(4);mx.eval(layer.verify_ffn_rows(x[:,:1],cache))
data={'device':mx.device_info(),'scope':__doc__,'samples':[],'checks':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
for count in (4,6,8):
 value=x[:,:count];normed=layer._branch_inputs(value,'ffn',layer.post_attention_layernorm)[0];mx.eval(normed)
 refs={};fns={}
 for mode in ('original','batched'):
  module=g._clone_graph_modules(layer);g.mlx_hc_collapse_norm=original_hc if mode=='original' else batched_hc
  def entire(v,module=module):return module.verify_ffn_rows(v,clone_cache(cache))
  def ffn(v,module=module):
   y,p,r=module._branch_inputs(v,'ffn',module.post_attention_layernorm);return module._branch_output(module.mlp(y,batch_fusion=True),p,r)
  fns[(mode,'entire')]=mx.compile(entire);fns[(mode,'ffn')]=mx.compile(ffn);fns[(mode,'moe')]=mx.compile(lambda v,m=module:m.mlp(v,batch_fusion=True))
  for phase in ('entire','ffn','moe'):mx.eval(fns[(mode,phase)](normed if phase=='moe' else value))
 for phase in ('entire','ffn','moe'):
  inp=normed if phase=='moe' else value;ref=fns[('original',phase)](inp);out=fns[('batched',phase)](inp);mx.eval(ref,out)
  def arrays(v):
   from mlx.utils import tree_flatten
   return [t for _,t in tree_flatten(v) if isinstance(t,mx.array)]
  exact=all(mx.array_equal(a,b).item() for a,b in zip(arrays(ref),arrays(out)));assert exact,(count,phase);data['checks'].append({'count':count,'phase':phase,'exact':exact})
  for key in ((m,phase) for m in ('original','batched')):
   for _ in range(3):mx.eval(fns[key](inp))
 raw={k:[] for k in fns}
 for block in range(4):
  keys=list(fns) if block%2==0 else list(reversed(fns))
  for key in keys:
   for step in range(6):
    inp=(normed if key[1]=='moe' else value)+mx.array((step+block*6)*1e-8,mx.bfloat16);mx.eval(inp);t=time.perf_counter();out=fns[key](inp);mx.eval(out);raw[key].append(time.perf_counter()-t)
 for (mode,phase),seconds in raw.items():
  row={'count':count,'mode':mode,'phase':phase,'raw_seconds':seconds,'mean_ms':statistics.mean(seconds)*1000};data['samples'].append(row);print('HC_BATCH24',count,mode,phase,row['mean_ms'],flush=True)
 (O/'hc_batch.json').write_text(json.dumps(data,indent=2))
data['complete']=True;(O/'hc_batch.json').write_text(json.dumps(data,indent=2))
