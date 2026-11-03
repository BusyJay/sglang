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
from multi_reuse_probe import MultiReuseSwitch
x=mx.load(O/'real_verify_residual.npz')['residual'];mx.set_wired_limit(12*2**30);mx.eval(x,layer.parameters());cache=g.NativeCache(4);mx.eval(layer.verify_ffn_rows(x[:,:1],cache))
from collections import Counter
class CaptureSwitch(g.ClampedSwitchGLU):
 def __call__(self,x,indices):
  self.route_indices=indices;self.route_inputs=x
  return super().__call__(x,indices)
layer.mlp.switch_mlp.__class__=CaptureSwitch
operand=mx.load(R/'benchmark/mlx/remote_validation_20261103/actual_ffn_operands.npz')['c4_l26_x'].astype(mx.bfloat16)
synthetic=mx.repeat(mx.concatenate([operand,operand],1)[...,None,:],4,-2)
results=[]
for label,residual in [('constructed',synthetic),('real',x)]:
 normed=layer._branch_inputs(residual,'ffn',layer.post_attention_layernorm)[0];mx.eval(normed)
 out=layer.mlp(normed,batch_fusion=True);mx.eval(out,layer.mlp.switch_mlp.route_indices)
 ids=layer.mlp.switch_mlp.route_indices.tolist();hist=Counter(v for rows in ids for row in rows for v in row);fills={reuse:dict(Counter(min(reuse,n-start) for n in hist.values() for start in range(0,n,reuse))) for reuse in (2,4)}
 row={'label':label,'norm_dtype':str(normed.dtype),'indices':ids,'unique_experts':len(hist),'frequency_histogram':dict(Counter(hist.values())),'group_fill_histogram':fills,'mean_abs_row_change':float(mx.abs(normed[:,1:].astype(mx.float32)-normed[:,:-1].astype(mx.float32)).mean().item())};results.append(row);print('REAL_ROUTE',row,flush=True)
(O/'reuse_distribution.json').write_text(json.dumps(results,indent=2))
