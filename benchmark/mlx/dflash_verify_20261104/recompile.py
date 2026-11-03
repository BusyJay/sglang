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
owner=g._FixedLayerGroup([layer]);value=x[:,:4]
records=[];old=g._observe_graph;plans=[]
def observe(who,modules):
 before=getattr(who,'_observation',None);start=time.perf_counter();out=old(who,modules);records.append({'phase':'observe','seconds':time.perf_counter()-start,'rebuild':getattr(who,'_observation',None) is not before});return out
g._observe_graph=observe
for step in range(12):
 records=[];state=clone_cache(cache);value=x[:,:4]+mx.array(step*1e-8,mx.bfloat16);mx.eval(value)
 start=time.perf_counter();out=owner(value,[state],verification=True);submit=time.perf_counter()-start;mx.eval(out)
 row={'step':step,'seconds':time.perf_counter()-start,'submit_seconds':submit,'graphs':len(owner.graphs),'observations':records};data['samples'].append(row);print('REPLAY12',row,flush=True)
(O/'recompile.json').write_text(json.dumps(data,indent=2))
