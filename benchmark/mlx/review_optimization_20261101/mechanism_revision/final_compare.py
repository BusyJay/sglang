import json,time,hashlib,importlib,statistics
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing,clone_cache
from sglang.srt.hardware_backend.mlx.models import glm5_next as G
OUT=Path(__file__).parent
A=importlib.import_module('sglang.kernels.ops.gemm.mlx_affine_rows');H=importlib.import_module('sglang.kernels.ops.gemm.mlx_dense_rows')
ns={};exec(compile((OUT/'baseline/python/sglang/kernels/ops/gemm/mlx_affine_rows.py').read_text(),'prior_gate','exec'),ns)
old_gate=ns['mlx_affine_routed_swiglu'];new_gate=G.mlx_affine_routed_swiglu;new_geo=H._dense_geometry
ns_mix={};exec(compile((OUT/'baseline/python/sglang/kernels/ops/layernorm/mlx_hc_mix.py').read_text(),'prior_mix','exec'),ns_mix)
new_mix=G.mlx_hc_rms_mix
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
model=load_mtp_model(Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'))
base=model.make_cache();engine=MlxMTPServing(model,2)
for start in range(0,2050,256):
 h=engine.prefill('base',(mx.arange(start,min(start+256,2050))%128)[None],base);mx.eval(h)
root=mx.argmax(model.lm_head(h[:,-1:]),-1);mx.eval(root);engine.remove('base')
def old_geo(n,k=None,count=1):
 if k==4096 and n==24896 and count==3:return 4,8
 if k==4096 and n==154880 and count>=9:return 8,8
 return (8 if n>=4096 else 4,4)
def configure(arm):
 G.mlx_affine_routed_swiglu=new_gate if arm=='final' else old_gate
 H._dense_geometry=new_geo if arm=='final' else old_geo
 # Retained graphs capture kernel/geometry choice. Invalidate each explicit
 # experiment arm; discard cold calls rather than time them as steady state.
 G.mlx_hc_rms_mix=new_mix if arm=='final' else ns_mix['mlx_hc_rms_mix']
 model.configure_fixed_groups(True)
 for layer in model.layers:
  sw=getattr(layer.mlp,'switch_mlp',None)
  if sw is not None and hasattr(sw,'_compiled_gate_up_spec'):sw._compiled_gate_up_spec=None
files=['python/sglang/kernels/ops/gemm/mlx_affine_rows.py','python/sglang/kernels/ops/gemm/mlx_dense_rows.py','python/sglang/kernels/ops/layernorm/mlx_hc_mix.py']
(OUT/'final_compare_source.json').write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=2))
rows=[]
for steps in (0,2):
 for group in range(2):
  for arm in ('baseline','final','final','baseline'):
   configure(arm);cache=[clone_cache(c) for c in base];cache[0].mlx_mtp=base[0].mlx_mtp;e=MlxMTPServing(model,steps);token=int(root.item());tokens=[];raw=[]
   for j in range(48):
    t=time.perf_counter();out=e.decode_batch(['probe'],mx.array([[token]]),[cache],[token],token_only=True);mx.eval(out,[c.cache for c in cache]);token=int(out.item());tokens.append(token);raw.append(time.perf_counter()-t)
   rows.append(dict(arm=arm,steps=steps,group=group,tok_s=43/sum(raw[5:]),all_tok_s=48/sum(raw),raw=raw,hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),stats=e.stats(),active=mx.get_active_memory(),peak=mx.get_peak_memory()));e.remove('probe')
   (OUT/'final_compare.json').write_text(json.dumps(rows,indent=2));print(steps,group,arm,round(rows[-1]['tok_s'],2),flush=True)
