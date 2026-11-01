import json,time,hashlib,importlib,statistics
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing,clone_cache
from sglang.srt.hardware_backend.mlx.models import glm5_next as G
OUT=Path(__file__).parent
M=importlib.import_module('sglang.kernels.ops.layernorm.mlx_hc_mix')
new_mix=G.mlx_hc_rms_mix
ns={};exec(compile((OUT/'baseline/python/sglang/kernels/ops/layernorm/mlx_hc_mix.py').read_text(),'prior_mix','exec'),ns)
old_mix=ns['mlx_hc_rms_mix']
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
model=load_mtp_model(Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'))
base=model.make_cache();engine=MlxMTPServing(model,2)
for start in range(0,2050,256):
 h=engine.prefill('base',(mx.arange(start,min(start+256,2050))%128)[None],base);mx.eval(h)
root=mx.argmax(model.lm_head(h[:,-1:]),-1);mx.eval(root);engine.remove('base')
def configure(arm):
 G.mlx_hc_rms_mix=new_mix if arm=='parallel' else old_mix
 model.configure_fixed_groups(True)
rows=[]
for steps in (2,):
 for group in range(2):
  for arm in ('baseline','parallel','parallel','baseline'):
   configure(arm);cache=[clone_cache(c) for c in base];cache[0].mlx_mtp=base[0].mlx_mtp;e=MlxMTPServing(model,steps);token=int(root.item());tokens=[];raw=[]
   for j in range(64):
    t=time.perf_counter();out=e.decode_batch(['probe'],mx.array([[token]]),[cache],[token],token_only=True);mx.eval(out,[c.cache for c in cache]);token=int(out.item());tokens.append(token);raw.append(time.perf_counter()-t)
   rows.append(dict(arm=arm,steps=steps,group=group,tok_s=59/sum(raw[5:]),all_tok_s=64/sum(raw),raw=raw,hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),stats=e.stats(),active=mx.get_active_memory(),peak=mx.get_peak_memory()));e.remove('probe')
   (OUT/'mhc_full.json').write_text(json.dumps(rows,indent=2));print(steps,group,arm,round(rows[-1]['tok_s'],2),flush=True)
