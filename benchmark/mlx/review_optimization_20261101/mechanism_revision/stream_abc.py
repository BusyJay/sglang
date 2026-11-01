"""A fixed group; B same split graph/single stream; C same split/side stream."""
import sys,json,time,hashlib,importlib.util,statistics,cProfile,pstats,io
from pathlib import Path
import mlx.core as mx
OUT=Path(__file__).parent
# Isolated module replaces the model class only within this diagnostic process.
name='sglang.srt.hardware_backend.mlx.models.glm5_next'
spec=importlib.util.spec_from_file_location(name,OUT/'stream_abc_model.py');G=importlib.util.module_from_spec(spec);sys.modules[name]=G;spec.loader.exec_module(G)
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing,clone_cache
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
model=load_mtp_model(Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'))
base=model.make_cache();pre=MlxMTPServing(model,2)
for start in range(0,2050,256):
 h=pre.prefill('base',(mx.arange(start,min(start+256,2050))%128)[None],base);mx.eval(h)
root=mx.argmax(model.lm_head(h[:,-1:]),-1);mx.eval(root);pre.remove('base');side=mx.new_stream(mx.gpu)
rows=[]
def run(arm,steps,count=32):
 model.configure_shared_stream({'A':None,'B':'split_single','C':side}[arm])
 cache=[clone_cache(c) for c in base];cache[0].mlx_mtp=base[0].mlx_mtp;engine=MlxMTPServing(model,steps);token=int(root.item());raw=[];tokens=[]
 for _ in range(count):
  t=time.perf_counter();out=engine.decode_batch(['probe'],mx.array([[token]]),[cache],[token],token_only=True);mx.eval(out,[c.cache for c in cache]);token=int(out.item());raw.append(time.perf_counter()-t);tokens.append(token)
 engine.remove('probe')
 return dict(arm=arm,steps=steps,tok_s=count/sum(raw),raw=raw,hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
for steps in (0,2):
 # First instances separated, then two bounded paired blocks.
 for arm in ('A','B','C'):run(arm,steps,8)
 for repeat in range(2):
  for arm in ('A','B','C','C','B','A'):
   row=run(arm,steps);row['repeat']=repeat;rows.append(row);print(steps,arm,round(row['tok_s'],2),flush=True);(OUT/'stream_abc.json').write_text(json.dumps(rows,indent=2))
 # Dedicated cProfile round excluded from throughput, separates host submit.
 for arm in ('A','B','C'):
  pr=cProfile.Profile();pr.enable();run(arm,steps,8);pr.disable();pr.dump_stats(str(OUT/f'host_{steps}_{arm}.pstats'))
  buf=io.StringIO();pstats.Stats(pr,stream=buf).sort_stats('tottime').print_stats(18);(OUT/f'host_{steps}_{arm}.txt').write_text(buf.getvalue())
model.configure_shared_stream(None)
