import json,time,hashlib,subprocess,collections
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_fp4 as f
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing
root=Path('benchmark/mlx/fp4_optimization_20261103');target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
mx.set_memory_limit(188*2**30);mx.set_wired_limit(188*2**30);mx.set_cache_limit(64*2**20)
r={'device':mx.device_info(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'samples':[],'micro':[],'attribution':[],'swap_before':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}
def save(): (root/'runtime_ab.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
t=time.perf_counter();m,_=f.load_fp4_model(target,enable_mtp=True);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters());r['load_s']=time.perf_counter()-t;print('LOADED',r['load_s'],mx.get_active_memory()/2**30,flush=True)
tok=load_tokenizer(target);engine=MlxMTPServing(m,2)
texts=['Explain why the sky is blue.','Write a Python function that computes the first n Fibonacci numbers, with an explanation.','请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
def generate(text,count=64):
 engine.clear();cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True);h=engine.prefill('p',mx.array([p],mx.int32),cache);mx.eval(h,[c.cache for c in cache]);token=int(m.target_tokens(h[:,-1:]).item());ids=[token];times=[]
 for i in range(count-1):
  engine.set_remaining('p',count-i-1);t=time.perf_counter();v=engine.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(v,[c.cache for c in cache]);token=int(v.item());times.append(time.perf_counter()-t);ids.append(token)
 row={'prompt':p,'ids':ids,'raw_s':times,'tok_s':len(times)/sum(times),'stats':engine.stats(),'active':mx.get_active_memory(),'peak':mx.get_peak_memory()};engine.remove('p');return row
import cProfile,pstats,io
prof=cProfile.Profile();prof.enable();generate(texts[0],32);prof.disable();prof.dump_stats(str(root/'short_cpu.prof'));buf=io.StringIO();pstats.Stats(prof,stream=buf).sort_stats('cumulative').print_stats(40);(root/'short_cpu.txt').write_text(buf.getvalue());print(buf.getvalue(),flush=True)
for repeat in range(2):
 for cap in ([64,256,1024,4096] if repeat==0 else [4096,1024,256,64]):
  mx.set_cache_limit(cap*2**20)
  for j,text in enumerate(texts):
   mx.reset_peak_memory();v=generate(text);v.update(cap_mib=cap,repeat=repeat,text_id=j);r['samples'].append(v);save();print('CACHE',repeat,cap,j,v['tok_s'],v['peak']/2**30,flush=True)
mx.set_cache_limit(64*2**20)
for depth in (1,2,3,4,6):
 engine=MlxMTPServing(m,depth)
 for j,text in enumerate(texts):
  mx.reset_peak_memory();v=generate(text);v.update(depth=depth,text_id=j);r['samples'].append(v);save();print('DEPTH',depth,j,v['tok_s'],v['stats']['verify_ct'],v['stats']['num_reuse_tokens'],flush=True)
r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()
