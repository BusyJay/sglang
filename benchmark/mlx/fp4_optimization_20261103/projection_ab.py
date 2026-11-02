import json,time,hashlib,subprocess,collections
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_fp4 as f
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing
root=Path('benchmark/mlx/fp4_optimization_20261103');target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
mx.set_memory_limit(188*2**30);mx.set_wired_limit(188*2**30);mx.set_cache_limit(64*2**20)
r={'device':mx.device_info(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'samples':[],'micro':[],'attribution':[],'swap_before':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}
def save(): (root/'projection_ab.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
t=time.perf_counter();m,_=f.load_fp4_model(target,enable_mtp=True);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters());r['load_s']=time.perf_counter()-t;print('LOADED',r['load_s'],mx.get_active_memory()/2**30,flush=True)
tok=load_tokenizer(target);engine=MlxMTPServing(m,2)
texts=['Explain why the sky is blue.','Write a Python function that computes the first n Fibonacci numbers, with an explanation.','请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
def generate(text,count=64):
 engine.clear();cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True);h=engine.prefill('p',mx.array([p],mx.int32),cache);mx.eval(h,[c.cache for c in cache]);token=int(m.target_tokens(h[:,-1:]).item());ids=[token];times=[]
 for i in range(count-1):
  engine.set_remaining('p',count-i-1);t=time.perf_counter();v=engine.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(v,[c.cache for c in cache]);token=int(v.item());times.append(time.perf_counter()-t);ids.append(token)
 row={'prompt':p,'ids':ids,'raw_s':times,'tok_s':len(times)/sum(times),'stats':engine.stats(),'active':mx.get_active_memory(),'peak':mx.get_peak_memory()};engine.remove('p');return row
import ast,importlib,types,copy
base=Path('benchmark/mlx/fp4_optimization_20261103/source_baseline')
oldop=types.ModuleType('old_fp8');exec(compile((base/'python/sglang/kernels/ops/gemm/mlx_fp8_rows.py').read_text(),'old_fp8','exec'),oldop.__dict__)
tree=ast.parse((base/'python/sglang/srt/hardware_backend/mlx/models/glm5_fp4.py').read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mxfp8_project_rows');namespace={'mx':mx};exec(compile(ast.Module(body=[func],type_ignores=[]),'old_mxfp8','exec'),namespace)
newmx=f.mxfp8_project_rows;newblock=f.mlx_fp8_rows
# Graphs trace projection choices; reset diagnostic owners between execution plans.
def reset_graphs():
 seen=set()
 def visit(v):
  if id(v) in seen or isinstance(v,mx.array):return
  seen.add(id(v))
  if isinstance(v,dict):
   for x in list(v.values()):visit(x)
  elif isinstance(v,(list,tuple)):
   for x in v:visit(x)
  elif type(v).__module__.endswith('glm5_next') or hasattr(v,'named_modules'):
   if hasattr(v,'graphs') and isinstance(v.graphs,dict):v.graphs.clear()
   for x in list(vars(v).values()):visit(x)
 visit(m)
for repeat in range(2):
 for j,text in enumerate(texts):
  order=['baseline','head','block','both']
  if (j+repeat)%2:order.reverse()
  for arm in order:
   f.mxfp8_project_rows=namespace['mxfp8_project_rows'] if arm in ('baseline','block') else newmx
   f.mlx_fp8_rows=oldop.mlx_fp8_rows if arm in ('baseline','head') else newblock
   reset_graphs();before=copy.deepcopy(engine.stats());v=generate(text);after=engine.stats();v['stats_delta']={k:after.get(k,0)-before.get(k,0) for k in ('verify_ct','num_proposed_drafts','num_reuse_tokens')};v.update(text_id=j,repeat=repeat,arm=arm,order=order);r['samples'].append(v);save();print('AB',j,repeat,arm,v['tok_s'],v['stats_delta'],flush=True)
f.mxfp8_project_rows=newmx;f.mlx_fp8_rows=newblock
for j in range(3):
 rows=[v for v in r['samples'] if v['text_id']==j];ref=rows[0]['ids'];print('TOKENS',j,all(v['ids']==ref for v in rows),flush=True)
r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()
