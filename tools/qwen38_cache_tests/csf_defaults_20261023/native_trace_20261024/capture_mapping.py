import hashlib, importlib.util, json, os, signal, subprocess, time, urllib.request
from pathlib import Path
from transformers import AutoTokenizer
R=Path(os.environ.get('NATIVE_TRACE_OUTPUT','/results/native_trace_20261024/mapping'))
R.mkdir(exist_ok=True,parents=True)
repo=Path('/workspace/sglang')
env=dict(os.environ)
env.update({k:v for k,v in json.loads(Path('/results/capacity-native-half-repeat/environment.json').read_text()).items() if k.startswith('SGLANG_')})
env['SGLANG_QWEN38_OBSERVE']='0'
env.update(PYTHONPATH=str(repo/'python'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
cmd=json.loads(Path('/results/production-native-config-sync-20261024/launch.json').read_text())['command']
cmd=['/opt/sglang/bin/python3','-m','sglang.launch_server']+cmd[1:]
for flag in ['--model-path','--served-model-name']:
 cmd[cmd.index(flag)+1]='/hf-cache/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300'
cmd[cmd.index('--download-dir')+1]='/hf-cache'
cmd[cmd.index('--port')+1]='30938'
cmd[cmd.index('--host')+1]='127.0.0.1'
i=cmd.index('--encoder-urls'); del cmd[i:i+2]
cmd[cmd.index('--language-only')]='--language-model-only'
for flag in ['--enable-hierarchical-cache','--hicache-ratio','--hicache-mem-layout','--hicache-io-backend','--hicache-write-policy']:
 i=cmd.index(flag); del cmd[i:i+(1 if flag=='--enable-hierarchical-cache' else 2)]
cmd=[x.replace('--cuda-graph-backend-decode=breakable','--cuda-graph-backend-decode=disabled') for x in cmd]
cmd+=['--random-seed','42']
(R/'command.json').write_text(json.dumps({'command':cmd},indent=2))
(R/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('SGLANG_','PYTORCH_','PYTHONPATH'))},indent=2))
(R/'source-manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'python/sglang').rglob('*.py')},indent=2))
spec=importlib.util.spec_from_file_location('bench',repo/'tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
tok=AutoTokenizer.from_pretrained(cmd[cmd.index('--model-path')+1],local_files_only=True)
phrase=tok.encode('Explain how an expert cache handles a miss. ',add_special_tokens=False)
def fixture(n,k):
 return [1000+((k+1)*2654435761+s*40503)%100000 for s in range(64)]+(phrase*((n+len(phrase)-1)//len(phrase)))[:n]
chat=tok.apply_chat_template([{'role':'user','content':'Explain in detail how a CPU and GPU can cooperate to run a mixture of experts language model when GPU memory is limited. Discuss cache misses, data movement, latency, and dynamic cache resizing.'}],tokenize=True,add_generation_prompt=True)
if hasattr(chat,'get'): chat=chat['input_ids']
if chat and isinstance(chat[0],list): chat=chat[0]
def post(path,data):
 req=urllib.request.Request('http://127.0.0.1:30938'+path,json.dumps(data).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=600) as resp:return resp.read().decode()
p=None
try:
 with (R/'server.log').open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+600
  while True:
   if p.poll() is not None:raise RuntimeError('server exited')
   if time.monotonic()>deadline:raise TimeoutError('startup')
   try:
    with urllib.request.urlopen('http://127.0.0.1:30938/ready',timeout=3) as resp:
     if resp.status==200:break
   except OSError:pass
   time.sleep(2)
  print('ready',flush=True)
  (R/'server_info.json').write_text(urllib.request.urlopen('http://127.0.0.1:30938/get_server_info').read().decode())
  for name,ids,count in [('warmup',fixture(1024,0),128)]:
   (R/(name+'.json')).write_text(json.dumps(bench.generate(ids,count),indent=2));print(name,'done',flush=True)
  for label,ids,count in [('short',fixture(1024,2),32)]:
   dest=R/label;dest.mkdir(exist_ok=True)
   post('/start_profile',{'output_dir':str(dest),'activities':['CPU','GPU'],'with_stack':True,'record_shapes':True,'profile_id':'native-'+label})
   result=bench.generate(ids,count) if count>1 else json.loads(post('/generate',{'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':1,'ignore_eos':True}}))
   (dest/'request.json').write_text(json.dumps(result,indent=2))
   post('/stop_profile',{});print(label,'captured',flush=True)
  time.sleep(8)
finally:
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
