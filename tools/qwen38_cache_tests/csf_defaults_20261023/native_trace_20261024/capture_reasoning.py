import hashlib, importlib.util, json, os, signal, subprocess, time, urllib.request
from pathlib import Path
from transformers import AutoTokenizer
R=Path(os.environ.get('NATIVE_TRACE_OUTPUT','/results/native_trace_20261024/reasoning-baseline'))
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
  import sys
  sys.path.insert(0,'/results/native_trace_20261024')
  import reasoning_workload as rw
  (R/'prompts.json').write_text(json.dumps(rw.PROMPTS,ensure_ascii=False,indent=2))
  ids_list=[tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=True) for text in rw.PROMPTS]
  ids_list=[ids['input_ids'] if hasattr(ids,'get') else ids for ids in ids_list]
  for repeat in range(3):
   for index,ids in enumerate(ids_list):
    result=rw.generate(ids,2048)
    (R/f'case-{index}-repeat-{repeat}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('reasoning',index,repeat,result['output_tokens'],result['after_128_tokens_per_second'],flush=True)
  dest=R/'steady';dest.mkdir(exist_ok=True)
  result=rw.generate(ids_list[0],2048,profile=str(dest))
  (dest/'request.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
  print('reasoning steady captured',flush=True)
  time.sleep(8)
finally:
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
