"""Real sequential HTTP/cache/sampling and concurrency contracts and timings."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time,urllib.request,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['ar','mtp','dflash'],required=True);p.add_argument('--count',type=int,default=7);p.add_argument('--output',required=True);p.add_argument('--tokens',type=int,default=128);a=p.parse_args()
root=Path('/Users/jay/workspace/sglang');out=root/a.output;out.parent.mkdir(parents=True,exist_ok=True)
model=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e');draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
port=30007
base=f'http://127.0.0.1:{port}'
def req(route,body=None):
 r=urllib.request.Request(base+route,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=600) as f:
  raw=f.read()
 return json.loads(raw) if raw and route not in ['/health','/flush_cache','/abort_request'] else raw.decode()
def body(prompt,limit=32,**kw):return dict(rid='df-'+uuid.uuid4().hex,input_ids=prompt,sampling_params=dict(temperature=0,max_new_tokens=limit,ignore_eos=True),**kw)
command=[sys.executable,'-u','-m','sglang.launch_server','--model-path',str(model),'--host','127.0.0.1','--port',str(port),'--context-length','8192','--max-total-tokens','17024','--chunked-prefill-size','256','--max-running-requests','2','--mlx-enable-sampling','--random-seed','1234','--enable-deterministic-inference','--disable-cuda-graph','--disable-overlap-schedule']
if a.mode=='mtp':command+=['--mlx-mtp-num-steps','2']
elif a.mode=='dflash':command+=['--speculative-algorithm','DFLASH','--speculative-draft-model-path',str(draft),'--speculative-dflash-block-size',str(a.count+1)]
r=dict(mode=a.mode,count=a.count,command=command,cases=[],swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True))
def save():out.write_text(json.dumps(r,indent=2)+'\n')
log=out.with_suffix('.server.log').open('w');env=dict(os.environ,PYTHONPATH=str(root/'python'),SGLANG_USE_MLX='1',MLX_ENABLE_TF32='0')
# Refuse any pre-existing listener before starting a resident full model.
import socket
with socket.socket() as guard:
 guard.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 guard.bind(('127.0.0.1',port))
proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);r['server_pid']=proc.pid;started=time.perf_counter();save()
try:
 for _ in range(300):
  if proc.poll() is not None:raise RuntimeError('server exited')
  try:req('/health');break
  except Exception:time.sleep(1)
 else:raise TimeoutError('health')
 r['startup_seconds']=time.perf_counter()-started;r['server_info']=req('/get_server_info');r['rss_at_health']=subprocess.check_output(['ps','-axo','pid,ppid,rss,command'],text=True);save();print('HEALTH',r['startup_seconds'],flush=True)
 def signature(event):return dict(tokens=event['output_ids'],logprobs=event['meta_info'].get('output_token_logprobs'),top=event['meta_info'].get('output_top_logprobs'),finish=event['meta_info'].get('finish_reason'))
 def request_case(label,payload):
  start=time.perf_counter();event=req('/generate',payload);elapsed=time.perf_counter()-start
  if 'error' in event:raise RuntimeError(event)
  assert event['meta_info']['completion_tokens']==len(event['output_ids'])
  return dict(label=label,elapsed_seconds=elapsed,event=event,signature=signature(event))
 # Natural prompts for single request speed and aggregate concurrent throughput.
 from mlx_lm.utils import load_tokenizer
 tokenizer=load_tokenizer(model)
 prompts=[tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True) for text in ['Explain why the sky is blue.','Write a Python function that computes Fibonacci numbers.','请解释过拟合和解决办法。']]
 for trial in range(7):
  req('/flush_cache')
  start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   events=list(pool.map(lambda prompt:request_case('concurrent',body(prompt,a.tokens)),prompts[:2]))
  row=dict(trial=trial,elapsed_seconds=time.perf_counter()-start,events=events)
  r['cases'].append(row);save();print('CONCURRENT',trial,row['elapsed_seconds'],flush=True)
 r['complete']=True;save()
finally:
 from sglang.test.test_utils import terminate_and_kill_process_tree
 terminate_and_kill_process_tree(proc);log.close();r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);r['server_exit_code']=proc.poll();save()
