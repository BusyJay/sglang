import argparse,sys,os,subprocess,time,json,concurrent.futures,statistics,hashlib
from pathlib import Path
ROOT=Path('/Users/jay/workspace/sglang');OUT=ROOT/'benchmark/mlx/review_optimization_20261101/mechanism_revision'
sys.path.insert(0,str(ROOT/'benchmark/mlx'))
from bench_glm5_next_mtp_http import panel,contracts
import urllib.request
def request(base,route,payload=None,*,stream=False):
 req=urllib.request.Request(base+route,data=json.dumps(payload).encode() if payload is not None else None,headers={"Content-Type":"application/json"})
 started=time.perf_counter()
 with urllib.request.urlopen(req,timeout=1200) as response:
  if not stream:
   data=response.read();return data.decode() if route in ("/health","/flush_cache") else json.loads(data)
  first=last=None;previous=first_count=0;gaps=[];event=None
  for line in response:
   if not line.startswith(b"data:") or line[5:].strip()==b"[DONE]":continue
   event=json.loads(line[5:])
   if "error" in event:raise RuntimeError(event["error"])
   count=event["meta_info"]["completion_tokens"]
   if count>previous:
    now=time.perf_counter()
    if first is None:first,first_count=now,count
    else:gaps.append({"seconds":now-last,"tokens":count-previous})
    last,previous=now,count
  if first is None:raise RuntimeError("no stream tokens")
  return event,dict(elapsed_s=time.perf_counter()-started,ttft_s=first-started,decode_tokens_per_s=(previous-first_count)/(last-first) if last>first else 0,raw_gaps=gaps,completion_tokens=previous,cached_tokens=event["meta_info"].get("cached_tokens",0))
from sglang.test.test_utils import terminate_and_kill_process_tree
MODEL='/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('baseline','candidate'),required=True);p.add_argument('--group',type=int,default=0);p.add_argument('--repeat',type=int,default=1);p.add_argument('--steps',default='0,2');p.add_argument('--resources',action='store_true');p.add_argument('--port',type=int,default=30002);args=p.parse_args()
checkout=OUT/'baseline' if args.mode=='baseline' else ROOT
report=dict(mode=args.mode,group=args.group,cases=[],contracts={},source={})
import hashlib
for path in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','python/sglang/srt/hardware_backend/mlx','python/sglang/kernels/ops'],cwd=ROOT,text=True).splitlines():
 file=checkout/path
 if file.exists():report['source'][path]=hashlib.sha256(file.read_bytes()).hexdigest()
for steps in map(int,args.steps.split(',')):
 port=args.port;base=f'http://127.0.0.1:{port}'
 env=dict(os.environ,PYTHONPATH=str(checkout/'python'),SGLANG_USE_MLX='1',MLX_ENABLE_TF32='0')
 command=[sys.executable,'-u','-m','sglang.launch_server','--model-path',MODEL,'--host','127.0.0.1','--port',str(port),'--context-length','32768','--max-total-tokens','65536','--chunked-prefill-size','256','--max-running-requests','4','--mlx-enable-sampling','--mlx-mtp-num-steps',str(steps),'--random-seed','1234','--enable-deterministic-inference','--disable-cuda-graph']
 log=(OUT/f'http_g{args.group}_{args.mode}_step{steps}.log').open('w');proc=subprocess.Popen(command,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for i in range(300):
   if proc.poll() is not None:raise RuntimeError('server exited')
   try:request(base,'/health');break
   except Exception:time.sleep(1)
  else:raise RuntimeError('server load timeout')
  prompts={str(n):[10,11,12,13]*(n//4)+[10]*(n%4) for n in (26,2050,8192,32700)}
  for length in (2050,):
   for concurrency in (1,2):
    for repeat in range(args.repeat):
     request(base,'/flush_cache')
     payload=dict(input_ids=prompts[str(length)],sampling_params=dict(temperature=0,max_new_tokens=64,ignore_eos=True),stream=True)
     start=time.perf_counter()
     with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:results=list(pool.map(lambda j:request(base,'/generate',payload,stream=True),range(concurrency)))
     elapsed=time.perf_counter()-start
     row=dict(steps=steps,length=length,concurrency=concurrency,repeat=repeat,elapsed=elapsed,aggregate_tokens_per_s=sum(metrics["completion_tokens"] for _,metrics in results)/elapsed,requests=[dict(panel(event),**metrics) for event,metrics in results]);report['cases'].append(row)
     (OUT/f'http_g{args.group}_{args.mode}.json').write_text(json.dumps(report,indent=2));print(steps,length,concurrency,repeat,elapsed,flush=True)
  request(base,'/flush_cache')
  cold=request(base,'/generate',dict(input_ids=[10,11,12,13]*80,sampling_params=dict(temperature=0,max_new_tokens=16,ignore_eos=True),return_logprob=True,top_logprobs_num=3))
  warm=request(base,'/generate',dict(input_ids=[10,11,12,13]*80,sampling_params=dict(temperature=0,max_new_tokens=16,ignore_eos=True),return_logprob=True,top_logprobs_num=3))
  assert panel(cold)['tokens_sha256']==panel(warm)['tokens_sha256']
  assert warm['meta_info']['cached_tokens']>0
  report['contracts'][str(steps)]=dict(cold=panel(cold),prefix=panel(warm))
  # Long context resource/lifecycle separately, bounded chunk replay.
  if args.resources:
   payload=dict(input_ids=prompts['32700'],sampling_params=dict(temperature=0,max_new_tokens=32,ignore_eos=True),stream=True)
   request(base,'/flush_cache');event,metrics=request(base,'/generate',payload,stream=True)
   report['cases'].append(dict(steps=steps,length=32700,concurrency=1,resource_validation=True,requests=[dict(panel(event),**metrics)]))
   # Keep the server's bounded 65K token pool. Concurrent clients can queue;
   # report cached/admitted behavior rather than claiming four active32K slots.
   for concurrency in (2,4):
    request(base,'/flush_cache')
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
     results=list(pool.map(lambda j:request(base,'/generate',payload,stream=True),range(concurrency)))
    elapsed=time.perf_counter()-started
    report['cases'].append(dict(steps=steps,length=32700,concurrency=concurrency,
        resource_validation=True,token_pool=65536,elapsed=elapsed,
        aggregate_tokens_per_s=sum(metrics['completion_tokens'] for _,metrics in results)/elapsed,
        requests=[dict(panel(event),**metrics) for event,metrics in results]))
   report['contracts'][str(steps)]=contracts(base,{'english':prompts['26'],'chinese':prompts['2050'],'code':prompts['26']})
  report['swap']=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip()
  (OUT/f'http_g{args.group}_{args.mode}.json').write_text(json.dumps(report,indent=2))
 finally:terminate_and_kill_process_tree(proc);log.close()
