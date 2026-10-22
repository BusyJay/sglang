"""Same-source GPU/LFU serving profiles; timing under profiler is diagnostic only."""
import importlib.util,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path('/results');REPO=Path('/workspace/sglang')
spec=importlib.util.spec_from_file_location('bench',REPO/'tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
base=json.loads((ROOT/'base-command.json').read_text())['command']
env=dict(os.environ,SGLANG_ENABLE_QWEN38_CPU_HYBRID='0',SGLANG_QWEN38_CPU_CORES='2,3,4,5',SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH='1',SGLANG_QWEN38_FILL_CTA='adaptive',SGLANG_QWEN38_STARTUP_CACHE='auto',SGLANG_QWEN38_SOURCE_CACHE_DIR='/dev/shm/sglang-qwen38-startup',SGLANG_QWEN38_HOST_TOKEN_EMBEDDING='1',SGLANG_QWEN38_BORROW_PREFILL_ARENA='1',SGLANG_QWEN38_PLE_PACKED_CACHE='1',SGLANG_QWEN38_CACHE_W4A16='1',SGLANG_QWEN38_CACHE_W4A16_BACKEND='b12x',SGLANG_QWEN38_CACHE_PREFILL_W4A16='0',SGLANG_QWEN38_FP32_NORM_FUSION='1',SGLANG_QWEN38_CACHE_W4A16_SPLIT_K='1')
for name,mode in [('gpu','off'),('lfu','layered_lfu')]:
 out=ROOT/('profile-'+name);out.mkdir(exist_ok=True);command=list(base);i=command.index('--model-loader-extra-config')+1;cfg=json.loads(command[i]);cfg['cpu_hybrid_mode']=mode;cfg['stats_log_interval']=19200
 if mode=='layered_lfu':cfg['layered_cpu_max_experts']=1
 command[i]=json.dumps(cfg);(out/'command.json').write_text(json.dumps(dict(command=command),indent=2))
 process=None
 try:
  with (out/'server.log').open('w') as log:
   process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
   while True:
    if process.poll() is not None:raise RuntimeError(f'server exited {process.returncode}')
    if time.monotonic()-start>600:raise TimeoutError('startup')
    try:
     with urllib.request.urlopen('http://127.0.0.1:30938/ready',timeout=3) as r:
      if r.status==200:break
    except OSError:pass
    time.sleep(2)
   # Match the production cache budget with a warm short prompt before capture.
   warm=bench.generate([1001]*1088,128);(out/'warmup.json').write_text(json.dumps(warm,indent=2))
   args=['python3','/profile-skill/scripts/analyze_llm_torch_profile.py','--framework','sglang','--url','http://127.0.0.1:30938','--output-dir',str(out/'trace'),'--num-steps','24','--warmup-steps','10','--profile-workload','decode','--decode-input-len','1088','--decode-output-len','512','--profile-prefix',name]
   with (out/'triage.txt').open('w') as report:subprocess.run(args,env=env,stdout=report,stderr=subprocess.STDOUT,check=True)
   print(name,'captured',flush=True)
 finally:
  if process is not None and process.poll() is None:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
