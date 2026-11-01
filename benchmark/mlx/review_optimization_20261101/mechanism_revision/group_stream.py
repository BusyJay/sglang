import os,sys,json,time,hashlib,importlib.util,cProfile,pstats,io,subprocess,signal
from pathlib import Path
import mlx.core as mx
OUT=Path(__file__).parent
name='sglang.srt.hardware_backend.mlx.models.glm5_next'
spec=importlib.util.spec_from_file_location(name,OUT/'group_stream_model.py');G=importlib.util.module_from_spec(spec);sys.modules[name]=G;spec.loader.exec_module(G)
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing,clone_cache
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
model=load_mtp_model(Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'))
base=model.make_cache();pre=MlxMTPServing(model,2)
for start in range(0,2050,256):
 h=pre.prefill('base',(mx.arange(start,min(start+256,2050))%128)[None],base);mx.eval(h)
root=mx.argmax(model.lm_head(h[:,-1:]),-1);mx.eval(root);pre.remove('base');side=mx.new_stream(mx.gpu)
rows=[]
def run(arm,steps,count=48):
 model.configure_shared_stream(side if arm=='side' else None)
 cache=[clone_cache(c) for c in base];cache[0].mlx_mtp=base[0].mlx_mtp;engine=MlxMTPServing(model,steps);token=int(root.item());raw=[];tokens=[]
 for _ in range(count):
  t=time.perf_counter();out=engine.decode_batch(['probe'],mx.array([[token]]),[cache],[token],token_only=True);mx.eval(out,[c.cache for c in cache]);token=int(out.item());raw.append(time.perf_counter()-t);tokens.append(token)
 state=hashlib.sha256()
 for c in cache:
  for v in c.cache:
   if v is not None: state.update(memoryview(v.view(mx.uint8)).tobytes())
 result=dict(arm=arm,steps=steps,tok_s=(count-5)/sum(raw[5:]),all_tok_s=count/sum(raw),raw=raw,hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),state_hash=state.hexdigest(),stats=engine.stats())
 engine.remove('probe');return result
for steps in (0,2):
 for repeat in range(2):
  for arm in ('single','side','side','single'):
   row=run(arm,steps);row['repeat']=repeat;rows.append(row);print(steps,arm,round(row['tok_s'],2),flush=True);(OUT/'group_stream.json').write_text(json.dumps(rows,indent=2))
 for arm in ('single','side'):
  pr=cProfile.Profile();pr.enable();run(arm,steps,8);pr.disable();buf=io.StringIO();pstats.Stats(pr,stream=buf).sort_stats('tottime').print_stats(18);(OUT/f'group_host_{steps}_{arm}.txt').write_text(buf.getvalue())
# A short capture, independent from throughput. Workload labels delimit arms.
log=(OUT/'group_trace.log').open('w')
cmd=['xcrun','xctrace','record','--template','/Users/jay/workspace/sglang/benchmark/mlx/commit_profile_20261101/profile/Shader Timeline.tracetemplate','--attach',str(os.getpid()),'--time-limit','20s','--output',str(OUT/'group_stream.trace'),'--no-prompt']
proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer'))
deadline=time.monotonic()+20
while proc.poll() is None and time.monotonic()<deadline:
 if 'Attaching to:' in (OUT/'group_trace.log').read_text():break
 time.sleep(.2)
windows=[]
if proc.poll() is None:
 for arm in ('single','side'):
  start=time.time_ns();r=run(arm,2,16);windows.append(dict(arm=arm,start_epoch_ns=start,end_epoch_ns=time.time_ns()))
 proc.send_signal(signal.SIGINT)
try:rc=proc.wait(timeout=45)
except subprocess.TimeoutExpired:proc.terminate();rc=proc.wait(timeout=10)
log.close();(OUT/'group_trace_windows.json').write_text(json.dumps(dict(pid=os.getpid(),rc=rc,windows=windows),indent=2))
print('TRACE',rc,flush=True)
