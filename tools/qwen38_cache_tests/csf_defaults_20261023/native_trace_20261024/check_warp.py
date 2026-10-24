import json,statistics
from pathlib import Path
import torch
from sglang.kernels.jit.utils import load_jit
from sglang.srt.layers.moe.qwen38_cache_buffers import AdaptiveMetadataBuffers
R=Path('/results/native_trace_20261024')
fields=('counts','lookup','keys','frequency','ages','generation','clock','plan')
mods={name:load_jit('qwen38_warp_check_'+name,cuda_files=[str(R/('qwen38_expert_cache.'+name+'.cu'))],extra_include_paths=['/workspace/sglang/python/sglang/kernels/jit/csrc/moe'],cuda_wrappers=[('decode','Qwen38CacheDecodePlan::run_parallel')]) for name in ['before','warp']}
def state(slots):
 b=AdaptiveMetadataBuffers(1,512,slots,torch.device('cuda'));s=b.reset(0,slots)
 b.restore(0,s,[(e,float(e%5),e+1) for e in range(slots)]);s['plan'].zero_();s['counts'].zero_();return s
cases={
 'decode10':list(range(10)),
 'single':[511],
 'hit':list(range(30)),
 'mixed':[0,1,2,3,4,5,6,7,8,511]*3,
 'miss':list(range(511,481,-1)),
 'duplicate':[0,511,-1]*10,
 'masked':[-1]*32,
 'lane31':list(range(511,479,-1)),
}
torch.manual_seed(42)
matched=0
for slots in [32,147]:
 for policy in [0,1]:
  a,b=state(slots),state(slots)
  for row in list(cases.values())+[torch.randint(-1,512,(30,)).tolist() for _ in range(100)]:
   ids=torch.tensor([row],dtype=torch.int32,device='cuda');outs=[]
   for name,s in [('before',a),('warp',b)]:
    out=torch.empty_like(ids);mods[name].decode(ids,out,*(s[k] for k in fields),.95,policy,.7,0,0,2);outs.append(out)
   torch.testing.assert_close(*outs,rtol=0,atol=0)
   for k in fields:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0,msg=lambda message,k=k:f'{k}: {message}')
   matched+=1
print('exact_state_comparisons',matched,flush=True)
report={}
for label,row in cases.items():
 ids=torch.tensor([row],dtype=torch.int32,device='cuda');source=state(147)
 per={}
 for name,mod in mods.items():
  s={k:v.clone() for k,v in source.items()};out=torch.empty_like(ids)
  def run():
   for k in fields:s[k].copy_(source[k])
   mod.decode(ids,out,*(s[k] for k in fields),.95,0,.7,0,0,2)
  run();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(8):run()
  samples=[]
  for _ in range(9):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(40):g.replay()
   b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/320)
  per[name]={'reset_and_plan_us':statistics.median(samples),'samples':samples}
 report[label]=per;print(label,per,flush=True)
(R/'warp-check.json').write_text(json.dumps({'exact_state_comparisons':matched,'timings':report},indent=2))
