import json, statistics, time
from pathlib import Path
import torch
from sglang.kernels.ops.gemm.hc_mix import _hc_mix_persistent_kernel
R=Path('/results/native_trace_20261024')
torch.manual_seed(42)
# Eight independent weight pairs exceed L2, rather than repeatedly timing one hot layer.
pairs=[(torch.randn(320,10240,device='cuda',dtype=torch.bfloat16)*.02,torch.randn(10240,320,device='cuda',dtype=torch.bfloat16)*.02) for _ in range(8)]
counters=torch.zeros(3,dtype=torch.int32,device='cuda')
sm=torch.cuda.get_device_properties(0).multi_processor_count
res=[]
for rows in [1,3]:
 x=torch.randn(rows,10240,device='cuda',dtype=torch.bfloat16);scratch=torch.empty(16,320,device='cuda');outs=[torch.empty(rows,2560,device='cuda',dtype=torch.bfloat16) for _ in pairs]
 ref=[]
 for down,up in pairs:
  t=torch.nn.functional.silu(x.float()@down.float().T/4).to(torch.bfloat16)
  gate=torch.sigmoid(t.float()@up.float().T).reshape(rows,4,2560)
  ref.append((gate*x.float().reshape(rows,4,2560)).mean(1))
 for bn,bk,warps in [(32,256,8),(32,256,4),(64,256,8),(32,512,8),(64,512,8),(16,256,4)]:
  for ctas in [sm,128,80,64,40]:
   if ctas>sm:continue
   try:
    def launch():
     for (down,up),out in zip(pairs,outs):
      _hc_mix_persistent_kernel[(ctas,)](x,down,up,scratch,out,counters,10240,320,2560,rows,ctas,.25,ROWS=16,HC=4,BLOCK_N=bn,BLOCK_K=bk,BLOCK_J=32,BLOCK_R=64,num_warps=warps)
    launch();torch.cuda.synchronize()
    for out,target in zip(outs,ref):torch.testing.assert_close(out.float(),target,rtol=.01,atol=.005)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):launch()
    ds=[]
    for _ in range(5):
     a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
     for i in range(20):g.replay()
     b.record();b.synchronize();ds.append(a.elapsed_time(b)*1000/160)
    item=dict(rows=rows,bn=bn,bk=bk,warps=warps,ctas=ctas,us=statistics.median(ds),samples=ds)
    res.append(item);print(json.dumps(item),flush=True)
    (R/'hc-tuning.json').write_text(json.dumps(res,indent=2))
   except Exception as e:
    print('FAILED',rows,bn,bk,warps,ctas,str(e),flush=True)
