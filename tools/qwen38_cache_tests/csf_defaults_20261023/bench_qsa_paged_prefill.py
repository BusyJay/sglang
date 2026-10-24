"""CUDA-event timing of packing-inclusive and direct-paged QSA prefill APIs."""
import json
import statistics
from pathlib import Path

import torch
from triton.backends.nvidia.compiler import get_ptxas
from sglang.srt.layers.attention.qsa.sparse_attn import (
    pack_qsa_prefill_kv, sparse_gqa_fwd_interface_triton_ck,
)

torch.manual_seed(173)
tokens, rows, topk = 219007, 8192, 2051
q = (torch.randn(rows,24,256,device='cuda') * .05).bfloat16()
k,v = [(torch.randn(tokens,2,256,device='cuda') * .05).to(torch.float8_e4m3fn)
       for _ in range(2)]
table = torch.randperm(tokens,device='cuda',dtype=torch.int32).reshape(1,-1)
requests = torch.zeros(1,device='cuda',dtype=torch.int32)
cu_k = torch.tensor([0,tokens],device='cuda',dtype=torch.int32)
cu_q = torch.tensor([0,rows],device='cuda',dtype=torch.int32)
lens = torch.tensor([tokens],device='cuda',dtype=torch.int32)
indices = torch.randint(tokens-rows,(rows,topk),device='cuda',dtype=torch.int32)
indices[::2,-3:] = -1

def pack():
    return pack_qsa_prefill_kv(k=k,v=v,req_to_token=table,req_indices=requests,
        cu_k=cu_k,total_k=tokens,max_k=tokens,output_dtype=q.dtype)

packed_k, packed_v = pack()
def run(mode):
    keys,values,extra = packed_k,packed_v,{}
    if mode == 'packing_and_attention':
        keys,values = pack()
    elif mode == 'paged_fp8':
        keys,values = k,v
        extra = dict(req_to_token=table,req_indices=requests)
    return sparse_gqa_fwd_interface_triton_ck(
        q,keys,values,indices,cu_q,cu_k,lens,256**-.5,**extra)

modes = ['packing_and_attention','packed_attention_only','paged_fp8']
reference = run('packing_and_attention')
for mode in modes:
    actual = run(mode)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
torch.cuda.synchronize()
times = {mode:[] for mode in modes}
for iteration in range(7):
    for mode in (modes if iteration%2==0 else modes[::-1]):
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        result = run(mode)
        end.record()
        end.synchronize()
        times[mode].append(start.elapsed_time(end))
capability = torch.cuda.get_device_capability(q.device)
assembler = get_ptxas(capability[0] * 10 + capability[1])
report = dict(tokens=tokens,queries=rows,topk=topk,
    assembler=dict(path=assembler.path,version=assembler.version),
    note='Synthetic uniform historical selections; shuffled physical slots; CUDA-event API timings include allocation and host dispatch gaps, no inter-run L2 flush.',
    samples_ms=times,median_ms={m:statistics.median(t) for m,t in times.items()})
Path('/results/qsa-paged-performance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
