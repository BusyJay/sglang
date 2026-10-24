"""Isolate FP8 attention rounding and retain compiled intermediate representations."""
import inspect
import json
import os
import textwrap
from pathlib import Path

import torch
import sglang.srt.layers.attention.qsa.sparse_attn as op

root = Path(os.getenv('QSA_DIAG_OUTPUT', '/results/qsa-fp8-diagnostic'))
root.mkdir(exist_ok=True)
compiled = []
kernel = op._sparse_gqa_chunk_prefill

class Record:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            result = kernel[grid](*args, **kwargs)
            compiled.append(result)
            return result
        return launch

namespace = dict(op.sparse_gqa_fwd_interface_triton_ck.__globals__)
namespace['_sparse_gqa_chunk_prefill'] = Record()
source = textwrap.dedent(inspect.getsource(op.sparse_gqa_fwd_interface_triton_ck))
exec(compile(source, '<diagnostic-fp8-attention>', 'exec'), namespace)
attention = namespace['sparse_gqa_fwd_interface_triton_ck']

torch.manual_seed(173)
lengths, queries = [1031, 517], [17, 9]
slots = sum(lengths) + 17
k, v = [(torch.randn(slots, 2, 264, device='cuda') * .05)
        .to(torch.float8_e4m3fn)[..., :256] for _ in range(2)]
table = torch.zeros(4, 1044, device='cuda', dtype=torch.int32)
req = torch.tensor([3, 1], device='cuda', dtype=torch.int32)
physical = torch.randperm(slots, device='cuda', dtype=torch.int32)
table[3, :1031], table[1, :517] = physical[:1031], physical[1031:1548]
cu_k = torch.tensor([0, 1031, 1548], device='cuda', dtype=torch.int32)
cu_q = torch.tensor([0, 17, 26], device='cuda', dtype=torch.int32)
lens = torch.tensor(lengths, device='cuda', dtype=torch.int32)
q = (torch.randn(26, 24, 256, device='cuda') * .05).bfloat16()
visible = torch.cat([torch.arange(n-t+1,n+1,device='cuda') for n,t in zip(lengths,queries)])
indices = (torch.rand(26,2048,device='cuda') * visible[:,None]).int()
indices[:,-7:] = -1

def pack(dtype):
    return op.pack_qsa_prefill_kv(k=k,v=v,req_to_token=table,req_indices=req,
        cu_k=cu_k,total_k=1548,max_k=1031,output_dtype=dtype)

bf_k, bf_v = pack(torch.bfloat16)
fp_k, fp_v = pack(torch.float8_e4m3fn)
def run(name, keys, values, paged):
    kwargs = dict(req_to_token=table, req_indices=req) if paged else {}
    output = attention(q,keys,values,indices,cu_q,cu_k,lens,256**-.5,**kwargs)
    torch.cuda.synchronize()
    for extension in ('ttir','ttgir','llir','ptx','sass'):
        try:
            (root/(name+'.'+extension)).write_text(compiled[-1].asm[extension])
        except (KeyError, RuntimeError):
            pass
    return output

results = {
    'packed_bf16': run('packed_bf16',bf_k,bf_v,False),
    'paged_bf16': run('paged_bf16',k.bfloat16(),v.bfloat16(),True),
    'packed_fp8': run('packed_fp8',fp_k,fp_v,False),
    'paged_fp8': run('paged_fp8',k,v,True),
}
report = {'conversion_exact': bool(torch.equal(bf_k,fp_k.bfloat16()) and torch.equal(bf_v,fp_v.bfloat16()))}
for name, value in results.items():
    expected = results['packed_bf16']
    mismatch = (value != expected).nonzero()
    report[name] = dict(count=len(mismatch), max_abs=float((value-expected).abs().max()),
        positions=mismatch.tolist(), values=value[value!=expected].float().tolist(),
        expected=expected[value!=expected].float().tolist())
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
