"""Replay a locally captured expert call against expanded native scales."""
import json
import struct
import sys
from pathlib import Path
import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import decode
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import load_manifest
from sglang.srt.model_loader.qwen38_nvfp4_pack.tiered import fixed_bytes
from sglang.kernels.ops.moe.qwen38_expert_cache import csf_patch_headers

path = Path(sys.argv[1])
d = torch.load(path, weights_only=True)
x, ids, router = (d[k].cuda() for k in ('x', 'ids', 'router'))
t = d['tensors']
capacity = t['w13_weight'].shape[0]
specs = d['specs']
manifest = load_manifest(Path('/root/.cache/sglang/qwen38-fp4-local-csf-tiered/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json'))
cc = {k:v.cuda() for k,v in t.items()}
nn = dict(cc)
active = torch.unique(d['ids'].clamp_min(0)).tolist()
banks = []
with manifest.payload_path.open('rb') as source:
    for name, spec in specs.items():
        rows, cols = spec['rows'], spec['columns']
        expanded = torch.zeros(capacity, rows, cols, dtype=torch.uint8)
        for slot in active:
            head = bytearray(t[name][slot].numpy().tobytes())
            off = struct.unpack_from('<Q', head, 976)[0]
            count = struct.unpack_from('<I', head, 1008)[0]
            head[976:1008] = bytes(32)
            head[1016:1024] = bytes(8)
            if off:
                source.seek(off)
                head = head[:fixed_bytes(rows, cols)] + source.read(count*4) + bytes((-count*4)%16)
            expanded[slot] = torch.frombuffer(bytearray(decode(head,rows,cols)),dtype=torch.uint8).reshape(rows,cols)
        nn[name] = expanded.cuda().view(torch.float8_e4m3fn)
        rg = manifest.scale_encoding[name]['overflow'][str(d['layer'])]
        if rg['nbytes']:
            source.seek(rg['offset'])
            bank = torch.frombuffer(bytearray(source.read(rg['nbytes'])), dtype=torch.uint8).cuda()
            banks.append(bank)
            csf_patch_headers(cc[name], bank.data_ptr()-rg['offset'],rg['offset'],rg['offset']+rg['nbytes'])
H, I = x.shape[1], t['w2_weight'].shape[2]*2
a = CachedB12xW4A16(x.device,H,I,capacity)
b = CachedB12xW4A16(x.device,H,I,capacity,scale_encoding=specs)

def stats(y):
    return {'nonfinite_per_row':(~torch.isfinite(y)).sum(-1).cpu().tolist(), 'finite_max': torch.nan_to_num(y.float()).abs().max().item()}
results = {'captured':stats(d['output']), 'input':stats(d['x']), 'capacity':capacity,'ids':d['ids'].tolist(), 'router':d['router'].tolist()}
for name, call in [
    ('csf_poison',lambda:(b.intermediate.fill_(float('nan')),b.run(x,ids,router,cc))[1]),
    ('native',lambda:a.run(x,ids,router,nn)),
    ('csf',lambda:b.run(x,ids,router,cc)),
    ('triton',lambda:cached_w4a16_moe(x,ids,router,nn)),
    ('csf_split',lambda:torch.cat([b.run(x[i:i+1],ids[i:i+1].clone(),router[i:i+1].clone(),cc) for i in range(3)])),
]:
    y = call()
    results[name]=stats(y)
print(json.dumps(results,indent=2))
path.with_suffix('.replay.json').write_text(json.dumps(results,indent=2))

if __import__('os').getenv('ROOT_BENCH') == '1':
    import statistics
    timing = {}
    for name, adapter, tensors in [('native',a,nn),('csf',b,cc)]:
        adapter.done = torch.cuda.Event()
        adapter.used = False
        for _ in range(3):
            adapter.run(x,ids,router,tensors)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        adapter.used = False
        with torch.cuda.graph(graph):
            for _ in range(50):
                out = adapter.run(x,ids,router,tensors)
        graph.replay()
        times = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end)*1000/5000)
        timing[name]={'microseconds':statistics.median(times),'samples':times}
    path.with_suffix('.timing.json').write_text(json.dumps(timing,indent=2))
    print('TIMING',json.dumps(timing),flush=True)
