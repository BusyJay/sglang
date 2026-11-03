"""M5 diagnostics: actual activations/routes, full banks, per-layer FFN ABBA.

Capture uses the eager target for observability; synchronized per-layer timings
are attribution, not serving throughput. No checkpoint arrays are changed.
"""
import copy, hashlib, importlib.util, json, os, statistics, subprocess, sys, time
from collections import Counter
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import mlx_nvfp4_gather_rows
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
R=Path('/Users/jay/workspace/sglang');O=R/'benchmark/mlx/remote_validation_20261103'
target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
s=importlib.util.spec_from_file_location('sglang.srt.hardware_backend.mlx.models.replay_baseline',O/'baseline/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
budget=int(mx.device_info()['max_recommended_working_set_size']*.78)
mx.set_memory_limit(budget);mx.set_wired_limit(budget);mx.set_cache_limit(1024**3)
data=dict(device=mx.device_info(),note=__doc__,cases=[],projection_cases=[],swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True))
def save(): (O/'rootcause_replay.json').write_text(json.dumps(data,indent=2))
t=time.perf_counter();m,_=load_fp4_model(target,enable_mtp=False);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.configure_eval_groups(decode=2,prefill=4,verification=2);m.prepare_for_inference();mx.eval(m.parameters())
e=MlxDFlashServing(m,draft,3);tok=load_tokenizer(target);print('REPLAY_READY',time.perf_counter()-t,flush=True)
inputs=mx.load(O/'actual_ffn_operands.npz')
from sglang.kernels.ops.gemm.mlx_nvfp4_swiglu import mlx_nvfp4_swiglu
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
modes=('native','reuse','fused')
functions={mode:[] for mode in modes}
for i,l in enumerate(m.layers):
    if not isinstance(l.mlp,g.MoE):continue
    sw=l.mlp.switch_mlp
    x=inputs[f'c4_l{i}_x'];ids=inputs[f'c4_l{i}_ids']
    def ffn(value,route,sw=sw,mode='native'):
        work=mlx_expert_pair_worklist(route) if mode=='reuse' else None
        def project(value,p):
            if mode=='reuse':return nv.mlx_nvfp4_gather_rows(value,p.weight,p.scales,p.weight_scale_2*2688,route,worklist=work)
            return p(value,route)
        if mode=='fused':
            g,u=sw.gate_proj,sw.up_proj
            act=mlx_nvfp4_swiglu(value[...,None,None,:],g.weight,g.scales,g.weight_scale_2*2688,u.weight,u.scales,u.weight_scale_2*2688,route,limit=sw.activation.limit)
        else:
            g,u=[project(value[...,None,None,:],p) for p in (sw.gate_proj,sw.up_proj)]
            act=sw.activation(u,g)
        return project(act,sw.down_proj).squeeze(-2)
    ref=None
    for mode in modes:
        fn=mx.compile(lambda value,route,mode=mode,ffn=ffn:ffn(value,route,mode=mode))
        y=fn(x,ids);mx.eval(y)
        if ref is None:ref=y
        else:assert mx.array_equal(ref,y).item(),(i,mode)
        functions[mode].append((fn,x,ids))
print('SWEEP_READY',mx.get_active_memory()/2**30,flush=True)
data=dict(device=mx.device_info(),scope='42 actual routed FFNs, one full-weight sweep per step, dynamic ids; not full model throughput',samples=[])
for repeat in range(2):
    raw={mode:[] for mode in modes}
    for block in range(30):
        for mode in modes if (block+repeat)%2==0 else reversed(modes):
            for j in range(10):
                bump=mx.array((block*10+j)*1e-8,mx.float32)
                mx.eval(bump)
                t=time.perf_counter();ys=[]
                for fn,x,ids in functions[mode]:
                    y=fn(x+bump,ids);ys.append(y);mx.async_eval(y)
                mx.eval(ys);raw[mode].append(time.perf_counter()-t)
    for mode in modes:
        row=dict(round=repeat,mode=mode,raw_seconds=raw[mode],mean_ms=statistics.mean(raw[mode])*1000)
        data['samples'].append(row);print('FULL_FFN_SWEEP300',repeat,mode,row['mean_ms'],flush=True)
    (O/'full_ffn_sweep.json').write_text(json.dumps(data,indent=2))
data['complete']=True;data['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
(O/'full_ffn_sweep.json').write_text(json.dumps(data,indent=2))
