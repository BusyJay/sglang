"""Same resident checkpoint: baseline, FP8, NVFP4 and combined 300-token ABBA.

Only diagnostic instance classes change between arms; production modules and
checkpoint arrays are never replaced. Each graph is recaptured for its arm.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_tokenizer

from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing

parser = argparse.ArgumentParser()
parser.add_argument('--tokens', type=int, default=300)
parser.add_argument('--modes', default='baseline,nvfp4,fp8,combined')
parser.add_argument('--rounds', type=int, default=2)
parser.add_argument('--prompts', type=int, default=3)
parser.add_argument('--output', default='benchmark/mlx/remote_validation_20261103/fullmodel_ab.json')
args = parser.parse_args()
root = Path('/Users/jay/workspace/sglang')
out = root / args.output
target = Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
draft = Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
spec = importlib.util.spec_from_file_location('sglang.srt.hardware_backend.mlx.models.glm5_baseline_ab',
                                           root / 'benchmark/mlx/remote_validation_20261103/baseline/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py')
baseline = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
recommended = mx.device_info()['max_recommended_working_set_size']
budget = int(recommended * .78)
mx.set_memory_limit(budget)
mx.set_wired_limit(budget)
mx.set_cache_limit(1024**3)
data = dict(device=mx.device_info(), checkpoint=str(target), draft=str(draft),
            tokens=args.tokens, budget=budget, samples=[], warmups=[],
            env={k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'SGLANG_MLX_'))},
            swap_before=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True))
data['source_sha256'] = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__), root/'python/sglang/srt/hardware_backend/mlx/models/glm5_next.py',
              root/'python/sglang/srt/hardware_backend/mlx/warmup.py',
              root/'python/sglang/kernels/ops/gemm/mlx_block_fp8.py',
              root/'python/sglang/kernels/ops/gemm/mlx_nvfp4_swiglu.py',
              root/'python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py']
}


def save():
    out.write_text(json.dumps(data, indent=2))


start = time.perf_counter()
model, _ = load_fp4_model(target, enable_mtp=False)
model.configure_optimizations('affine_fused')
model.configure_mtp_verification('ffn')
model.configure_latent_projection_cache(True)
model.configure_eval_groups(decode=2, prefill=4, verification=2)
model.prepare_for_inference()
mx.eval(model.parameters())
data['load_prepare_seconds'] = time.perf_counter() - start
engine = MlxDFlashServing(model, draft, 3)
tokenizer = load_tokenizer(target)
data['resident_bytes'] = mx.get_active_memory()
print('MODEL_READY', data['load_prepare_seconds'], data['resident_bytes'] / 2**30, flush=True)
save()

class AR:
    def stats(self): return {}
    def clear(self): pass
    def remove(self, rid): pass
    def set_remaining(self, rid, count): pass
    def prefill(self, rid, inputs, cache): return model.model(inputs, cache)
    def decode_batch(self, rids, inputs, caches, tokens, token_only=True):
        return model.forward_with_hidden(inputs, caches[0], token_only=token_only)[0]


engines = {'ar': AR(), 'dflash3': engine}
texts = ['Explain why the sky is blue.',
         'Write a Python function that computes the first n Fibonacci numbers, with an explanation.',
         '请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
texts = texts[:args.prompts]
prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                                       add_generation_prompt=True) for text in texts]
canonical = {}
classes = {}
checkpoint_array_ids = tuple(id(v) for _, v in __import__('mlx.utils', fromlist=['tree_flatten']).tree_flatten(model.parameters()))
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.gemm.mlx_nvfp4_swiglu import mlx_nvfp4_swiglu
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
class Reuse2(baseline.ClampedSwitchGLU):
    def __call__(self,x,indices,**kw):
        if not 1<x.shape[1]<=8:
            return g.ClampedSwitchGLU.__call__(self,x,indices,**kw)
        work=mlx_expert_pair_worklist(indices)
        def project(value,p):
            w,s,gs=p.weight,p.scales,p.weight_scale_2*2688
            count,picks=indices.size,indices.shape[-1];e,n,pk=w.shape;k=pk*2
            return nv._kernel(False)(inputs=[value.reshape(-1,k),w,s,gs.reshape(-1),indices.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',1),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,n,count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*indices.shape,1,n)
        gate,up=[project(x[...,None,None,:],p) for p in (self.gate_proj,self.up_proj)]
        act=nn.silu(mx.minimum(gate,self.activation.limit))*mx.clip(up,-self.activation.limit,self.activation.limit)
        return project(act,self.down_proj).squeeze(-2)
class FusedRows(baseline.ClampedSwitchGLU):
    def __call__(self,x,indices,**kw):
        if not 1<x.shape[1]<=8:
            return g.ClampedSwitchGLU.__call__(self,x,indices,**kw)
        gate,up=self.gate_proj,self.up_proj
        act=mlx_nvfp4_swiglu(x[...,None,None,:],gate.weight,gate.scales,gate.weight_scale_2*2688,up.weight,up.scales,up.weight_scale_2*2688,indices,limit=self.activation.limit)
        return self.down_proj(act,indices).squeeze(-2)
for mode in args.modes.split(','):
    classes[mode]=(g.MLP,{'fp8':baseline.ClampedSwitchGLU,'current':g.ClampedSwitchGLU,'reuse1':Reuse2,'fused':FusedRows}[mode])


def select(mode):
    mlp_class, switch_class = classes[mode]
    for layer in model.layers:
        blocks = (layer.mlp.shared_experts,) if isinstance(layer.mlp, g.MoE) else (layer.mlp,)
        for block in blocks:
            block.__class__ = mlp_class
        if isinstance(layer.mlp, g.MoE):
            layer.mlp.switch_mlp.__class__ = switch_class
        for name in ('_fixed_mla_layer', '_fixed_mla_attention'):
            owner = getattr(layer, name, None)
            if owner is not None:
                owner.graphs.clear()
                if hasattr(owner, '_observation'):
                    owner._observation = None
    model.model._fixed_groups.clear()
    assert tuple(id(v) for _, v in __import__('mlx.utils', fromlist=['tree_flatten']).tree_flatten(model.parameters())) == checkpoint_array_ids
    engine.clear()
    start = time.perf_counter()
    model.warmup_for_inference(verification_counts=(2, 4),
                              capture_layer_ids=engine.capture_layer_ids, token_only=True)
    data['warmups'].append(dict(mode=mode, seconds=time.perf_counter()-start))
    save()


modes = list(classes)
for round_id, order in enumerate((modes, list(reversed(modes)))[:args.rounds]):
    for mode in order:
        select(mode)
        for text_id, prompt in enumerate(prompts):
            arms = ('ar', 'dflash3') if (text_id + round_id) % 2 == 0 else ('dflash3', 'ar')
            for arm in arms:
                e = engines[arm]
                e.clear()
                cache = model.make_cache()
                before = copy.deepcopy(e.stats())
                start = time.perf_counter()
                for offset in range(0, len(prompt), 256):
                    hidden = e.prefill('p', mx.array([prompt[offset:offset+256]], mx.int32), cache)
                    mx.eval(hidden, [c.state for c in cache])
                token = int(model.target_tokens(hidden[:, -1:]).item())
                prefill = time.perf_counter() - start
                ids, raw = [token], []
                mx.reset_peak_memory()
                for step in range(args.tokens - 1):
                    e.set_remaining('p', args.tokens-step-1)
                    start = time.perf_counter()
                    result = e.decode_batch(['p'], mx.array([[token]], mx.int32), [cache], [token], token_only=True)
                    mx.eval(result, [c.state for c in cache])
                    token = int(result.item())
                    raw.append(time.perf_counter() - start)
                    ids.append(token)
                after = copy.deepcopy(e.stats())
                key = text_id
                exact = key not in canonical or ids == canonical[key]
                if key not in canonical:
                    canonical[key] = ids
                first_difference = next((i for i, (a, b) in enumerate(zip(ids, canonical[key])) if a != b), None)
                row = dict(mode=mode, round=round_id, arm=arm, text_id=text_id, input_tokens=len(prompt),
                           prefill_seconds=prefill, raw_seconds=raw, tok_s=len(raw)/sum(raw),
                           steady_tok_s=len(raw[16:])/sum(raw[16:]), tokens=ids,
                           token_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), exact=exact,
                           first_difference=first_difference,
                           stats={k: after.get(k, 0)-before.get(k, 0) for k in
                                  ('verify_ct', 'num_proposed_drafts', 'num_correct_drafts',
                                   'num_accept_tokens', 'reject_ct', 'num_reuse_tokens')},
                           correct_by_depth={k: v-before.get('correct_by_depth', {}).get(k, 0) for k, v in after.get('correct_by_depth', {}).items()},
                           eligible_by_depth={k: v-before.get('eligible_by_depth', {}).get(k, 0) for k, v in after.get('eligible_by_depth', {}).items()},
                           host_timings={k: v-before.get('timings', {}).get(k, 0) for k, v in after.get('timings', {}).items()}, peak_bytes=mx.get_peak_memory(),
                           active_bytes=mx.get_active_memory())
                data['samples'].append(row)
                data['swap_after'] = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)
                save()
                print('FULL300', round_id, mode, arm, text_id, round(row['tok_s'], 3),
                      'steady', round(row['steady_tok_s'], 3), 'exact', exact, row['stats'], flush=True)
                e.remove('p')
                del cache, hidden, result
                if not exact:
                    raise RuntimeError(f'Full model token mismatch: {mode=} {arm=} {text_id=} {first_difference=}')
data['complete'] = True
save()
