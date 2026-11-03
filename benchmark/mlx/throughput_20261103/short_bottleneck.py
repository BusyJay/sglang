"""300-token resident-model baseline and synchronized projection diagnostics."""
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_fp4 as fp4
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing

root = Path('benchmark/mlx/throughput_20261103')
target = Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
draft = Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
mx.set_memory_limit(200 * 2**30)
mx.set_wired_limit(200 * 2**30)
mx.set_cache_limit(2**30)
r = {'device': mx.device_info(), 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'target': str(target), 'draft': str(draft), 'tokens': 300, 'env': {k:v for k,v in os.environ.items() if k.startswith(('MLX_', 'SGLANG_MLX'))}, 'swap_before': subprocess.check_output(['sysctl', 'vm.swapusage'], text=True), 'samples': [], 'projection_diagnostics': []}
def save():
    (root / 'profile.json').write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
start = time.perf_counter()
m, _ = fp4.load_fp4_model(target, enable_mtp=False)
m.configure_optimizations('affine_fused')
m.configure_mtp_verification('ffn')
m.configure_latent_projection_cache(True)
m.prepare_for_inference()
m.configure_eval_groups(decode=2, prefill=4, verification=2)
mx.eval(m.parameters())
e = MlxDFlashServing(m, draft, 3)
tok = load_tokenizer(target)
r['load_seconds'] = time.perf_counter() - start
r['resident_bytes'] = mx.get_active_memory()
print('RESIDENT', r['load_seconds'], r['resident_bytes']/2**30, flush=True)
texts = ['Explain why the sky is blue.', 'Write a Python function that computes the first n Fibonacci numbers, with an explanation.', '请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
def generate(text_id, arm, count=300):
    text = texts[text_id]
    e.clear()
    cache = m.make_cache()
    prompt = tok.apply_chat_template([{'role':'user', 'content':text}], tokenize=True, add_generation_prompt=True)
    before = copy.deepcopy(e.stats())
    start = time.perf_counter()
    h = e.prefill('p', mx.array([prompt], mx.int32), cache) if arm == 'dflash3' else m.model(mx.array([prompt], mx.int32), cache)
    v = m.target_tokens(h[:, -1:])
    mx.eval(v, [c.cache for c in cache])
    token = int(v.item())
    prefill = time.perf_counter() - start
    ids = [token]
    raw = []
    mx.reset_peak_memory()
    for i in range(count - 1):
        e.set_remaining('p', count-i-1)
        start = time.perf_counter()
        inputs = mx.array([[token]], mx.int32)
        v = e.decode_batch(['p'], inputs, [cache], [token], token_only=True) if arm == 'dflash3' else m.forward_last_tokens(inputs, cache)
        mx.eval(v, [c.cache for c in cache])
        token = int(v.item())
        raw.append(time.perf_counter() - start)
        ids.append(token)
    after = e.stats()
    delta = {k:after.get(k,0)-before.get(k,0) for k in ('verify_ct','num_proposed_drafts','num_accept_tokens','num_correct_drafts','reject_ct')}
    row = {'arm':arm, 'text_id':text_id, 'input_tokens':len(prompt), 'tokens':ids, 'hash':hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'raw_seconds':raw, 'prefill_seconds':prefill, 'tok_s':len(raw)/sum(raw), 'steady_tok_s':len(raw[16:])/sum(raw[16:]), 'stats':delta, 'active_bytes':mx.get_active_memory(), 'peak_bytes':mx.get_peak_memory()}
    e.remove('p')
    del cache, h, v
    mx.clear_cache()
    return row
for j in range(3):
    for arm in (('ar','dflash3') if j%2 == 0 else ('dflash3','ar')):
        row = generate(j,arm)
        r['samples'].append(row)
        save()
        print('BASE300',j,arm,round(row['tok_s'],4),round(row['steady_tok_s'],4),row['stats'],flush=True)
    pair = [v for v in r['samples'] if v['text_id'] == j]
    assert pair[0]['tokens'] == pair[1]['tokens'], ('baseline mismatch',j)

# Diagnostic only: per-projection synchronizations destroy the normal overlap.
# Timings include upstream non-projection work; do not sum as kernel GPU time.
m.configure_fixed_groups(False)
original_fp = fp4.FP8Linear.project
original_nv = fp4.NVFP4SwitchLinear.__call__
records = []
label = {id(p):n for n,p in m.named_modules()}
active = False
def instrument_fp(self,x):
    if not active:
        return original_fp(self,x)
    start=time.perf_counter();y=original_fp(self,x);mx.eval(y);dt=time.perf_counter()-start
    records.append({'name':label.get(id(self),'?'), 'kind':'mxfp8' if self.block_size==32 else 'blockfp8', 'input':list(x.shape),'dtype':str(x.dtype),'weight':list(self.weight.shape),'seconds':dt})
    return y
def instrument_nv(self,x,indices,sorted_indices=False):
    if not active:
        return original_nv(self,x,indices,sorted_indices=sorted_indices)
    start=time.perf_counter();y=original_nv(self,x,indices,sorted_indices=sorted_indices);mx.eval(y);dt=time.perf_counter()-start
    records.append({'name':label.get(id(self),'?'),'kind':'nvfp4','input':list(x.shape),'dtype':str(x.dtype),'weight':list(self.weight.shape),'picks':list(indices.shape),'seconds':dt})
    return y
fp4.FP8Linear.project=instrument_fp
fp4.NVFP4SwitchLinear.__call__=instrument_nv
for arm in ('ar','verify4'):
    for repeat in range(2):
        cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':texts[0]}],tokenize=True,add_generation_prompt=True)
        h=m.model(mx.array([p],mx.int32),cache);v=m.target_tokens(h[:,-1:]);mx.eval(v,[c.cache for c in cache]);token=int(v.item())
        records=[];active=True;start=time.perf_counter()
        if arm=='ar':v=m.forward_last_tokens(mx.array([[token]],mx.int32),cache)
        else:v=m.verify_mtp_rows(mx.array([[token]*4],mx.int32),cache,token_only=True,capture_layer_ids=e.capture_layer_ids)
        mx.eval(v);mx.synchronize();elapsed=time.perf_counter()-start;active=False
        r['projection_diagnostics'].append({'arm':arm,'repeat':repeat,'whole_seconds':elapsed,'records':records,'note':'Synchronized diagnostic with compile groups disabled; includes upstream non-projection work, not additive GPU attribution.'})
        save()
        from collections import defaultdict
        totals=defaultdict(float)
        for rec in records:totals[rec['kind']]+=rec['seconds']
        print('DIAG',arm,repeat,elapsed,dict(totals),len(records),flush=True)
        del cache,h,v
        mx.clear_cache()
fp4.FP8Linear.project=original_fp
fp4.NVFP4SwitchLinear.__call__=original_nv
r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
save()
print('PROFILE_COMPLETED',flush=True)
