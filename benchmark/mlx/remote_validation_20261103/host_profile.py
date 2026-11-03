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
import cProfile,pstats,io
prompt=tok.apply_chat_template([{'role':'user','content':'Explain why the sky is blue.'}],tokenize=True,add_generation_prompt=True)
def generate(arm,profile=False):
 cache=m.make_cache();e.clear();h=e.prefill('p',mx.array([prompt],mx.int32),cache) if arm=='dflash' else m.model(mx.array([prompt],mx.int32),cache);mx.eval(h,[c.state for c in cache]);token=int(m.target_tokens(h[:,-1:]).item());raw=[]
 prof=cProfile.Profile()
 if profile:prof.enable()
 for step in range(299):
  t=time.perf_counter()
  if arm=='dflash':e.set_remaining('p',299-step);v=e.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True)
  else:v=m.forward_last_tokens(mx.array([[token]],mx.int32),cache)
  mx.eval(v,[c.state for c in cache]);token=int(v.item());raw.append(time.perf_counter()-t)
 if profile:
  prof.disable();prof.dump_stats(str(O/f'host_{arm}.prof'));buf=io.StringIO();pstats.Stats(prof,stream=buf).sort_stats('cumulative').print_stats(45);(O/f'host_{arm}.txt').write_text(buf.getvalue());print(buf.getvalue(),flush=True)
 print('HOST300',arm,profile,299/sum(raw),flush=True);e.remove('p')
for arm in ('ar','dflash'):
 generate(arm);generate(arm,True)
