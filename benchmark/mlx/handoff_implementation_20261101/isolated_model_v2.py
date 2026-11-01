import argparse,hashlib,json,os,subprocess,time,copy
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--arm',required=True);parser.add_argument('--group',type=int,required=True)
parser.add_argument('--output',required=True);args=parser.parse_args()
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing,clone_cache
import sglang.srt.hardware_backend.mlx.models.glm5_next as G
checkpoint=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e')
result=dict(arm=args.arm,group=args.group,source=str(Path(G.__file__).resolve()),device=mx.device_info(),checkpoint=str(checkpoint),env={k:v for k,v in os.environ.items() if k.startswith('MLX_')},swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True),samples=[])
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
start=time.perf_counter();model=load_mtp_model(checkpoint);result['load_seconds']=time.perf_counter()-start
result['runtime_configuration']=[dict(layer=i,defer_hc=getattr(layer,'defer_hc',False),fuse_hc=layer.fuse_hc,preserve_residual_dtype=layer.preserve_residual_dtype,bc_gate_mode=getattr(layer.mlp,'bc_gate_mode',None),bc_down_mode=getattr(layer.mlp,'bc_down_mode',None)) for i,layer in enumerate(model.layers)]
for prompt in (32,2050):
 base=model.make_cache();e=MlxMTPServing(model,2)
 start=time.perf_counter()
 for pos in range(0,prompt,256):
  h=e.prefill('base',(mx.arange(pos,min(pos+256,prompt))%128)[None],base);mx.eval(h)
 root=int(mx.argmax(model.lm_head(h[:,-1:]),-1).item());e.remove('base')
 prefill=time.perf_counter()-start
 for steps in (0,2):
  for repeat in range(2):
   cache=[clone_cache(c) for c in base];cache[0].mlx_mtp=base[0].mlx_mtp
   engine=MlxMTPServing(model,steps);token=root;tokens=[];raw=[];cold_stats=None
   mx.reset_peak_memory()
   for j in range(96):
    start=time.perf_counter();out=engine.decode_batch(['probe'],mx.array([[token]]),[cache],[token],token_only=True)
    mx.eval(out,[c.cache for c in cache]);token=int(out.item());raw.append(time.perf_counter()-start);tokens.append(token)
    if j == 4: cold_stats=copy.deepcopy(engine.stats())
   engine.remove('probe')
   row=dict(prompt=prompt,steps=steps,repeat=repeat,raw=raw,cold=raw[:5],tok_s=91/sum(raw[5:]),all_tok_s=96/sum(raw),prefill_seconds=prefill,tokens=tokens,hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),stats=engine.stats(),cold_stats=cold_stats,cache_memory=mx.get_cache_memory(),active=mx.get_active_memory(),peak=mx.get_peak_memory())
   result['samples'].append(row)
   Path(args.output).write_text(json.dumps(result,indent=2));print(args.arm,args.group,prompt,steps,repeat,round(row['tok_s'],3),flush=True)
   del cache,engine,out;mx.clear_cache()
 del base,e,h;mx.clear_cache()
result['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
Path(args.output).write_text(json.dumps(result,indent=2))
