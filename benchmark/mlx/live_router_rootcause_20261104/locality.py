"""M5 physical expert-bank locality A/B with lossless block permutations.

LFU uses one archived c4 and c1 route per layer. This is a small training sample,
not request-wide LFU. Oracle-hot packs this same test route and is an upper-bound
locality diagnostic. Router arithmetic stays in original expert-ID order.
Both original and permuted live arms run the same GPU ID remap; no CPU route sync.
"""
import argparse,hashlib,json,statistics,time,gc
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
ROOT=Path('/Users/jay/workspace/sglang');OUT=ROOT/'benchmark/mlx/live_router_rootcause_20261104'
CACHE=Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
TARGET=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,default=26);p.add_argument('--steps',type=int,default=300);args=p.parse_args()
mx.set_memory_limit(12*2**30);mx.set_cache_limit(256*2**20)
a=SimpleNamespace(**json.loads((TARGET/'config.json').read_text())['text_config'])
arrays=mx.load(CACHE/f'layer-{args.layer:02}.safetensors');bank=[]
for name in ('gate_proj','up_proj','down_proj'):
 pre='mlp.switch_mlp.'+name+'.';bank.append(tuple(arrays[pre+n] for n in ('weight','scales','weight_scale_2')))
shared=g.MLP(a,a.moe_intermediate_size*a.n_shared_experts)
for name in ('gate_proj','up_proj','down_proj'):
 pre='mlp.shared_experts.'+name+'.';setattr(shared,name,FP8Linear(arrays[pre+'weight'],arrays[pre+'scales'],block_size=128,activation_mode='dynamic32'))
router=arrays['mlp.gate.weight'].astype(mx.float32);bias=arrays['mlp.e_score_correction_bias'];mx.eval(bank,shared.parameters(),router,bias)
operands=mx.load(ROOT/'benchmark/mlx/remote_validation_20261103/actual_ffn_operands.npz');x=operands[f'c4_l{args.layer}_x'];archived=operands[f'c4_l{args.layer}_ids'];ar=operands[f'c1_l{args.layer}_ids'];del arrays,operands
E=bank[0][0].shape[0]
def route(value):
 values=value.astype(mx.float32).reshape(-1,value.shape[-1]);count=values.shape[0]
 if count<8:values=mx.concatenate([values,mx.zeros((8-count,values.shape[1]),mx.float32)],0)
 scores=(values @ router.T)[:count].reshape(*value.shape[:-1],E)
 return g._group_expert_select(scores,bias,a.num_experts_per_tok,a.n_group,a.topk_group,a.routed_scaling_factor,a.norm_topk_prob,a.scoring_func)
ids,weights=route(x);mx.eval(ids,weights);assert mx.array_equal(ids,archived).item()
train=Counter(ids.reshape(-1).tolist()+ar.reshape(-1).tolist())
lfu=sorted(range(E),key=lambda i:(-train[i],i))
hot=list(dict.fromkeys(ids.reshape(-1).tolist()));oracle=hot+[i for i in range(E) if i not in hot]
import random
random_order=list(range(E));random.Random(1234).shuffle(random_order)
layouts={'lfu':lfu,'oracle_coaccess':oracle,'reverse':list(reversed(range(E))),'random':random_order}
identity=mx.arange(E,dtype=mx.uint32)
data=dict(scope=__doc__,device=mx.device_info(),layer=args.layer,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),train_counts=dict(train),original_ids=ids.tolist(),checks=[],samples=[],layouts={},memory=[])
def project(value,params,selected,mode,work):
 w,s,gs=params;k=w.shape[-1]*2;n=w.shape[-2];count=selected.size;picks=selected.shape[-1]
 if mode=='native':return mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=selected,mode='nvfp4',group_size=16,bits=4,global_scale=gs*2688)
 return nv._kernel(False)(inputs=[value.reshape(-1,k),w,s,(gs*2688).reshape(-1),selected.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',1),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,n,count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*selected.shape,1,n)
def forward(value,selected,selected_weights,params,mapping,mode,scope):
 if scope!='captured':
  selected,selected_weights=route(value)
  if mode=='dependency':value=mx.depends(value,selected_weights)
  selected=mx.take(mapping,selected)
 shared_y=shared(value,batch_fusion=True)
 work=mlx_expert_pair_worklist(selected) if mode!='native' else None
 v=value[...,None,None,:];up=project(v,params[1],selected,mode,work);gate=project(v,params[0],selected,mode,work)
 act=nn.silu(mx.minimum(gate,a.swiglu_limit))*mx.clip(up,-a.swiglu_limit,a.swiglu_limit)
 down=project(act,params[2],selected,mode,work).squeeze(-2)
 return (down*selected_weights[...,None]).sum(-2).astype(value.dtype)+shared_y
specs=[(mode,scope) for mode in ['native','reuse1','dependency'] for scope in ['captured','live']]
# Runtime bank parameters prevent closure/address-specialized graph differences.
fns={spec:mx.compile(lambda value,selected,selected_weights,params,mapping,spec=spec:forward(value,selected,selected_weights,params,mapping,*spec)) for spec in specs}
for layout,order in layouts.items():
 inv=[0]*E
 for new,old in enumerate(order):inv[old]=new
 perm=mx.array(order,mx.uint32);mapping=mx.array(inv,mx.uint32)
 start=time.perf_counter();other=[tuple(mx.take(v,perm,axis=0) for v in triple) for triple in bank];mapped=mx.take(mapping,ids);mx.eval(other,mapped)
 setup=time.perf_counter()-start
 data['layouts'][layout]=dict(order=order,inverse=inv,repack_seconds=setup)
 # exact inverse restoration on every stored weight/scale/global byte
 for old,new in zip(bank,other):
  for before,after in zip(old,new):assert mx.array_equal(mx.take(after,mapping,axis=0),before).item()
 data['memory'].append(dict(layout=layout,active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory()))
 for spec,fn in fns.items():
  ref=fn(x,ids,weights,bank,identity);y=fn(x,mapped,weights,other,mapping);mx.eval(ref,y);exact=mx.array_equal(ref,y).item();assert exact,(layout,spec)
  data['checks'].append(dict(layout=layout,spec=spec,exact=exact,weight_bytes_restored=True))
  for _ in range(4):mx.eval(fn(x,ids,weights,bank,identity),fn(x,mapped,weights,other,mapping))
 raw={(spec,side):[] for spec in specs for side in ['original','permuted']}
 for block in range((args.steps+9)//10):
  sequence=specs if block%2==0 else list(reversed(specs))
  for spec in sequence:
   for side in ['original','permuted'] if block%2==0 else ['permuted','original']:
    params,selected,lookup=(bank,ids,identity) if side=='original' else (other,mapped,mapping)
    for j in range(min(10,args.steps-block*10)):
     value=x+mx.array((block*10+j)*1e-8,mx.float32);mx.eval(value);start=time.perf_counter();y=fns[spec](value,selected,weights,params,lookup);mx.eval(y);raw[(spec,side)].append(time.perf_counter()-start)
 for (spec,side),seconds in raw.items():
  row=dict(layout=layout,spec=spec,side=side,raw_seconds=seconds,mean_ms=statistics.mean(seconds)*1000);data['samples'].append(row)
  print('LOCALITY300',args.layer,layout,spec,side,row['mean_ms'],flush=True)
 (OUT/f'locality_{args.layer}.json').write_text(json.dumps(data,indent=2))
 del other,params,selected,lookup,new,after,y,ref,mapped;gc.collect();mx.clear_cache()
data['complete']=True;(OUT/f'locality_{args.layer}.json').write_text(json.dumps(data,indent=2))
