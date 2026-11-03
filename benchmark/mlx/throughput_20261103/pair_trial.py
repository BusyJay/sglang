exec(open('benchmark/mlx/throughput_20261103/short_bottleneck.py').read().split('for j in range(3):')[0])
import importlib,statistics
from nv_exact_pair import pair
from sglang.srt.hardware_backend.mlx.models import glm5_next as glm
nvorig=glm.ClampedSwitchGLU.__call__
records=[]
labels={id(p):name for name,p in m.named_modules()}
captured={}
def capture(self,x,indices):
    if isinstance(self.gate_proj,fp4.NVFP4SwitchLinear):captured.setdefault((tuple(self.gate_proj.weight.shape),str(x.dtype)),(self,x,indices))
    return nvorig(self,x,indices)
glm.ClampedSwitchGLU.__call__=capture
m.configure_fixed_groups(False)
for count in (1,4):
    captured.clear();cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':texts[0]}],tokenize=True,add_generation_prompt=True)
    h=m.model(mx.array([p],mx.int32),cache);v=m.target_tokens(h[:,-1:]);mx.eval(v,[c.cache for c in cache]);token=int(v.item());captured.clear()
    v=m.forward_last_tokens(mx.array([[token]],mx.int32),cache) if count==1 else m.verify_mtp_rows(mx.array([[token]*count],mx.int32),cache,token_only=True,capture_layer_ids=e.capture_layer_ids)
    mx.eval(v)
    for key,(sw,x,ids) in captured.items():
        x=mx.contiguous(x);ids=mx.contiguous(ids);mx.eval(x,ids)
        g,u=sw.gate_proj,sw.up_proj
        def base():return g(x[...,None,None,:],ids),u(x[...,None,None,:],ids)
        variants={'native':base}
        for rps in (1,2,4,8):variants[str(rps)]=lambda rps=rps:pair(x,g,u,ids,rps)
        a=base();mx.eval(a);errors={};trials=[]
        for name,fn in variants.items():
            b=fn();mx.eval(b)
            errors[name]=[{'exact':mx.array_equal(v,w).item(),'max':mx.max(mx.abs(v.astype(mx.float32)-w.astype(mx.float32))).item()} for v,w in zip(a,b)]
        for j in range(7):
            ts={}
            for name in list(variants) if j%2==0 else list(reversed(variants)):
                fn=variants[name];mx.eval(fn());t=time.perf_counter()
                for _ in range(15):mx.eval(fn())
                ts[name]=(time.perf_counter()-t)/15
            trials.append(ts)
        row={'count':count,'key':key,'errors':errors,'trials':trials};records.append(row)
        print('PAIR',count,key,errors,{k:statistics.median(t[k] for t in trials)*1e3 for k in variants},flush=True)
    captured.clear();del cache,h,v;mx.clear_cache()
(root/'pair_trial.json').write_text(json.dumps(records,indent=2))
glm.ClampedSwitchGLU.__call__=nvorig
# Full-model ABBA only if actual inputs are exact; retain native prefill.
best={v['count']:min((k for k in v['errors'] if k!='native'),key=lambda k:statistics.median(t[k] for t in v['trials']),default=None) for v in records}
assert best.get(1) and best.get(4), best
# Diagnostic only: measure generation impact instead of discarding float differences.
# Adoption depends on routing/tokens/counters validation and arithmetic localization.
def candidate(self,x,indices):
    if isinstance(self.gate_proj,fp4.NVFP4SwitchLinear) and 1<=x.shape[1]<=4:
        g,u=pair(x,self.gate_proj,self.up_proj,indices,int(best[1 if x.shape[1]==1 else 4]));act=self.activation(u,g)
        return self.down_proj(act,indices).squeeze(-2)
    return nvorig(self,x,indices)
samples=[]
for mode in ('baseline','pair','pair','baseline'):
    glm.ClampedSwitchGLU.__call__=candidate if mode=='pair' else nvorig
    m.configure_fixed_groups(True)
    for j in range(3):
        for arm in ('ar','dflash3') if j%2==0 else ('dflash3','ar'):
            row=generate(j,arm);row['mode']=mode;samples.append(row)
            (root/'pair_ab.json').write_text(json.dumps(samples,indent=2))
            print('PAIR300',mode,j,arm,row['tok_s'],row['stats'],flush=True)
    m.configure_fixed_groups(False)
glm.ClampedSwitchGLU.__call__=nvorig

for j in range(3):
 ss=[v for v in samples if v['text_id']==j]
 print('PAIR_TOKENS',j,all(v['tokens']==ss[0]['tokens'] for v in ss),flush=True)
print('PAIR_AB_COMPLETED',flush=True)
