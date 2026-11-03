"""Replay actual projection inputs; no upstream work inside timing."""
exec(open('benchmark/mlx/throughput_20261103/short_bottleneck.py').read().split('for j in range(3):')[0])
import importlib
import statistics
import types
fp = importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows')
mxop = importlib.import_module('sglang.kernels.ops.gemm.mlx_mxfp8_rows')
source = Path(fp.__file__).read_text()
start=source.index('_DECODE = r"""');end=source.index('\n_SOURCE =',start)
replacement='''_DECODE = r"""
inline float e4m3(uchar b) {
    ushort v=b&127;
    if(v==127)return NAN;
    ushort u=(v<<7)|(((v+1)>>7)<<14)|(ushort(b>>7)<<15);
    return float(as_type<half>(u)*256.0);
}
"""
'''
ns={};exec(source[:start]+replacement+source[end:],ns)
half_fn=ns['mlx_fp8_rows']
# Test pack-load rewrite; keep one original FP32 multiplication/add per K lane.
vecsrc=source.replace('uint lane = thread_index_in_simdgroup;', 'uint lane = thread_index_in_simdgroup;')
# Actual captures retained once per shape/dtype, no large bank copies.
captures={}
orig=fp4.FP8Linear.project
nv_orig=fp4.NVFP4SwitchLinear.__call__
labels={id(p):name for name,p in m.named_modules()}
def capture(self,x):
    captures.setdefault(('fp',self.block_size,tuple(self.weight.shape),str(x.dtype)),(self,x))
    return orig(self,x)
def capture_nv(self,x,indices,sorted_indices=False):
    captures.setdefault(('nv',tuple(self.weight.shape),str(x.dtype)),(self,x,indices))
    return nv_orig(self,x,indices,sorted_indices=sorted_indices)
fp4.FP8Linear.project=capture;fp4.NVFP4SwitchLinear.__call__=capture_nv
m.configure_fixed_groups(False)
results=[]
for count in (1,4):
    captures.clear();cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':texts[0]}],tokenize=True,add_generation_prompt=True)
    h=m.model(mx.array([p],mx.int32),cache);v=m.target_tokens(h[:,-1:]);mx.eval(v,[c.cache for c in cache]);token=int(v.item());captures.clear()
    if count==1:v=m.forward_last_tokens(mx.array([[token]],mx.int32),cache)
    else:v=m.verify_mtp_rows(mx.array([[token]*count],mx.int32),cache,token_only=True,capture_layer_ids=e.capture_layer_ids)
    mx.eval(v, [z[1] for z in captures.values()]);mx.synchronize()
    for key,entry in list(captures.items()):
        proj,x,*extra=entry;x=mx.contiguous(x);mx.eval(x)
        if key[0]=='nv':
            ids=mx.contiguous(extra[0]);mx.eval(ids)
            variants={'native':lambda: nv_orig(proj,x,ids)}
        elif proj.block_size==128:
            q=fp.mlx_fp8_dynamic_32(x);mx.eval(q)
            variants={'native':lambda:fp.mlx_fp8_rows(x,proj.weight,proj.scales,block_size=128,quantized_input=q),'half_decode':lambda:half_fn(x,proj.weight,proj.scales,block_size=128,quantized_input=q)}
        else:
            variants={'native':lambda:mxop.mlx_mxfp8_rows(x,proj.weight,proj.scales)}
        ref=variants['native']();mx.eval(ref);trials=[];errors={}
        for name,fn in variants.items():
            y=fn();mx.eval(y)
            errors[name]={'exact':mx.array_equal(y,ref).item(),'max_error':mx.max(mx.abs(y.astype(mx.float32)-ref.astype(mx.float32))).item()}
        for trial in range(7):
            times={}
            for name in (list(variants) if trial%2==0 else list(reversed(variants))):
                fn=variants[name];mx.eval(fn());t=time.perf_counter()
                for _ in range(16):mx.eval(fn())
                times[name]=(time.perf_counter()-t)/16
            trials.append(times)
        row={'count':count,'name':labels.get(id(proj),'?'),'key':key,'trials':trials,'errors':errors}
        results.append(row);(root/'operator_replay.json').write_text(json.dumps(results,indent=2))
        print('REPLAY',count,key,{name:round(statistics.median(t[name] for t in trials)*1e3,4) for name in variants},errors,flush=True)
    captures.clear();del cache,h,v;mx.clear_cache()
fp4.FP8Linear.project=orig;fp4.NVFP4SwitchLinear.__call__=nv_orig
print('REPLAY_COMPLETED',flush=True)
