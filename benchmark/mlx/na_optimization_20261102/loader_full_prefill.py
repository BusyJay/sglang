import hashlib,json,time,statistics,subprocess
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model,load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
import sglang.kernels.ops.gemm.mlx_qmm_swiglu as q
OUT=Path("benchmark/mlx/na_optimization_20261102")
target=Path("/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e")
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"]);mx.set_cache_limit(1024**3)
m,_=load_model(target,lazy=True,get_model_classes=lambda config:(lambda args:g.Model(args,enable_mtp=False),g.ModelArgs));m.configure_optimizations("affine_fused");m.configure_mtp_verification("ffn");m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters())
header=Path("python/sglang/kernels/ops/gemm/mlx_qmm_swiglu_nax.metal").read_text()
old="""    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          StridedPackedBytes{src, packed_stride, byte_offset + (bj + i) * bytes_per_pack}, scale, bias, dst + i * pack_factor);
    }"""
new="""    if(packed_stride==1 && bits==5){
      for(int i=0;i<n_reads;++i)
        dequantize<T,pack_factor,bits>(src+byte_offset+(bj+i)*bytes_per_pack,scale,bias,dst+i*pack_factor);
    }else{
"""+old+"""
    }"""
assert header.count(old)==1;header=header.replace(old,new)
kernel=mx.fast.metal_kernel(name="sglang_nax_contiguous_loader_full",input_names=["x","work","work_count","gate_w","gate_s","gate_b","up_w","up_s","up_b","limit"],output_names=["out"],source=q._NAX_SOURCE,header=header,ensure_row_contiguous=False)
original=q._nax_kernel
tok=load_tokenizer(target)
r=dict(device=mx.device_info(),checkpoint=str(target),head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),raw=[],swap_before=subprocess.check_output(["sysctl","vm.swapusage"],text=True))
def cache_arrays(cache):return [a for c in cache for a in c.cache if isinstance(a,mx.array)]
for count in (256,1024):
 text="Explain how a hash table resolves collisions, with a Python example."
 p=tok.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True)
 p=tok.encode("Background: "+("Machine learning uses data. "*count))[:count-len(p)]+p
 inputs=mx.array([p],mx.int32)
 ref=None
 for group in range(4):
  for mode in ("A","B","B","A"):
   q._nax_kernel=original if mode=="A" else lambda:kernel
   cache=m.make_cache();mx.reset_peak_memory();start=time.perf_counter()
   h=m.model(inputs,cache);mx.eval(h,cache_arrays(cache));elapsed=time.perf_counter()-start
   if ref is None:ref=(h,[a.view(a.dtype) for a in cache_arrays(cache)])
   assert mx.array_equal(h,ref[0]).item(),(count,group,mode,"hidden")
   assert all(mx.array_equal(a,b).item() for a,b in zip(cache_arrays(cache),ref[1])),(count,group,mode,"cache")
   row=dict(count=count,group=group,mode=mode,seconds=elapsed,peak_bytes=mx.get_peak_memory(),active_bytes=mx.get_active_memory(),exact=True)
   r["raw"].append(row);(OUT/"loader_full_prefill.json").write_text(json.dumps(r,indent=2));print(row,flush=True)
   del h,cache
 del ref;mx.synchronize();mx.clear_cache()
q._nax_kernel=original
r["swap_after"]=subprocess.check_output(["sysctl","vm.swapusage"],text=True);(OUT/"loader_full_prefill.json").write_text(json.dumps(r,indent=2))
