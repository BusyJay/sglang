"""Independent PyTorch CPU oracle for real checkpoint vision equations."""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
from PIL import Image
from transformers import AutoImageProcessor
from sglang.srt.hardware_backend.mlx.models.glm5_vision import load_glm5_vision
root=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e')
torch.set_num_threads(8)
m=load_glm5_vision(root);cfg=m.cfg
# Extract native weights once; reference uses independent conv2d/conv3d and SDPA.
from mlx.utils import tree_flatten
w={k:torch.from_numpy(np.array(v.astype(mx.float32))) for k,v in tree_flatten(m.parameters())}
def linear(x,p):return F.linear(x,w[p+'.weight'],w.get(p+'.bias'))
def rms(x,p,eps):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w[p+'.weight']
def mlp(x,p):
 g=linear(x,p+'.gate_proj').clamp(max=cfg['swiglu_limit']);u=linear(x,p+'.up_proj').clamp(-cfg['swiglu_limit'],cfg['swiglu_limit']);return linear(F.silu(g)*u,p+'.down_proj')
def oracle(pixels,grid):
 x=torch.from_numpy(pixels).float().reshape(-1,3,2,14,14)
 x=F.conv3d(x,w['patch_embed.weight'].reshape(1024,3,2,14,14),w['patch_embed.bias']).reshape(-1,1024)
 coordinates=[];lengths=[]
 for t,h,width in grid.tolist():
  for _ in range(t):
   coordinates.extend([(r+dr,c+dc) for r in range(0,h,2) for c in range(0,width,2) for dr in range(2) for dc in range(2)])
   lengths.append(h*width)
 pos=torch.tensor(coordinates)
 freq=1/(10000**(torch.arange(0,32,2).float()/32));angle=(pos[:,:,None]*freq).reshape(-1,32);angle=torch.cat([angle,angle],-1)
 cos,sin=angle.cos()[:,None],angle.sin()[:,None]
 intermediates={}
 for i in range(24):
  p=f'blocks.{i}';h=rms(x,p+'.norm1',1e-5);q,k,v=linear(h,p+'.attn.qkv').reshape(-1,3,16,64).unbind(1)
  q=rms(q,p+'.attn.q_norm',cfg['rms_norm_eps']);k=rms(k,p+'.attn.k_norm',cfg['rms_norm_eps'])
  rot=lambda a:a*cos+torch.cat([-a[...,32:],a[...,:32]],-1)*sin
  q,k=rot(q),rot(k);outputs=[];start=0
  for n in lengths:
   qs,ks,vs=[a[start:start+n].transpose(0,1)[None] for a in (q,k,v)]
   outputs.append(F.scaled_dot_product_attention(qs,ks,vs)[0].transpose(0,1).reshape(n,-1));start+=n
  x=x+linear(torch.cat(outputs),p+'.attn.proj');x=x+mlp(rms(x,p+'.norm2',1e-5),p+'.mlp')
  intermediates[i]=x
 x=rms(x,'post_layernorm',1e-5).reshape(-1,2,2,1024).permute(0,3,1,2)
 x=F.conv2d(x,w['downsample.weight'].reshape(4096,1024,2,2),w['downsample.bias']).reshape(-1,4096)
 x=linear(x,'merger.proj');x=F.layer_norm(x,(4096,),w['merger.post_projection_norm.weight'],w['merger.post_projection_norm.bias'],1e-5)
 return mlp(F.gelu(x),'merger')
p=AutoImageProcessor.from_pretrained(root)
results=[]
for sizes in [[(112,112)],[(112,168)],[(112,112),(168,112)]]:
 images=[Image.fromarray(np.random.default_rng(i+19).integers(0,256,(h,width,3),dtype=np.uint8)) for i,(h,width) in enumerate(sizes)]
 data=p(images=images,return_tensors='np');start=time.perf_counter();actual=m(mx.array(data['pixel_values']),data['image_grid_thw']);mx.eval(actual);seconds=time.perf_counter()-start
 expected=oracle(data['pixel_values'],data['image_grid_thw']).numpy();got=np.array(actual.astype(mx.float32))
 saved=tree_flatten(m.parameters());m.load_weights([(k,v.astype(mx.float32)) for k,v in saved]);fp32=m(mx.array(data['pixel_values']),data['image_grid_thw']);mx.eval(fp32);fp32_error=float(np.linalg.norm(np.array(fp32)-expected)/np.linalg.norm(expected));assert fp32_error<1e-4,fp32_error;m.load_weights(saved)
 diff=got-expected;rel=float(np.linalg.norm(diff)/np.linalg.norm(expected));cosine=float((got*expected).sum()/np.sqrt((got*got).sum()*(expected*expected).sum()))
 row=dict(sizes=sizes,grid=data['image_grid_thw'].tolist(),shape=list(got.shape),relative_l2=rel,fp32_relative_l2=fp32_error,cosine=cosine,max_abs=float(np.abs(diff).max()),seconds=seconds)
 results.append(row);print(row,flush=True)
 # This initial diagnostic measures checkpoint BF16 against FP32 oracle; no
 # production acceptance threshold inferred solely from these measurements.
Path('benchmark/mlx/vision_20261102/encoder_parity.json').write_text(json.dumps(results,indent=2)+'\n')
