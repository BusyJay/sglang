"""Locate BF16 rounding at identical inputs before interpreting end-to-end drift."""
import numpy as np, torch, mlx.core as mx, mlx.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from sglang.srt.hardware_backend.mlx.models.glm5_vision import load_glm5_vision
root=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e')
m=load_glm5_vision(root);mx.random.seed(29);results=[]
def cpu(a):return torch.from_numpy(np.array(a.astype(mx.float32))).bfloat16()
def check(name,a,b):
 mx.eval(a);a=np.array(a.astype(mx.float32));b=b.float().numpy();diff=a-b
 r=dict(name=name,rel=float(np.linalg.norm(diff)/np.linalg.norm(b)),max=float(np.max(np.abs(diff))),equal=float(np.mean(a==b)));print(r,flush=True);results.append(r)
x=mx.random.normal((64,1024)).astype(mx.bfloat16);t=cpu(x)
check('silu',nn.silu(x),F.silu(t));check('silu_fp32',nn.silu(x.astype(mx.float32)).astype(x.dtype),F.silu(t))
check('gelu',nn.gelu(x),F.gelu(t));check('gelu_fp32',nn.gelu(x.astype(mx.float32)).astype(x.dtype),F.gelu(t))
for name,l in [('norm',m.blocks[0].norm1),('qkv',m.blocks[0].attn.qkv),('mlp_gate',m.blocks[0].mlp.gate_proj)]:
 if name=='norm':
  tf=t.float();out=(tf*torch.rsqrt(tf.square().mean(-1,keepdim=True)+l.eps)).bfloat16()*cpu(l.weight)
 else:out=F.linear(t,cpu(l.weight),cpu(l.bias))
 check(name,l(x),out)
from transformers.models.glm5_next.modeling_glm5_next import apply_rotary_pos_emb_vision
import inspect
print(inspect.getsource(apply_rotary_pos_emb_vision))
Path('benchmark/mlx/vision_20261102/primitive_parity.json').write_text(json.dumps(results,indent=2)+'\n')
