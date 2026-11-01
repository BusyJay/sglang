"""Capture layer-boundary drift using installed checkpoint reference."""
exec(open('benchmark/mlx/vision_20261102/hf_vision_parity.py').read().split('for sizes in ')[0])
from mlx.utils import tree_flatten
import inspect
from transformers.models.glm5_next import modeling_glm5_next as hf
print(inspect.getsource(hf.Glm5NextVisionRotaryEmbedding),flush=True)
captured={}
for name,module in reference.named_modules():
 if name in ['patch_embed','blocks.0.norm1','blocks.0.attn.qkv','blocks.0.attn.q_norm','blocks.0.attn.k_norm','blocks.0.attn','blocks.0.norm2','blocks.0.mlp','blocks.0','blocks.1','blocks.5','blocks.23','post_layernorm','downsample','merger']:
  module.register_forward_hook(lambda mod,args,out,n=name:captured.update({n:out.detach().float().numpy()}))
images=[Image.fromarray(np.random.default_rng(19).integers(0,256,(112,112,3),dtype=np.uint8))]
data=processor(images=images,return_tensors='pt')
with torch.inference_mode():reference(data['pixel_values'].bfloat16(),data['image_grid_thw'])
# Run the same sequence, inspect actual errors and differences at identical inputs.
rows=[]
def check(name,a):
 mx.eval(a);a=np.array(a.astype(mx.float32));b=captured[name];diff=a-b
 row={'name':name,'rel':float(np.linalg.norm(diff)/np.linalg.norm(b)),'max':float(np.abs(diff).max())};print(row,flush=True);rows.append(row)
grid=data['image_grid_thw'].numpy();pixels=mx.array(data['pixel_values'].numpy()).astype(mx.bfloat16)
x=model.patch_embed(pixels);check('patch_embed',x)
h,w=map(int,grid[0,1:]);merge=2;r,c=np.indices((h,w));order=lambda a:a.reshape(h//2,2,w//2,2).transpose(0,2,1,3).reshape(-1)
pos=mx.array(np.stack([order(r),order(c)],-1));inv=1/(10000**(mx.arange(0,32,2,dtype=mx.float32)/32));angle=(pos[...,None]*inv).reshape(-1,32);angle=mx.concatenate([angle,angle],-1);cos,sin=mx.cos(angle),mx.sin(angle)
for i,b in enumerate(model.blocks):
 if i==0:
  n=b.norm1(x);check('blocks.0.norm1',n);qkv=b.attn.qkv(n);check('blocks.0.attn.qkv',qkv)
  q,k,v=mx.split(qkv.reshape(-1,3,16,64),3,1);check('blocks.0.attn.q_norm',b.attn.q_norm(q[:,0]));check('blocks.0.attn.k_norm',b.attn.k_norm(k[:,0]))
  attn=b.attn(n,cos,sin,[h*w]);check('blocks.0.attn',attn);x=x+attn;n=b.norm2(x);check('blocks.0.norm2',n);mlp=b.mlp(n);check('blocks.0.mlp',mlp);x=x+mlp
 else:x=b(x,cos,sin,[h*w])
 if i in [0,1,5,23]:check(f'blocks.{i}',x)
x=model.post_layernorm(x);check('post_layernorm',x);x=model.downsample(x.reshape(-1,4,1024).transpose(0,2,1).reshape(-1,4096));check('downsample',x);check('merger',model.merger(x))
Path('benchmark/mlx/vision_20261102/stage_parity.json').write_text(json.dumps(rows,indent=2)+'\n')
