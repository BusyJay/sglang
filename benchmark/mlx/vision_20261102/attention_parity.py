exec(open('benchmark/mlx/vision_20261102/hf_vision_parity.py').read().split('for sizes in ')[0])
from transformers.models.glm5_next.modeling_glm5_next import apply_rotary_pos_emb_vision
from sglang.srt.hardware_backend.mlx.vision import to_mlx
mx.random.seed(54)
b=model.blocks[0].attn;t=reference.blocks[0].attn
x=mx.random.normal((64,1024)).astype(mx.bfloat16);cpu=lambda x:torch.from_numpy(np.array(x.astype(mx.float32))).bfloat16()
r,c=np.indices((8,8));order=lambda a:a.reshape(4,2,4,2).transpose(0,2,1,3).reshape(-1);pos=np.stack([order(r),order(c)],-1)
with torch.inference_mode():
 cos,sin=reference.rotary_pos_emb(cpu(x),torch.tensor(pos));q,k,v=t.qkv(cpu(x)).reshape(64,3,16,64).permute(1,0,2,3).unbind(0);q,k=t.q_norm(q),t.k_norm(k);qr,kr=apply_rotary_pos_emb_vision(q,k,cos,sin)
 expected=torch.nn.functional.scaled_dot_product_attention(qr.transpose(0,1)[None],kr.transpose(0,1)[None],v.transpose(0,1)[None])
 qm,km,vm=[to_mlx(a).astype(mx.bfloat16) for a in (qr,kr,v)]
 actual=mx.fast.scaled_dot_product_attention(qm.transpose(1,0,2)[None],km.transpose(1,0,2)[None],vm.transpose(1,0,2)[None],scale=64**-.5)
 def row(name,a,e):
  mx.eval(a);a=np.array(a.astype(mx.float32));e=e.float().numpy();d=a-e;print(name,'rel',np.linalg.norm(d)/np.linalg.norm(e),'max',np.abs(d).max(),'equal',np.mean(a==e),flush=True)
 row('identical_sdpa_inputs',actual,expected)
 expected32=torch.nn.functional.scaled_dot_product_attention(qr.float().transpose(0,1)[None],kr.float().transpose(0,1)[None],v.float().transpose(0,1)[None])
 row('fp32_both_sdpa',mx.fast.scaled_dot_product_attention(qm.transpose(1,0,2)[None].astype(mx.float32),km.transpose(1,0,2)[None].astype(mx.float32),vm.transpose(1,0,2)[None].astype(mx.float32),scale=64**-.5),expected32)
 print('torch_bf16_vs_fp32_sdpa',torch.linalg.vector_norm(expected.float()-expected32)/torch.linalg.vector_norm(expected32),flush=True)
 row('identical_sdpa_inputs_fp32',mx.fast.scaled_dot_product_attention(qm.transpose(1,0,2)[None].astype(mx.float32),km.transpose(1,0,2)[None].astype(mx.float32),vm.transpose(1,0,2)[None].astype(mx.float32),scale=64**-.5).astype(mx.bfloat16),expected)
 row('full_attention_identical_input',b(x,to_mlx(cos),to_mlx(sin),[64]),t(cpu(x),torch.tensor([0,64],dtype=torch.int32),(cos,sin)))
