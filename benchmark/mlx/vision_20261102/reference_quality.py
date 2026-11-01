"""Generate with Torch reference and MLX vision features on one resident target."""
import json,time
from pathlib import Path
import numpy as np,torch,mlx.core as mx
from PIL import Image,ImageDraw,ImageFont
from transformers import AutoProcessor
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
from safetensors.torch import load_file
from mlx_lm.utils import load_model,load_tokenizer
from sglang.srt.hardware_backend.mlx.models.glm5_next import Model,ModelArgs
from sglang.srt.hardware_backend.mlx.models.glm5_vision import load_glm5_vision
from sglang.srt.hardware_backend.mlx.vision import VisionPrompt,to_mlx
from sglang.srt.parser.glm53_chat_template import render_glm53_flash
root=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e')
torch.set_num_threads(8);cfg=Glm5NextVisionConfig(**json.loads((root/'config.json').read_text())['vision_config']);cfg._attn_implementation='sdpa'
ref=Glm5NextVisionModel(cfg).to(dtype=torch.bfloat16).eval();weights={k.removeprefix('model.visual.'):v for k,v in load_file(root/'model-00062-of-00062.safetensors').items() if k.startswith('model.visual.')};ref.load_state_dict(weights,strict=True);del weights
vision=load_glm5_vision(root);processor=AutoProcessor.from_pretrained(root);tok=load_tokenizer(root);image_id=json.loads((root/'config.json').read_text())['image_token_id'];stop=set(json.loads((root/'generation_config.json').read_text())['eos_token_id'])
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size']);mx.set_cache_limit(1024**3)
m,_=load_model(root,lazy=True,get_model_classes=lambda config:(lambda a:Model(a,enable_mtp=False),ModelArgs));m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters())
results=[]
im1=Image.new('RGB',(112,112),'red');im2=Image.new('RGB',(168,112),'blue');ocr=Image.new('RGB',(448,448),'white');ImageDraw.Draw(ocr).text((30,160),'HELLO 123',fill='black',font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',52))
for images,prompt,wanted in [([im1],'What color is this image? Answer with one color word.',['red']),([im2],'What color is this image? Answer with one color word.',['blue']),([im1,im2],'Describe the color of each image in order.',['red','blue']),([ocr],'Read the text in the image. Reply with only the text.',['hello','123'])]:
 rendered=render_glm53_flash([{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]}],add_generation_prompt=True,reasoning_effort='low');data=processor(text=[rendered],images=images,return_tensors='pt',images_kwargs={'max_image_tokens':256})
 with torch.inference_mode():expected=ref(data['pixel_values'].bfloat16(),data['image_grid_thw']).pooler_output
 actual=vision(to_mlx(data['pixel_values']),data['image_grid_thw'].numpy());mx.eval(actual)
 ids=data['input_ids'][0].tolist();spans=[];start=None
 for i,x in enumerate(ids+[0]):
  if x==image_id and start is None:start=i
  if x!=image_id and start is not None:spans.append((start,i));start=None
 row=dict(prompt=prompt,arms={})
 for name,values in [('mlx',actual),('torch_bf16',to_mlx(expected).astype(mx.bfloat16))]:
  cursor=0;vps=[]
  for start,end in spans:vps.append((start,end,values[cursor:cursor+end-start]));cursor+=end-start
  vp=VisionPrompt(tuple(vps));cache=m.make_cache();tokens=[]
  for i in range(0,len(ids),32):
   x=mx.array([[1000000 if v==image_id else v for v in ids[i:i+32]]],mx.int32);embeds=vp.embed(m.model.embed_tokens,x,i);h=m.model(x,cache,inputs_embeds=embeds);mx.eval(h,[c.cache for c in cache])
  token=int(m.target_tokens(h[:,-1:]).item())
  for i in range(128):
   if token in stop:break
   tokens.append(token);v=m.forward_with_hidden(mx.array([[token]],mx.int32),cache,token_only=True)[0];mx.eval(v,[c.cache for c in cache]);token=int(v.item())
  text=tok.decode(tokens);assert all(w in text.lower() for w in wanted),text;row['arms'][name]=dict(text=text,tokens=tokens);del cache,h;mx.clear_cache()
 print(row,flush=True);results.append(row);Path('benchmark/mlx/vision_20261102/reference_quality.json').write_text(json.dumps(results,indent=2)+'\n')
