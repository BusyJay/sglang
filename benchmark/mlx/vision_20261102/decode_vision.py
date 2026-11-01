"""Alternating full-model arms with one identical resident target checkpoint."""
import argparse,copy,importlib.util,sys,hashlib,json,os,subprocess,time
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model,load_tokenizer
from sglang.srt.hardware_backend.mlx.models.glm5_next import Model,ModelArgs
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--tokens',type=int,default=256);p.add_argument('--repeats',type=int,default=2);p.add_argument('--output',default='benchmark/mlx/dflash2_20261102/compare_final.json');a=p.parse_args()
if a.baseline:
 spec=importlib.util.spec_from_file_location('host_baseline','benchmark/mlx/dflash2_20261102/baseline_host/dflash_serving.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
 MlxDFlashServing=base.MlxDFlashServing

target=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e');draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
out=Path(a.output);r=dict(device=mx.device_info(),checkpoint=str(target),draft=str(draft),tokens=a.tokens,note='All arms share identical resident target without native MTP weights. Baseline scalar proposal readback vs vector readback once per chain; separate clean processes.',samples=[],swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True),env={k:v for k,v in os.environ.items() if k.startswith('MLX_')})
def save():out.write_text(json.dumps(r,indent=2)+'\n')
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size']);mx.set_cache_limit(1024**3)
start=time.perf_counter();m,_=load_model(target,lazy=True,get_model_classes=lambda config:(lambda args:Model(args,enable_mtp=False),ModelArgs));m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters());r['load_seconds']=time.perf_counter()-start
class AR:
 def clear(self):pass
 def remove(self,rid):pass
 def stats(self):return {}
 def set_remaining(self,rid,n):pass
 def prefill(self,rid,inputs,cache,inputs_embeds=None):return m.model(inputs,cache,inputs_embeds=inputs_embeds)
 def decode_batch(self,rids,inputs,caches,tokens,token_only=True):return m.forward_with_hidden(inputs,caches[0],token_only=token_only)[0]
assert m.mtp is None
r['native_mtp_loaded']=False;r['baseline']=a.baseline
engines={'ar':AR()};d=MlxDFlashServing(m,draft,7)
for n in (3,5):engines['dflash'+str(n)]=MlxDFlashServing(m,draft,n,drafter=d.drafter)
r['resident_bytes']=mx.get_active_memory();tok=load_tokenizer(target)

from PIL import Image,ImageDraw,ImageFont
from transformers import AutoProcessor
from types import SimpleNamespace
from sglang.srt.hardware_backend.mlx.vision import prepare_vision_prompt
from sglang.srt.hardware_backend.mlx.models.glm5_vision import load_glm5_vision
from sglang.srt.parser.glm53_chat_template import render_glm53_flash
processor=AutoProcessor.from_pretrained(target);vision=load_glm5_vision(target)
images=[Image.new('RGB',(112,112),'red'),Image.new('RGB',(168,112),'blue'),Image.new('RGB',(448,448),'white')]
ImageDraw.Draw(images[2]).text((30,160),'HELLO 123',fill='black',font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',52))
r['encoder_samples']=[]
for text_id,(im,text) in enumerate(zip(images,['What color is this image? Answer with one color word.','What color is this image? Answer with one color word.','Read the text in the image. Reply with only the text.'])):
 rendered=render_glm53_flash([{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}],add_generation_prompt=True,reasoning_effort='low')
 data=processor(text=[rendered],images=[im],return_tensors='np',images_kwargs={'max_image_tokens':256})
 ids=data['input_ids'][0].tolist();
 # The model config owns the image placeholder id, independent of tokenizer API.
 image_id=__import__('json').loads((target/'config.json').read_text())['image_token_id'];positions=[i for i,x in enumerate(ids) if x==image_id]
 encraw=[]
 for repeat in range(4):
  start=time.perf_counter();values=vision(mx.array(data['pixel_values']),data['image_grid_thw']);mx.eval(values);encraw.append(time.perf_counter()-start)
 r['encoder_samples'].append(dict(text_id=text_id,grid=data['image_grid_thw'].tolist(),shape=list(values.shape),raw_seconds=encraw));save()
 from sglang.srt.hardware_backend.mlx.vision import VisionPrompt
 vp=VisionPrompt(((positions[0],positions[-1]+1,values),))
 ids=[1000000 if x==image_id else x for x in ids]
 for arm in (['ar','dflash3','dflash5'] if text_id%2==0 else ['dflash5','dflash3','ar']):
  e=engines[arm];e.clear();cache=m.make_cache();before=copy.deepcopy(e.stats());start=time.perf_counter()
  for i in range(0,len(ids),32):
   inputs=mx.array([ids[i:i+32]],mx.int32);embeds=vp.embed(m.model.embed_tokens,inputs,i);h=e.prefill('p',inputs,cache,inputs_embeds=embeds);mx.eval(h,[c.cache for c in cache])
  token=int(m.target_tokens(h[:,-1:]).item());prefill=time.perf_counter()-start;tokens=[token];raw=[]
  for i in range(a.tokens-1):
   e.set_remaining('p',a.tokens-i-1);start=time.perf_counter();v=e.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(v,[c.cache for c in cache]);token=int(v.item());raw.append(time.perf_counter()-start);tokens.append(token)
  after=copy.deepcopy(e.stats());e.remove('p');delta={k:after.get(k,0)-before.get(k,0) for k in ('verify_ct','num_proposed_drafts','num_accept_tokens','num_correct_drafts','reject_ct')}
  row=dict(context=len(ids),text_id=text_id,arm=arm,prefill_seconds=prefill,raw_seconds=raw,tok_s=len(raw)/sum(raw),tokens=tokens,hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),stats=delta,text=tok.decode(tokens));r['samples'].append(row);save();print('DECODE',text_id,arm,row['tok_s'],delta,flush=True)
  del cache,h,v;mx.clear_cache()
 for row in r['samples'][-3:]:assert row['tokens']==r['samples'][-3]['tokens'], 'AR/DFlash image token mismatch'
r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()
