"""Local MLX vision then EPD ZMQ tokenizer transport serving acceptance."""
import base64,io,json,os,subprocess,time,urllib.request,socket,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
root=Path('/Users/jay/workspace/sglang');art=root/'benchmark/mlx/vision_20261102';art.mkdir(exist_ok=True)
target='/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'
draft='/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a'
env=dict(os.environ,PYTHONPATH=str(root/'python'),SGLANG_USE_MLX='1',MLX_ENABLE_TF32='0')
base=[sys.executable,'-u','-m','sglang.launch_server','--model-path',target,'--served-model-name','glm-vision','--host','127.0.0.1','--disable-cuda-graph','--disable-overlap-schedule','--mm-process-config','{"image":{"max_image_tokens":256}}']
language=base+['--port','30007','--context-length','8192','--max-total-tokens','17024','--chunked-prefill-size','32','--max-running-requests','2','--mlx-enable-sampling','--random-seed','1234','--enable-deterministic-inference','--speculative-algorithm','DFLASH','--speculative-draft-model-path',draft,'--speculative-dflash-block-size','4','--reasoning-parser','glm45']
procs=[]
def call(port,route,body=None):
 data=None if body is None else json.dumps(body).encode();req=urllib.request.Request(f'http://127.0.0.1:{port}'+route,data=data,headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=600) as response:
   raw=response.read();return json.loads(raw) if route!='/health' else raw
 except urllib.error.HTTPError as e:
  if route!='/health':print('HTTP_ERROR',e.read().decode(),flush=True)
  raise

def launch(name,command,port):
 with socket.socket() as sock:
  sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
 log=(art/(name+'.server.log')).open('w');p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);procs.append(p)
 start=time.perf_counter()
 for attempt in range(180):
  if p.poll() is not None:raise RuntimeError(name+' exited; see '+str(art/(name+'.server.log')))
  try:call(port,'/health');return time.perf_counter()-start
  except Exception:
   if attempt and attempt % 15 == 0:print('WAIT_HEALTH',name,attempt,flush=True)
   if attempt > 30 and 'Encoder health check failed' in (art/(name+'.server.log')).read_text():raise RuntimeError('Encoder health failed; see server log')
   time.sleep(1)
 raise RuntimeError(name+' startup timeout')
def shutdown():
 from sglang.test.test_utils import terminate_and_kill_process_tree
 for p in reversed(procs):terminate_and_kill_process_tree(p)
 procs.clear()
def image(size,color,label=''):
 im=Image.new('RGB',size,color)
 if label:ImageDraw.Draw(im).text((30,160),label,fill='black',font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',52))
 buff=io.BytesIO();im.save(buff,format='PNG');return 'data:image/png;base64,'+base64.b64encode(buff.getvalue()).decode()
imgs=[image((112,112),'red'),image((168,112),'blue'),image((448,448),'white','HELLO 123')]
r={'commands':{},'modes':{},'swap_before':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}
def save(): (art/'http_vision.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 for mode in ['local','epd']:
  if mode=='epd':
   encoder=base+['--port','30008','--encoder-only','--encoder-transfer-backend','zmq_to_tokenizer']
   r['commands']['encoder']=encoder;print('ENCODER_START',launch('encoder',encoder,30008),flush=True)
   command=language+['--language-only','--encoder-urls','http://127.0.0.1:30008','--encoder-transfer-backend','zmq_to_tokenizer']
  else:command=language
  r['commands'][mode]=command;startup=launch(mode,command,30007);print('LANGUAGE_START',mode,startup,flush=True)
  samples=[]
  for ids,prompt in [([0],'What color is this image? Answer with one color word.'),([1],'What color is this image? Answer with one color word.'),([0,1],'Describe the color of each image in order.'),([2],'Read the text in the image. Reply with only the text.')]:
   messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':imgs[i]}} for i in ids]+[{'type':'text','text':prompt}]}]
   for repeat in range(2):
    start=time.perf_counter();out=call(30007,'/v1/chat/completions',dict(model='glm-vision',messages=messages,temperature=0,max_tokens=256,chat_template_kwargs={'reasoning_effort':'low'}));elapsed=time.perf_counter()-start
    row=dict(images=ids,repeat=repeat,seconds=elapsed,text=out['choices'][0]['message']['content'],reasoning=out['choices'][0]['message'].get('reasoning_content'),usage=out['usage']);samples.append(row);print(mode,row,flush=True)
    r['modes'][mode]=dict(startup=startup,samples=samples);save()
  shutdown()
 r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()
 for a,b in zip(r['modes']['local']['samples'],r['modes']['epd']['samples']):assert a['text']==b['text'],(a,b)
 print('LOCAL_EPD_RESPONSES_MATCH',flush=True)
finally:shutdown();save()
