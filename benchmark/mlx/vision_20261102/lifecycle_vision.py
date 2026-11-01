"""Vision hash isolation, long prefix reuse, concurrency and abort recovery."""
import json,urllib.request,time,concurrent.futures,base64,io
from PIL import Image
from pathlib import Path
port=30007
out=Path('benchmark/mlx/vision_20261102/lifecycle_'+__import__('os').environ.get('VISION_MODE','local')+'.json')
def call(route,body):
 with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}'+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as r:
  data=r.read();return json.loads(data) if data and route!='/abort_request' else data.decode()
def uri(color):
 b=io.BytesIO();Image.new('RGB',(112,112),color).save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def body(color,long=False):
 text=('Background facts: This is a color identification task. '*120 if long else '')+'What color is this image? Answer with one color word.'
 return dict(model='glm-vision',messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':uri(color)}),dict(type='text',text=text)])],temperature=0,max_tokens=128,chat_template_kwargs={'reasoning_effort':'low'})
r={'cases':[]}
def check(name,payload,expected):
 start=time.perf_counter();result=call('/v1/chat/completions',payload);message=result['choices'][0]['message'];text=message['content'];assert expected in text.lower(),(name,message);row=dict(name=name,text=text,seconds=time.perf_counter()-start,usage=result['usage']);r['cases'].append(row);print(row,flush=True);out.write_text(json.dumps(r,indent=2)+'\n');return row
for color in ['red','blue','red']:check('same_shape_'+color,body(color),color)
for repeat in range(2):check('long_prefix_'+str(repeat),body('red',True),'red')
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 futures=[pool.submit(check,'concurrent_'+color,body(color),color) for color in ['red','blue']]
 for f in futures:f.result()
payload={'rid':'vision-abort-case','text':'<|begin_of_image|><|image|><|end_of_image|> Describe this image at great length.','image_data':[uri('red')],'sampling_params':{'temperature':0,'max_new_tokens':512},'stream':True}
request=urllib.request.Request(f'http://127.0.0.1:{port}/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
events=[];sent=False
with urllib.request.urlopen(request,timeout=300) as response:
 for raw in response:
  line=raw.decode().strip()
  if not line.startswith('data: ') or line=='data: [DONE]':continue
  event=json.loads(line[6:]);events.append(event)
  if not sent and event.get('meta_info',{}).get('completion_tokens',0)>=8:
   call('/abort_request',{'rid':payload['rid'],'abort_all':False});sent=True
assert sent and events[-1]['meta_info']['finish_reason']['type']=='abort',events[-1]
r['abort']=events[-1]['meta_info'];check('post_abort',body('blue'),'blue');text=call('/v1/chat/completions',dict(model='glm-vision',messages=[{'role':'user','content':'What is 2 + 2? Reply with the number.'}],temperature=0,max_tokens=64,chat_template_kwargs={'reasoning_effort':'low'}));assert '4' in text['choices'][0]['message']['content'];r['text_regression']=text['choices'][0]['message'];r['complete']=True;out.write_text(json.dumps(r,indent=2)+'\n')
