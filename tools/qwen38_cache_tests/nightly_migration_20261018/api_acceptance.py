import base64,json,struct,zlib,urllib.request
from pathlib import Path
url='http://127.0.0.1:8329'
model='nvidia/Qwen3.8-Flash-Next-FP4-Local'
report={}
def post(payload):
    req=urllib.request.Request(url+'/v1/chat/completions',json.dumps({'model':model,'temperature':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False},**payload}).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)
report['math']=post({'messages':[{'role':'user','content':'只回答数字：2+3等于多少？'}]})
assert '5' in report['math']['choices'][0]['message']['content']
def chunk(kind,payload):return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*64)*64))+chunk(b'IEND',b'')
report['vision']=post({'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}},{'type':'text','text':'What is the dominant color? Answer with one color word.'}]}]})
assert 'red' in report['vision']['choices'][0]['message']['content'].lower()
report['tool']=post({'messages':[{'role':'user','content':'Use add to calculate 2+3.'}],'tools':[{'type':'function','function':{'name':'add','description':'Add two integers','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}],'tool_choice':{'type':'function','function':{'name':'add'}}})
call=report['tool']['choices'][0]['message']['tool_calls'][0]['function']
assert call['name']=='add' and sorted(json.loads(call['arguments']).values())==[2,3]
payload={'model':model,'messages':[{'role':'user','content':'Reply exactly OK.'}],'temperature':0,'max_tokens':16,'chat_template_kwargs':{'enable_thinking':False},'stream':True}
req=urllib.request.Request(url+'/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=120) as response:
    lines=response.read().decode()
assert '[DONE]' in lines
parts=[]
for line in lines.splitlines():
    if line.startswith('data: ') and '[DONE]' not in line:
        parts.append(json.loads(line[6:])['choices'][0].get('delta',{}).get('content') or '')
report['sse']=''.join(parts)
assert 'OK' in report['sse']
report['status']='PASS'
Path('/home/user/workspace/sglang-nightly-b12x/tools/qwen38_cache_tests/nightly_migration_20261018/api-acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('PASS math, remote vision, tool calling, OpenAI SSE')
