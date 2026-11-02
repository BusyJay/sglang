import json,time,urllib.request
from pathlib import Path
r=Path('benchmark/mlx/ar_optimization_20261103');url='http://127.0.0.1:30007'
def req(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode();request=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'});return json.loads(urllib.request.urlopen(request,timeout=180).read())
results=[]
texts=['Explain why the sky is blue.','Write a Python function that computes the first n Fibonacci numbers, with an explanation.','请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
for text in texts:
 for repeat in range(2):
  payload={'text':text,'sampling_params':{'temperature':0,'max_new_tokens':128,'ignore_eos':True},'return_logprob':False};t=time.perf_counter();v=req('/generate',payload);elapsed=time.perf_counter()-t;row={'text':text,'repeat':repeat,'elapsed':elapsed,'response':v};results.append(row);print('HTTP_AR',text[:8],repeat,elapsed,v.get('meta_info'),flush=True)
  (r/'http_ar.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
(r/'server_health.json').write_text(json.dumps(req('/get_server_info'),indent=2))
