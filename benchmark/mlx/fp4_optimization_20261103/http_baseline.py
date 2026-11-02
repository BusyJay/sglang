import json,time,urllib.request,subprocess
from pathlib import Path
root=Path('benchmark/mlx/fp4_optimization_20261103')
def req(route,body=None):
 r=urllib.request.Request('http://127.0.0.1:30007'+route,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=180) as f:return json.loads(f.read())
r={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'server':req('/get_server_info'),'samples':[]}
for text in ['Explain why the sky is blue.','Write an iterative Python Fibonacci function and explain it.','请介绍过拟合以及三种应对方法。']:
 for repeat in range(2):
  start=time.perf_counter();v=req('/generate',{'text':text,'sampling_params':{'temperature':0,'max_new_tokens':64,'ignore_eos':True}})
  row={'text':text,'repeat':repeat,'wall_s':time.perf_counter()-start,'response':v};r['samples'].append(row);(root/'http_baseline.json').write_text(json.dumps(r,indent=2,ensure_ascii=False));print(text,repeat,row['wall_s'],v['meta_info'],flush=True)
