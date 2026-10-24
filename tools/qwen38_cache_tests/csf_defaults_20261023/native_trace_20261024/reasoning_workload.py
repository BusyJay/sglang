import json,time,urllib.request
PROMPTS=[
'我想设计一个在单张32GB显卡上运行大型MoE模型的推理服务，目标是本地单用户自然聊天，常常会连续思考并生成数千token。权重缓存容量有限，缺失专家从主存搬入；模型有线性注意力、稀疏注意力和两步MTP。请认真推导：如何区分PCIe搬运、GPU算子、CPU提交、MTP接受率造成的瓶颈？如何设计有可信对照的实验，避免命中率变化和生成文本差异造成误判？哪些优化能在不改变数值精度的条件下改善持续生成延迟？请逐步分析各方案的依赖、代价和失败条件，最后给出可执行的优先级。',
'请深入推理这个实际排程问题，给出可验证的最优方案或清楚说明最优性边界。一个项目有任务A至H，耗时分别为3、4、2、5、3、4、2、3小时。A、B无前置依赖；C依赖A；D依赖A和B；E依赖C；F依赖C和D；G依赖D；H依赖E、F和G。最多同时运行两个任务，任务不可中断，两位执行者能力相同。请先分析关键路径和资源下界，再系统检查不同排程选择，找出最短工期并证明没有更短的方案。然后讨论如果D延长一小时、如果允许某一个任务中断、以及如果执行者交接需要半小时，结论各有什么变化。请解释你的推导过程，并检查每个任务的起止时间和前置约束。'
]
URL='http://127.0.0.1:30938'
def post(path,payload):
 req=urllib.request.Request(URL+path,json.dumps(payload).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=600) as r:return r.read().decode()
def generate(ids,limit,profile=None):
 req=urllib.request.Request(URL+'/generate',json.dumps({'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':limit,'ignore_eos':False}}).encode(),{'Content-Type':'application/json'})
 start=time.perf_counter();events=[];final=None;armed=False
 with urllib.request.urlopen(req,timeout=600) as r:
  for line in r:
   if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
   d=json.loads(line[6:]);n=d.get('meta_info',{}).get('completion_tokens',0)
   if n and (not events or n>events[-1][1]):events.append([time.perf_counter()-start,n])
   final=d
   if profile and not armed and n>=512:
    post('/start_profile',{'output_dir':profile,'activities':['CPU','GPU'],'with_stack':False,'record_shapes':False,'profile_id':'reasoning-steady','num_steps':25});armed=True
 if not events:raise RuntimeError(final)
 first,last=events[0],events[-1]
 steady=next((e for e in events if e[1]>=128),first)
 return dict(prompt_tokens=len(ids),output_tokens=last[1],ttft_seconds=first[0],decode_tokens_per_second=(last[1]-first[1])/(last[0]-first[0]),after_128_tokens_per_second=(last[1]-steady[1])/(last[0]-steady[0]),stream_events=events,meta_info=final['meta_info'],response_text=final.get('text'),profile_armed=armed)
