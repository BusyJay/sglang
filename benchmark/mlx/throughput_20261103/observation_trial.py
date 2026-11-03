exec(open('benchmark/mlx/throughput_20261103/short_bottleneck.py').read().split('for j in range(3):')[0])
from observation_fast import prepare,current
from sglang.srt.hardware_backend.mlx.models import glm5_next as glm
import statistics
original_init=glm._GraphObservation.__init__
original_current=glm._GraphObservation.current
def initialize(self,modules):original_init(self,modules);prepare(self)
glm._GraphObservation.__init__=initialize
plan=glm._GraphObservation(m.layers[:2]);assert current(plan,plan.modules)
# Every original observed field must agree with reference; mutate disposable copies.
from types import SimpleNamespace
for v,keys,length,observed in plan.nodes:
 assert current(plan,plan.modules)==original_current(plan,plan.modules)
 if keys is None or not observed:continue
 for key,scalar,old in observed[:2]:
  storage=v if isinstance(v,dict) and key in v else vars(v)
  if key not in storage:continue
  saved=storage[key]
  storage[key]=False if scalar and saved is not False else None if scalar else object()
  assert current(plan,plan.modules)==original_current(plan,plan.modules),(type(v),key)
  storage[key]=saved
assert current(plan,plan.modules)
for fn in (original_current,current):
 ts=[]
 for j in range(5):
  t=time.perf_counter()
  for _ in range(1000):assert fn(plan,plan.modules)
  ts.append(time.perf_counter()-t)
 print('OBS_MICRO',fn.__name__,statistics.median(ts),flush=True)
samples=[]
for mode in ('baseline','planned','planned','baseline'):
 glm._GraphObservation.current=current if mode=='planned' else original_current
 m.configure_fixed_groups(True)
 for j in range(3):
  for arm in ('ar','dflash3') if j%2==0 else ('dflash3','ar'):
   row=generate(j,arm);row['mode']=mode;samples.append(row)
   (root/'observation_ab.json').write_text(json.dumps(samples,indent=2))
   print('OBS300',mode,j,arm,row['tok_s'],row['stats'],flush=True)
glm._GraphObservation.current=original_current
glm._GraphObservation.__init__=original_init
for j in range(3):
 ss=[v for v in samples if v['text_id']==j];assert all(v['tokens']==ss[0]['tokens'] for v in ss),j
print('OBS_AB_COMPLETED_EXACT',flush=True)
