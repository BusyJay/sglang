cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1
export SGLANG_MLX_GLM_DECODE_EVAL_GROUP=2 SGLANG_MLX_GLM_VERIFY_EVAL_GROUP=2
export MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200
/Users/jay/workspace/prod/common/sglang-metal/bin/python - <<'PYCODE'
import subprocess,sys,json,statistics
from pathlib import Path
out=Path('benchmark/mlx/remote_validation_20261103')
d=json.loads((out/'fullmodel_ab.json').read_text())
for mode in ('baseline','nvfp4','fp8','combined'):
 for arm in ('ar','dflash3'):
  rows=[r for r in d['samples'] if r['mode']==mode and r['arm']==arm]
  print('FULLMODEL_MEAN',mode,arm,round(statistics.mean(r['tok_s'] for r in rows),4),flush=True)
print('SWAP',d['swap_before'],d['swap_after'],flush=True)
p=subprocess.Popen([sys.executable,'-u',str(out/'rootcause_replay.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print('ROOTCAUSE_PID',p.pid,flush=True)
with (out/'rootcause_replay.log').open('w') as f:
 for line in p.stdout:
  f.write(line);f.flush()
  if line.startswith(('REPLAY_READY','CAPTURED','FFN_REPLAY','PROJECT_REPLAY','Traceback','RuntimeError','ValueError','AssertionError')):print(line.strip(),flush=True)
rc=p.wait()
if rc:print('\n'.join((out/'rootcause_replay.log').read_text().splitlines()[-22:]),flush=True)
sys.exit(rc)
PYCODE
