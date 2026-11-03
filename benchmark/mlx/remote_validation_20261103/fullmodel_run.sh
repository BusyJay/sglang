cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1
export SGLANG_MLX_GLM_DECODE_EVAL_GROUP=2 SGLANG_MLX_GLM_VERIFY_EVAL_GROUP=2
export MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200
P=/Users/jay/workspace/prod/common/sglang-metal/bin/python
$P test/registered/kernels/ops/gemm/test_mlx_block_fp8.py -f > benchmark/mlx/remote_validation_20261103/test_fp8_final.log 2>&1
rc=$?; tail -n 8 benchmark/mlx/remote_validation_20261103/test_fp8_final.log
[ "$rc" = 0 ] || exit "$rc"
$P - <<'PYCODE'
import subprocess,sys
from pathlib import Path
out=Path('benchmark/mlx/remote_validation_20261103')
old=out/'fullmodel_ab.log'
if old.exists():old.rename(out/'fullmodel_failed_scale_layout.log')
p=subprocess.Popen([sys.executable,'-u',str(out/'fullmodel_ab.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print('FULLMODEL_PID',p.pid,flush=True)
with old.open('w') as f:
 for line in p.stdout:
  f.write(line);f.flush()
  if line.startswith(('MODEL_READY','FULL300','Traceback','RuntimeError','ValueError','AssertionError')):print(line.strip(),flush=True)
rc=p.wait()
if rc:print('\n'.join(old.read_text().splitlines()[-18:]),flush=True)
sys.exit(rc)
PYCODE
