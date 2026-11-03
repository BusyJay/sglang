cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1
P=/Users/jay/workspace/prod/common/sglang-metal/bin/python
$P - <<'PY'
from pathlib import Path
import json,subprocess
O=Path('benchmark/mlx/remote_validation_20261103');files=['python/sglang/kernels/ops/'+g+'/__init__.py' for g in ('gemm','moe','quantization')]
def violations(path):
 p=subprocess.run(['/Users/jay/workspace/prod/common/sglang-metal/bin/python','-m','ruff','check','--output-format','json',str(path)],capture_output=True,text=True)
 assert p.returncode in (0,1),p.stderr
 return [(v['code'],v['message']) for v in json.loads(p.stdout)]
from collections import Counter
r={}
for f in files:
 old=Counter(violations(O/'baseline'/f));new=Counter(violations(Path(f)));added=new-old
 assert not added,(f,added);r[f]=dict(existing_violations=sum(old.values()),added_violations=sum(added.values()))
(O/'registry_lint.json').write_text(json.dumps(r,indent=2));print('REGISTRY_NO_NEW_LINT',r)
from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend
import sglang.kernels.ops.gemm,sglang.kernels.ops.moe,sglang.kernels.ops.quantization
for op in ('gemm.mlx_block_fp8_pair','gemm.mlx_nvfp4_gather_rows','gemm.mlx_nvfp4_swiglu','moe.mlx_expert_pair_worklist','quantization.mlx_fp8_dynamic_32'):
 s=registry.get_backend(op,KernelBackend.MLX);print('REGISTERED',op,s.target)
PY
rc=$?
[ "$rc" = 0 ] || exit "$rc"
$P test/registered/unit/kernels/test_kernels_namespace.py -f > benchmark/mlx/remote_validation_20261103/test_namespace.log 2>&1
rc=$?
tail -n 12 benchmark/mlx/remote_validation_20261103/test_namespace.log
exit "$rc"
