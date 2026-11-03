cd /Users/jay/workspace/sglang || exit 1
/Users/jay/workspace/prod/common/sglang-metal/bin/python - <<'PYCODE'
from pathlib import Path
import hashlib
p=Path('python/sglang/kernels/ops/gemm/mlx_block_fp8.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='02bbf2184708364906ef533020dcdda5303615d8ae7c1d5f22a9d45cf5516bc5'
s=p.read_text();assert s.count('scale_shape = ((n + 127) // 128, k // 128)\n    if any(s.dtype != mx.float32 or s.shape != scale_shape for s in (gs, us)):')==1;p.write_text(s.replace('scale_shape = ((n + 127) // 128, k // 128)\n    if any(s.dtype != mx.float32 or s.shape != scale_shape for s in (gs, us)):','# Serialized ModelOpt scales retain singleton axes; the native\n    # projector consumes their row-major flattened /128 block layout.\n    scale_size = ((n + 127) // 128) * (k // 128)\n    if any(s.dtype != mx.float32 or s.size != scale_size for s in (gs, us)):'))
assert hashlib.sha256(p.read_bytes()).hexdigest()=='37b1e27748b60af5b0f19630423a58d65485ba3fe3d8d0075de34d7b787e3971'
print('FIXED',p)
p=Path('test/registered/kernels/ops/gemm/test_mlx_block_fp8.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='2e05cc741bb73085775337545fa98fd82797ddb9f73ea0b5c2e0f0de2dd97110'
s=p.read_text();assert s.count('q = mlx_fp8_dynamic_32(x)\n                expected = [')==1;p.write_text(s.replace('q = mlx_fp8_dynamic_32(x)\n                expected = [','# Real ModelOpt checkpoint scales include singleton axes.\n                gs = gs.reshape(3, 1, 2, 1)\n                us = us.reshape(3, 1, 2, 1)\n                q = mlx_fp8_dynamic_32(x)\n                expected = ['))
assert hashlib.sha256(p.read_bytes()).hexdigest()=='40dc4b22787c92a37881e17057763c44ca66e2dd9aee89fd266044a829e9b2ce'
print('FIXED',p)
PYCODE
