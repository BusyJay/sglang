cd /Users/jay/workspace/sglang || exit 1
/Users/jay/workspace/prod/common/sglang-metal/bin/python - <<'PYCODE'
from pathlib import Path
import platform,json,subprocess,hashlib,sys
import mlx.core as mx
out=Path('benchmark/mlx/hardware_mlx_20261104');paths=['python/sglang/srt/hardware_backend/mlx/models/glm5_next.py','python/sglang/srt/hardware_backend/mlx/models/glm5_fp4.py']
state={'hostname':platform.node(),'platform':platform.platform(),'mlx':mx.__version__,'device':mx.device_info(),'python':sys.executable,'xcode':subprocess.check_output(['xcodebuild','-version'],text=True).strip(),'repo':str(Path.cwd()),'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'model_sources':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'scope':'Read-only environment/source inspection; no benchmark'}
(out/'environment_m5.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))
p=Path(paths[1]);lines=p.read_text().splitlines()
for i,l in enumerate(lines):
 if 'gather_qqmm' in l or 'mode="nvfp4"' in l or "mode='nvfp4'" in l: print('MODEL_SOURCE',i+1,'\n'.join(lines[max(0,i-2):i+5]))
PYCODE
