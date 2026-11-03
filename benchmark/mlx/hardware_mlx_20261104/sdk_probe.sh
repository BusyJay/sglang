cd /Users/jay/workspace/sglang || exit 1
/Users/jay/workspace/prod/common/sglang-metal/bin/python - <<'PYCODE'
from pathlib import Path
import json, subprocess, re
out=Path('benchmark/mlx/hardware_mlx_20261104');sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip());files=[]
paths=list((sdk/'System/Library/Frameworks/Metal.framework/Headers').glob('MTLTensor*.h'))
paths+=list((sdk/'usr/metal').rglob('*tensor*')) if (sdk/'usr/metal').exists() else []
for p in paths:
 if not p.is_file(): continue
 text=p.read_text(errors='replace');lines=text.splitlines();matches=[]
 for i,line in enumerate(lines):
  if re.search('Float4|Float8|fp4|fp8|blockFactors|AuxiliaryPlane|tensor_blockwise',line):
   matches.append({'line':i+1,'text':'\n'.join(lines[max(0,i-1):min(len(lines),i+2)])})
 if matches: files.append({'path':str(p),'matches':matches})
state={'sdk':str(sdk),'files':files,'scope':'SDK source metadata; no allocation, dispatch or timing'}
(out/'sdk_quantized_support.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
PYCODE
