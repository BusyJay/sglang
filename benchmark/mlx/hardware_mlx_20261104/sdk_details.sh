cd /Users/jay/workspace/sglang || exit 1
/Users/jay/workspace/prod/common/sglang-metal/bin/python - <<'PYCODE'
from pathlib import Path
import subprocess,json,re,hashlib
out=Path('benchmark/mlx/hardware_mlx_20261104');sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip());headers=sdk/'System/Library/Frameworks/Metal.framework/Headers';matches=[]
for p in [headers/'MTLTensor.h',headers/'MTLTypes.h']:
 lines=p.read_text().splitlines();excerpts=[]
 if p.name=='MTLTensor.h': ranges=[(151,185),(196,204),(224,257)]
 else:
  points=[i for i,l in enumerate(lines) if 'MTLTensorDataTypeMetalFloat' in l];ranges=[(max(1,i-1),i+4) for i in points]
 for a,b in ranges: excerpts.append({'first_line':a,'text':'\n'.join(lines[a-1:b])})
 matches.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'excerpts':excerpts})
state={'sdk':str(sdk),'headers':matches,'scope':'SDK source metadata only'};(out/'sdk_quantized_details.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))
PYCODE
