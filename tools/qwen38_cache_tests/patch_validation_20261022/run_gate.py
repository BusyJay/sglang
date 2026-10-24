import os,subprocess,json
from pathlib import Path
root=Path('/results'); library=(root/'cpu-library.txt').read_text().strip().splitlines()[-1]
e=dict(os.environ,SGLANG_QWEN38_FILL_CTA='adaptive',SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH='1',SGLANG_QWEN38_CACHE_W4A16_SPLIT_K='1',SGLANG_QWEN38_SOURCE_CACHE_DIR='/dev/shm/sglang-qwen38-startup')
for shape in ('m3','m12'):
 cmd=['python3','test/manual/kernels/moe/bench_qwen38_hybrid_gate.py','--manifest','/root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json','--library',library,'--cases',str(root/f'fill-cases/gate-cases-{shape}.json'),'--cores','2,3,4,5','--source-mode','shared_thp','--epilogue','merge','--samples','11','--warmup','4','--margin','0.05','--output',str(root/f'gate-{shape}.json')]
 subprocess.run(cmd,env=e,check=True)
subprocess.run(['python3','tools/qwen38_cache_tests/plan_20261022/merge_calibration.py','--prefix','/results/gate','--output','/results/calibration.json'],check=True,env=e)
