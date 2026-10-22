import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('--baseline',action='store_true'); p.add_argument('--mode',default='off'); p.add_argument('--calibration'); p.add_argument('--capture',action='store_true'); p.add_argument('--calibrate-layered',action='store_true'); p.add_argument('--layered-table'); p.add_argument('--stats-interval',type=int,default=0); p.add_argument('--short',action='store_true'); a=p.parse_args()
root=Path('/results'); repo='/results/baseline' if a.baseline else '/workspace/sglang'
cmd=json.loads(Path('/workspace/sglang/tools/qwen38_cache_tests/overhead_20261022/command-no-stats.json').read_text())['command']
i=cmd.index('--model-loader-extra-config')+1; cfg=json.loads(cmd[i])
cfg['stats_log_interval']=a.stats_interval
if not a.baseline: cfg['cpu_hybrid_mode']=a.mode
cmd[i]=json.dumps(cfg)
out=root/a.name; out.mkdir(exist_ok=True); ref=out/'command.json'; ref.write_text(json.dumps({'command':cmd},indent=2))
e=dict(os.environ,PYTHONPATH=repo+'/python',SGLANG_ENABLE_QWEN38_CPU_HYBRID='1' if a.mode=='global' else '0',SGLANG_QWEN38_CPU_CORES='2,3,4,5',SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH='1',SGLANG_QWEN38_FILL_CTA='adaptive',SGLANG_QWEN38_STARTUP_CACHE='auto',SGLANG_QWEN38_SOURCE_CACHE_DIR='/dev/shm/sglang-qwen38-startup',SGLANG_QWEN38_HOST_TOKEN_EMBEDDING='1',SGLANG_QWEN38_BORROW_PREFILL_ARENA='1',SGLANG_QWEN38_PLE_PACKED_CACHE='1',SGLANG_QWEN38_CACHE_W4A16='1',SGLANG_QWEN38_CACHE_W4A16_BACKEND='b12x',SGLANG_QWEN38_CACHE_PREFILL_W4A16='0',SGLANG_QWEN38_FP32_NORM_FUSION='1',SGLANG_QWEN38_CACHE_W4A16_SPLIT_K='1')
if a.capture:
 e['PYTHONPATH']='/results/capture:'+repo+'/python'; e['QWEN38_CAPTURE_LAYERED']='/results/capture-frames'
if a.calibrate_layered:
 e['PYTHONPATH']='/results/calibrate_site:'+repo+'/python'; e['QWEN38_CALIBRATE_LAYERED']='/results/layered-calibration.json'
 Path('/results/calibration-enabled').unlink(missing_ok=True)
if a.layered_table: e['SGLANG_QWEN38_LAYERED_CPU_CALIBRATION']=a.layered_table
if a.calibration: e['SGLANG_QWEN38_CPU_COST_CALIBRATION']=a.calibration
if not a.baseline: e['SGLANG_QWEN38_CPU_LIBRARY']='/results/current-cpu.so'
if a.baseline: e['SGLANG_QWEN38_CPU_LIBRARY']='/results/baseline-cpu.so'
args=['python3',repo+'/tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py','--reference',str(ref),'--output',str(out),'--modes','default']
if a.calibrate_layered: args[1]='/results/benchmark_calibrate.py'
if a.short: args+=['--short-only']
subprocess.run(args,env=e,cwd=repo,check=True)
