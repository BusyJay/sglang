#!/usr/bin/env python3
"""Independent-process checkpoint startup ABBA; no module monkeypatch.
Run only with GPU lease. No production writes or checkpoint downloads.
"""
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
CHILD=r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
arm,root,checkpoint,output=sys.argv[1:]
sys.path.insert(0,str(Path(root)/'python'))
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
import sglang.srt.hardware_backend.mlx.mtp as loader
import sglang.srt.hardware_backend.mlx.models.glm5_next as G
paths=[Path(loader.__file__).resolve(),Path(G.__file__).resolve()]
assert all(p.is_relative_to((Path(root)/'python').resolve()) for p in paths),'wrong arm import'
def identity():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
result=dict(arm=arm,source=identity(),checkpoint=str(Path(checkpoint).resolve()),
 checkpoint_revision=Path(checkpoint).name,config_sha256=hashlib.sha256((Path(checkpoint)/'config.json').read_bytes()).hexdigest(),
 device=mx.device_info(),tf32=os.environ.get('MLX_ENABLE_TF32'),
 swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip())
assert result['tf32']=='0'
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
mx.reset_peak_memory()
result['memory_before']=dict(active=mx.get_active_memory(),peak=mx.get_peak_memory(),cache=mx.get_cache_memory())
start=time.perf_counter();model=load_mtp_model(Path(checkpoint));elapsed=time.perf_counter()-start
result.update(load_seconds=elapsed,memory_after=dict(active=mx.get_active_memory(),peak=mx.get_peak_memory(),cache=mx.get_cache_memory()),
 swap_after=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip(),source_after=identity())
assert result['source']==result['source_after'],'source changed during load'
Path(output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(arm=arm,load_seconds=elapsed,memory=result['memory_after'])),flush=True)
'''
def inventory(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'python').rglob('*.py'))}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--final',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',type=int,default=1);p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.groups<1:p.error('groups>=1 required')
    roots={'baseline':a.baseline.resolve(),'final':a.final.resolve()}
    source={arm:inventory(root) for arm,root in roots.items()}
    if any(not v for v in source.values()):raise RuntimeError('empty arm Python inventory')
    if not (a.checkpoint/'config.json').is_file():raise RuntimeError('existing local checkpoint required')
    report=dict(source=source,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),groups=a.groups,
                scope='total startup through load_mtp_model incl existing preparation/materialization; original HEAD vs final combines all mechanism deltas; not E-only prepare cost',samples=[])
    if not a.run:
        print(json.dumps(dict(groups=a.groups,files={k:len(v) for k,v in source.items()},scope=report['scope'],execution='CPU preflight only')));return
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    for group in range(a.groups):
        order=('baseline','final','final','baseline') if group%2==0 else ('final','baseline','baseline','final')
        for pos,arm in enumerate(order):
            root=roots[arm]
            if inventory(root)!=source[arm]:raise RuntimeError('arm source changed before startup')
            out=a.output/f'g{group}_p{pos}_{arm}.json';log=out.with_suffix('.log')
            print('START',group,pos,arm,flush=True)
            with log.open('w') as handle:
                r=subprocess.run([sys.executable,'-c',CHILD,arm,str(root),str(a.checkpoint.resolve()),str(out)],
                    env=dict(os.environ,PYTHONPATH=str(root/'python'),MLX_ENABLE_TF32='0'),stdout=handle,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError('failed startup; see '+str(log))
            if inventory(root)!=source[arm]:raise RuntimeError('arm source changed during startup')
            row=json.loads(out.read_text());row.update(group=group,position=pos);report['samples'].append(row)
            (a.output/'raw.json').write_text(json.dumps(report,indent=2)+'\n')
            print('DONE',group,pos,arm,row['load_seconds'],flush=True)
    medians={arm:statistics.median(s['load_seconds'] for s in report['samples'] if s['arm']==arm) for arm in roots}
    summary=dict(load_seconds=medians,elapsed_change_percent=100*(medians['final']/medians['baseline']-1),
                 raw_seconds={arm:[s['load_seconds'] for s in report['samples'] if s['arm']==arm] for arm in roots},
                 cold_definition='fresh Python process; OS filesystem cache is neither flushed nor assumed cold',scope=report['scope'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
