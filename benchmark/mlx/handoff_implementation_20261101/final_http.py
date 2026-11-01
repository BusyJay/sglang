#!/usr/bin/env python3
"""Task-only sequential HTTP acceptance. No server/GPU run without --run.

Derived from review_optimization_20261101/mechanism_revision/final_http.py;
reuses bench_glm5_next_mtp_http contracts. One server at a time; resource
arguments match baseline except requested max-running 1/2. Cancellation is
explicit abort + terminal SSE reason, not closing the transport.
"""
import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(base, route, body=None):
    req = urllib.request.Request(base+route, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=1200) as response:
        data=response.read()
    return data.decode() if route in ('/health','/flush_cache','/abort_request') else json.loads(data)


def check_generation(event, *, length=None, finish=None):
    if 'error' in event: raise RuntimeError(event['error'])
    ids=event['output_ids'];meta=event['meta_info']
    assert isinstance(ids,list) and all(isinstance(t,int) for t in ids)
    assert meta['completion_tokens']==len(ids)
    if length is not None: assert len(ids)==length,(len(ids),length)
    reason=meta.get('finish_reason');assert isinstance(reason,dict),meta
    if finish is not None: assert reason.get('type')==finish,reason
    logprobs=meta.get('output_token_logprobs');top=meta.get('output_top_logprobs')
    assert isinstance(logprobs,list) and len(logprobs)==len(ids)
    assert isinstance(top,list) and len(top)==len(ids)
    for token,lp,choices in zip(ids,logprobs,top):
        assert len(lp)>=2 and lp[1]==token and math.isfinite(lp[0]),lp
        assert isinstance(choices,list) and len(choices)==3,choices
        assert all(len(v)>=2 and math.isfinite(v[0]) for v in choices),choices
    return reason


def check_same_generation(reference, actual):
    check_generation(reference);check_generation(actual)
    assert reference['output_ids']==actual['output_ids']
    for key in ('output_token_logprobs','output_top_logprobs','finish_reason'):
        assert reference['meta_info'][key]==actual['meta_info'][key],key


def stream(base, body, cancel_at=None):
    req=urllib.request.Request(base+'/generate',data=json.dumps(body).encode(),
                               headers={'Content-Type':'application/json'})
    events=[]; abort_response=None; started=time.perf_counter(); sent=None
    with urllib.request.urlopen(req,timeout=1200) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            value=line[5:].strip()
            if value==b'[DONE]': break
            event=json.loads(value); events.append({'elapsed_s':time.perf_counter()-started,'event':event})
            meta=event.get('meta_info',{})
            if cancel_at is not None and sent is None and meta.get('completion_tokens',0)>=cancel_at:
                sent=time.perf_counter()-started
                abort_response=request(base,'/abort_request',{'rid':body['rid'],'abort_all':False})
            if 'error' in event: raise RuntimeError(event['error'])
    if not events: raise AssertionError('No SSE events')
    final=events[-1]['event']; reason=final.get('meta_info',{}).get('finish_reason')
    if cancel_at is None: check_generation(final,finish='length')
    if cancel_at is not None:
        # HTTP 200 acknowledges enqueuing abort only. Require terminal finish
        # reported through the original stream, which remains open to drain.
        assert sent is not None, 'request finished before cancellation was sent'
        assert isinstance(reason,dict) and reason.get('type')=='abort', ('No terminal abort evidence',final)
    return {'rid':body['rid'],'events':events,'final':final,'abort_ack':abort_response,
            'abort_sent_s':sent,'terminal_finish_reason':reason,'elapsed_s':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/Users/jay/workspace/sglang'))
    p.add_argument('--checkout',type=Path)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('benchmark/mlx/handoff_implementation_20261101/results/final_http'))
    p.add_argument('--source-manifest',type=Path,required=True,
                   help='exact final source mapping: relative path -> SHA256; no stale source accepted')
    p.add_argument('--steps',default='0,2')
    p.add_argument('--max-running',default='1,2')
    p.add_argument('--port',type=int,default=30002)
    p.add_argument('--run',action='store_true')
    a=p.parse_args(); root=a.root.resolve(); checkout=(a.checkout or root).resolve()
    expected=json.loads(a.source_manifest.read_text())
    expected=expected.get('source',expected)
    if not isinstance(expected,dict) or not expected: raise ValueError('Empty source manifest')
    for rel,digest in expected.items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert sha(checkout/rel)==digest, ('source mismatch',rel)
    api_paths=['python/sglang/srt/entrypoints/http_server.py','python/sglang/srt/managers/io_struct.py',
               'python/sglang/srt/managers/tokenizer_manager.py']
    for rel in api_paths: assert rel in expected, ('manifest must include final API',rel)
    assert '@app.post("/abort_request")' in (checkout/api_paths[0]).read_text()
    assert 'class AbortReq(' in (checkout/api_paths[1]).read_text()
    bench=checkout/'benchmark/mlx/bench_glm5_next_mtp_http.py'
    tree=ast.parse(bench.read_text()); assert any(isinstance(n,ast.FunctionDef) and n.name=='contracts' for n in tree.body)
    steps=list(map(int,a.steps.split(','))); caps=list(map(int,a.max_running.split(',')))
    assert steps and all(s in (0,2) for s in steps) and caps and all(c in (1,2) for c in caps)
    # Local checkpoint only; no resolver/download path.
    assert a.model_dir.is_dir() and (a.model_dir/'config.json').is_file()
    report={'source':expected,'api_source':{rel:sha(checkout/rel) for rel in api_paths},
            'harness_source_sha256':sha(bench),'checkpoint_revision':a.model_dir.name,
            'checkpoint_config_sha256':sha(a.model_dir/'config.json'),
            'resource_contract':{'context_length':32768,'max_total_tokens':65536,'tested_prompt_lengths':[32,320,2050],'long_context_capacity_validated':False},
            'execution':'CPU preflight only','configurations':[],'server_max_simultaneous':1,
            'cross_config_cold_reference':{},'cross_config_cold_checks':[]}
    if not a.run:
        print(json.dumps(report,indent=2)); return
    sys.path.insert(0,str(checkout/'python'))
    spec=importlib.util.spec_from_file_location('http_contracts',bench)
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    from sglang.test.test_utils import terminate_and_kill_process_tree
    output=a.output if a.output.is_absolute() else root/a.output;output.mkdir(parents=True,exist_ok=True)
    prompts={n:[10,11,12,13]*(n//4)+[10]*(n%4) for n in (32,320,2050)}
    report['execution']='real HTTP, sequential servers'
    def save(): (output/'raw.json').write_text(json.dumps(report,indent=2)+'\n')
    def body(ids,limit=32,**kw):
        return dict(rid='f-http-'+uuid.uuid4().hex,input_ids=ids,
                    sampling_params=dict(temperature=0,max_new_tokens=limit,ignore_eos=True),
                    return_logprob=True,top_logprobs_num=3,**kw)
    def raw(event): return {'panel':harness.panel(event),'event':event}
    for step in steps:
        for cap in caps:
            # Recheck exact source immediately before each server launch.
            for rel,digest in expected.items(): assert sha(checkout/rel)==digest,(rel,'changed during run')
            env=dict(os.environ,PYTHONPATH=str(checkout/'python'),SGLANG_USE_MLX='1',MLX_ENABLE_TF32='0')
            command=[sys.executable,'-u','-m','sglang.launch_server','--model-path',str(a.model_dir),
                '--host','127.0.0.1','--port',str(a.port),'--context-length','32768',
                '--max-total-tokens','65536','--chunked-prefill-size','256',
                '--max-running-requests',str(cap),'--mlx-enable-sampling','--mlx-mtp-num-steps',str(step),
                '--random-seed','1234','--enable-deterministic-inference','--disable-cuda-graph']
            row={'steps':step,'max_running':cap,'command':command,'cases':[],'complete':False,
                 'swap_before':subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip()}
            report['configurations'].append(row);save()
            log=(output/f'server_ar{step}_slots{cap}.log').open('w')
            proc=subprocess.Popen(command,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
            base=f'http://127.0.0.1:{a.port}'
            try:
                for _ in range(300):
                    if proc.poll() is not None: raise RuntimeError('server exited')
                    try: request(base,'/health');break
                    except Exception:time.sleep(1)
                else: raise TimeoutError('server health timeout')
                row['server_info']=request(base,'/get_server_info')
                for n,ids in prompts.items():
                    request(base,'/flush_cache')
                    cold=request(base,'/generate',body(ids));warm=request(base,'/generate',body(ids))
                    check_generation(cold,length=32,finish='length');check_same_generation(cold,warm)
                    # Persist the first cold result and each exact cross-config
                    # check before raising, so failures retain reviewable data.
                    reference_key=str(n)
                    signature=dict(output_ids=cold['output_ids'],meta_info={
                        k:cold['meta_info'][k] for k in
                        ('completion_tokens','output_token_logprobs','output_top_logprobs','finish_reason')})
                    reference=report['cross_config_cold_reference'].get(reference_key)
                    if reference is None:
                        report['cross_config_cold_reference'][reference_key]=dict(
                            steps=step,max_running=cap,generation=signature)
                        check=dict(length=n,steps=step,max_running=cap,reference=True,identical=True)
                    else:
                        check=dict(length=n,steps=step,max_running=cap,reference=False,
                                   identical=reference['generation']==signature,generation=signature)
                    report['cross_config_cold_checks'].append(check);save()
                    if reference is not None:
                        check_same_generation(reference['generation'],signature)

                    cached=warm['meta_info'].get('cached_tokens',0)
                    assert 0<=cached<=n,(cached,n)
                    if n>=320: assert cached>0,(cached,n)
                    row.setdefault("prefix_cache_observations",[]).append(dict(length=n,cached_tokens=cached,short_prompt_can_be_below_aux_snapshot_stride=n<320))
                    assert cold['meta_info'].get('cached_tokens',0)==0
                    stop_id=cold['output_ids'][5]
                    stop_body=body(ids);stop_body['sampling_params'].update(ignore_eos=False,stop_token_ids=[stop_id],no_stop_trim=True)
                    request(base,'/flush_cache');stop=request(base,'/generate',stop_body)
                    reason=check_generation(stop,finish='stop')
                    assert reason.get('matched')==stop_id,reason
                    end=cold['output_ids'].index(stop_id)+1
                    assert stop['output_ids']==cold['output_ids'][:end]
                    assert stop['meta_info']['output_token_logprobs']==cold['meta_info']['output_token_logprobs'][:end]
                    request(base,'/flush_cache')
                    canceled=stream(base,body(ids,256,stream=True),cancel_at=3)
                    after=request(base,'/generate',body(ids))
                    check_same_generation(cold,after)
                    assert 0<=after['meta_info'].get('cached_tokens',0)<=n
                    if n>=320: assert after['meta_info'].get('cached_tokens',0)>0
                    # Terminal abort has been observed before this request.
                    # Cold no-prefix reuse must regain a serving slot; repeat
                    # original prompt additionally checks committed-prefix reuse.
                    reuse_ids=[19,23,29,31]*8
                    reuse=request(base,'/generate',body(reuse_ids))
                    check_generation(reuse,length=32,finish='length')
                    request(base,'/flush_cache');reuse_reference=request(base,'/generate',body(reuse_ids))
                    check_same_generation(reuse_reference,reuse)
                    row['cases'].append({'length':n,'cold':raw(cold),'prefix':raw(warm),'stop':raw(stop),
                        'cancellation':canceled,'slot_reuse':raw(reuse),'slot_reuse_reference':raw(reuse_reference),
                        'after_abort':raw(after),'slot_reuse_evidence':'terminal abort then successful deterministic request; physical slot id not exposed'})
                    save()
                request(base,'/flush_cache')
                with concurrent.futures.ThreadPoolExecutor(max_workers=cap) as pool:
                    results=list(pool.map(lambda _:stream(base,body(prompts[32],32,stream=True)),range(cap)))
                row['parallel_streams']=results
                for result in results:check_generation(result['final'],length=32,finish='length')
                # Existing broad contracts retained, with strict cancellation
                # proof above independently required. Its disconnect-only case
                # is recorded as a separate weaker lifecycle observation.
                row['contracts']=harness.contracts(base,{'english':prompts[32],'chinese':prompts[2050],'code':prompts[32]})
                row['swap']=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip()
                row['complete']=True;save()
            except BaseException as exc:
                row['error']=repr(exc);save();raise
            finally:
                terminate_and_kill_process_tree(proc);proc.wait(timeout=30);log.close()


if __name__=='__main__':main()
