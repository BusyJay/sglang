#!/usr/bin/env python3
"""GPU byte/state checks for independently applied candidate patches."""
import argparse
import json
import os
import time


def transport_check(kind):
    import torch
    from tools.qwen38_cache_tests.test_cpu_runtime import TestSynchronousRuntime
    from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS
    TestSynchronousRuntime.setUpClass()
    fixture = TestSynchronousRuntime()
    fixture.setUp()
    runtime = None
    try:
        runtime = Qwen38ThreadedExpertRuntime(fixture.pack, 'cuda', slots_per_layer=2,
            staging_buffer_bytes=17 if kind == '03' else 64 << 20,
            packed_prefill='packed', packed_min_bytes=1,
            adaptive_cache=True, adaptive_cache_window=16,
            adaptive_cache_interval=0, adaptive_cache_mode='sample')
        runtime.reserve()
        if kind == '03':
            with torch.cuda.stream(runtime.transfer):
                runtime._fill_arena_selected(0, [0, 2])
            runtime.transfer.synchronize()
            for name in REQUIRED_TENSORS:
                want = getattr(fixture.resident[0], name)[[0, 2]]
                if not torch.equal(runtime.arena[name][[0, 2]].cpu(), want):
                    raise AssertionError(f'batch tail mismatch {name}')
        expected = []
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        def consume(prepared):
            ids = prepared.topk_ids.clamp_min(0).long()
            return {name: prepared.tensors[name][ids].clone() for name in REQUIRED_TENSORS}
        # Alternating layers and streams exercises source/arena last-reader fences.
        routes = [[[0, 1], [2, 3]], [[3, 0], [1, 2]], [[0, 2]], [[3, 1]], [[2, 2]], [[-1, 0]]]
        for step in range(48):
            layer = step % 2
            pair = routes[step % len(routes)]
            with torch.cuda.stream(streams[step % 2]):
                ids = torch.tensor(pair, dtype=torch.int32, device='cuda')
                result = runtime.run(layer, ids, consume)
            expected.append((layer, pair, result))
            if step % 8 == 7:
                runtime.synchronize()
        runtime.synchronize()
        for layer, pair, result in expected:
            ids = torch.tensor(pair).clamp_min(0)
            valid = torch.tensor(pair) >= 0
            for name in REQUIRED_TENSORS:
                want = getattr(fixture.resident[layer], name)[ids][valid]
                actual = result[name].cpu()[valid]
                if not torch.equal(actual, want):
                    raise AssertionError(f'byte mismatch {name} layer={layer} ids={pair}')
        stats = runtime.stats_snapshot()
        if kind == '03' and not runtime.packed_transport.counters['prefill_batch_experts']:
            raise AssertionError('batch path not executed; lower startup prefill threshold for fixture only')
        return dict(byte_parity=True, stats=stats)
    finally:
        if runtime is not None:
            runtime.close()
        fixture.tearDown()


def planner_check():
    import torch
    from sglang.kernels.ops.moe.qwen38_expert_cache import _module
    module = _module()
    def state(experts, slots):
        return dict(counts=torch.zeros(experts+1,dtype=torch.int32,device='cuda'),
            lookup=torch.full((experts,),-1,dtype=torch.int32,device='cuda'),
            keys=torch.full((slots,),-1,dtype=torch.int32,device='cuda'),
            frequency=torch.zeros(slots,device='cuda'),
            ages=torch.zeros(slots,dtype=torch.int64,device='cuda'),
            generation=torch.zeros(slots,dtype=torch.int64,device='cuda'),
            clock=torch.zeros(1,dtype=torch.int64,device='cuda'),
            plan=torch.zeros(4+4*slots,dtype=torch.int32,device='cuda'))
    torch.manual_seed(42)
    for experts,slots,width in [(32,10,10),(128,64,10),(512,112,10),(512,128,10)]:
        a,b=state(experts,slots),state(experts,slots)
        for step in range(192):
            ids=torch.randint(experts,(1,width),dtype=torch.int32,device='cuda')
            if step % 7 == 0:ids[:,1:]=ids[:,0:1]
            if step % 11 == 0:ids[:,-1]=-1
            outputs=[]
            for st,parallel in [(a,False),(b,True)]:
                output=torch.empty_like(ids)
                module.decode_plan_parallel(ids,output,*[st[k] for k in
                    ['counts','lookup','keys','frequency','ages','generation','clock','plan']],
                    .95,0,.7,0,0,parallel)
                outputs.append(output)
            for key in a:
                # Unused plan tails need not match; active/miss payloads must.
                if key != 'plan' and not torch.equal(a[key],b[key]):
                    raise AssertionError(f'planner {key} mismatch step={step} slots={slots}')
            if not torch.equal(outputs[0],outputs[1]) or not torch.equal(a['plan'][:4],b['plan'][:4]):
                raise AssertionError('planner output/header mismatch')
            misses,active=(int(a['plan'][i]) for i in [1,2])
            for lo,hi in [(4,4+2*misses),(4+2*slots,4+2*slots+active),(4+3*slots,4+3*slots+active)]:
                if not torch.equal(a['plan'][lo:hi],b['plan'][lo:hi]):
                    raise AssertionError('planner plan payload mismatch')
    return dict(policy_state_parity=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',choices=['01','02','03','04','05'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    result=dict(candidate=args.candidate,success=False)
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        if args.candidate=='03':
            import sglang.srt.layers.moe.qwen38_cache_threaded as threaded
            threaded._DENSE_STARTUP_PREFILL_TOKENS=0  # synthetic sparse fixture
            os.environ['SGLANG_QWEN38_PREFILL_BATCH_PACKED']='1'
        result.update(planner_check() if args.candidate=='05' else transport_check(args.candidate))
        result['success']=True
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
    result['elapsed_seconds']=time.perf_counter()-started
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['success'] else 1)

if __name__=='__main__':main()
