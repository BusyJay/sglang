"""Diagnostic-only kernel clones expose QK, normalization and PV accumulation."""
import importlib.util
import inspect
import json
import os
import re
import runpy
from pathlib import Path

import torch
import sglang.srt.layers.attention.qsa.sparse_attn as op

os.environ.setdefault('QSA_DIAG_OUTPUT', '/results/qsa-fp8-stages-half-final')
fixture = runpy.run_path('/results/diagnose_qsa_fp8.py')
root = Path(os.environ['QSA_DIAG_OUTPUT'])
source = inspect.getsource(op._sparse_gqa_chunk_prefill.fn)
source = source.replace('PAGED_KV: tl.constexpr = False,',
    'PAGED_KV: tl.constexpr = False, DEBUG_Q=None, DEBUG_S=None, DEBUG_A=None, DEBUG_N=None,')
source = source.replace('    k_base =', '''    if DEBUG_Q is not None:
        tl.store(DEBUG_Q + query * NUM_KV_HEADS * GROUP_SIZE * HEAD_DIM
            + (group * GROUP_SIZE + offs_h[:, None]) * HEAD_DIM + offs_d[None, :],
            q_values, mask=(offs_h < GROUP_SIZE)[:, None])
    k_base =''')
source = source.replace('        next_max =', '''        if DEBUG_S is not None:
            tl.store(DEBUG_S + query * NUM_KV_HEADS * GROUP_SIZE * topk
                + (group * GROUP_SIZE + offs_h[:, None]) * topk + current[None, :],
                scores, mask=(offs_h < GROUP_SIZE)[:, None] & (current < topk)[None, :])
        next_max =''')
source = source.replace('    output = accumulator /', '''    if DEBUG_A is not None:
        tl.store(DEBUG_A + query * NUM_KV_HEADS * GROUP_SIZE * HEAD_DIM
            + (group * GROUP_SIZE + offs_h[:, None]) * HEAD_DIM + offs_d[None, :],
            accumulator, mask=(offs_h < GROUP_SIZE)[:, None])
        tl.store(DEBUG_N + query * NUM_KV_HEADS * GROUP_SIZE + group * GROUP_SIZE + offs_h,
            normalizer, mask=offs_h < GROUP_SIZE)
    output = accumulator /''')

results, layouts = {}, {}
for name, paired_k, paired_v in [('reference',True,True),('old',False,False),
                                 ('keys_only',True,False),('values_only',False,True),('both',True,True)]:
    text = source.replace('            PAGED_KV\n            and k.dtype', f'            {paired_k}\n            and k.dtype')
    text = text.replace('            PAGED_KV\n            and v.dtype', f'            {paired_v}\n            and v.dtype')
    path = root / ('kernel_' + name + '.py')
    path.write_text('import triton\nimport triton.language as tl\nfrom sglang.srt.layers.attention.qsa.sparse_attn import _qsa_fp8_pair_to_bf16\n' + text)
    spec = importlib.util.spec_from_file_location('stage_kernel_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module._sparse_gqa_chunk_prefill
    dump = dict(DEBUG_Q=torch.full((26,24,256), float('nan'),device='cuda',dtype=torch.bfloat16),
                DEBUG_S=torch.full((26,24,2048),float('nan'),device='cuda'),
                DEBUG_A=torch.empty((26,24,256),device='cuda'),
                DEBUG_N=torch.empty((26,24),device='cuda'))
    class Record:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                result = kernel[grid](*args, **kwargs, **dump)
                ir = result.asm['ttgir']
                (root/(name+'.ttgir')).write_text(ir)
                layouts[name] = [int(x) for x in re.findall(r'kWidth = (\d+)',ir)]
                return result
            return launch
    namespace = dict(op.sparse_gqa_fwd_interface_triton_ck.__globals__)
    namespace['_sparse_gqa_chunk_prefill'] = Record()
    exec(compile(inspect.getsource(op.sparse_gqa_fwd_interface_triton_ck),'<stage-wrapper>','exec'),namespace)
    args = [fixture[n] for n in ('q','k','v','indices','cu_q','cu_k','lens')]
    extra = dict(req_to_token=fixture['table'],req_indices=fixture['req'])
    if name == 'reference':
        args[1:3] = [fixture['bf_k'],fixture['bf_v']]
        extra = {}
    output = namespace['sparse_gqa_fwd_interface_triton_ck'](*args,256**-.5,**extra)
    torch.cuda.synchronize()
    results[name] = dict(dump,output=output)

report = {}
for name, values in results.items():
    row = {}
    for stage, value in values.items():
        expected = results['reference'][stage]
        mismatch = (value != expected) & ~(value.isnan() & expected.isnan())
        row[stage] = dict(count=int(mismatch.sum()),
                         max_abs=float((value-expected).abs()[mismatch].max()) if mismatch.any() else 0.)
    row['kWidth'] = sorted(set(layouts[name]))
    report[name] = row
report['instrumentation_reference_output_exact'] = torch.equal(
    results['reference']['output'],fixture['results']['packed_bf16'])
report['instrumentation_old_output_exact'] = torch.equal(
    results['old']['output'],fixture['results']['packed_fp8'])
(root/'stages.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
