"""The registered parity test rejects the actual former contraction bug."""
import json
import mlx.core as mx
from common import OUT,bank,native,production,routing
mx.random.seed(955)
ids,e=routing(4);p=bank(e,64,4096);x=mx.random.normal((1,4,1,1,4096))
ref=native(x,p,ids)
def run():
    y=production.mlx_nvfp4_gather_rows(x,p.weight,p.scales,p.weight_scale_2*2688,ids)
    mx.eval(y,ref)
    return {'exact':mx.array_equal(y,ref).item(),'changed':mx.sum(y!=ref).item(),'max_abs':mx.max(mx.abs(y-ref)).item()}
good=run();source=production._SOURCE
try:
    production._SOURCE=source.replace('#pragma clang fp contract(off)','')
    production._kernel.cache_clear()
    # Use a distinct JIT name, preventing cached Metal code from hiding mutant.
    create=mx.fast.metal_kernel
    def mutant(**kwargs):
        kwargs['name']+='regression_mutant'
        return create(**kwargs)
    mx.fast.metal_kernel=mutant
    bad=run()
finally:
    production._SOURCE=source
    production._kernel.cache_clear()
    mx.fast.metal_kernel=create
assert good['exact'] and not bad['exact'],(good,bad)
result={'final_source':good,'removing_only_block_rounding_boundary':bad}
(OUT/'regression_proof.json').write_text(json.dumps(result,indent=2));print(result)
