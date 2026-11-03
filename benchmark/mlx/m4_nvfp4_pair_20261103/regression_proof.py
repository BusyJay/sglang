"""The new BF16 test rejects promotion of native sigmoid intermediates."""
import mlx.core as mx
import mlx.nn as nn
import json
from common import OUT,bank,native,routing,production
mx.random.seed(144);ids,e=routing(1);g,u=bank(e,2048,4096),bank(e,2048,4096);x=mx.random.normal((1,1,1,1,4096)).astype(mx.bfloat16)
gate,up=native(x,g,ids),native(x,u,ids);ref=nn.silu(mx.minimum(gate,7.0))*mx.clip(up,-7.0,7.0)
def evaluate():
    y=production.mlx_nvfp4_swiglu(x,g.weight,g.scales,g.weight_scale_2*2688,u.weight,u.scales,u.weight_scale_2*2688,ids,limit=7.0);mx.eval(y,ref)
    return {'exact':mx.array_equal(y,ref).item(),'changed':mx.sum(y!=ref).item(),'max_abs':mx.max(mx.abs(y.astype(mx.float32)-ref.astype(mx.float32))).item()}
good=evaluate();source=production._SOURCE;create=mx.fast.metal_kernel
try:
    production._SOURCE=source.replace('metal::abs(gate)','metal::abs(float(gate))')
    production._kernel.cache_clear()
    def mutant(**kwargs):
        kwargs['name']+='fp32_sigmoid_mutant';return create(**kwargs)
    mx.fast.metal_kernel=mutant
    bad=evaluate()
finally:
    production._SOURCE=source;production._kernel.cache_clear();mx.fast.metal_kernel=create
assert good['exact'] and not bad['exact'],(good,bad)
(OUT/'regression_proof.json').write_text(json.dumps({'final':good,'fp32_promoted_sigmoid':bad},indent=2));print(good,bad)
