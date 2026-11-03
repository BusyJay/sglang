"""Exact x-producing GPU copy tests for dependency integration boundaries."""
from pathlib import Path
p=Path('/Users/jay/workspace/sglang/benchmark/mlx/live_router_rootcause_20261104/production_route.py')
s=p.read_text()
s=s.replace("    def forward(value, ids, weights, mode, scope, order, placement):", "    producer=mx.fast.metal_kernel(name='diagnostic_copy_x_producer',input_names=['x'],output_names=['y'],source='uint i=thread_position_in_grid.x;y[i]=x[i];')\n    def forward(value, ids, weights, mode, scope, order, placement):\n        if scope.startswith('producer'):\n            value=producer(inputs=[value],grid=(value.size,1,1),threadgroup=(256,1,1),output_shapes=[value.shape],output_dtypes=[value.dtype])[0]")
s=s.replace("        if scope.startswith('live'):", "        if scope.startswith(('live','producer')):")
s=s.replace("elif scope == 'live_x_after_weights':", "elif scope in ('live_x_after_weights','producer_weights'):")
a=s.index('    specs = ');b=s.index('    fs = ',a)
s=s[:a]+"    specs=[(scope,mode,'up_first','after_shared') for scope in ('live','live_x_after_weights','producer','producer_weights') for mode in ('native','reuse1')]\n"+s[b:]
s=s.replace("production_route_{args.layer}.json", "producer_dependencies_{args.layer}.json")
ns={'__name__':'__main__','__file__':str(Path(__file__))};exec(compile(s,str(Path(__file__)),'exec'),ns)
