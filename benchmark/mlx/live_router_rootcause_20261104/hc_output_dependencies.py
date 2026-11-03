"""M5 single-layer MoE graph with actual mHC weights and residual dependencies.

Archived x is collapsed MoE input, so repeated residual is a controlled diagnostic
input, not a captured serving residual. Same residual tensor/native equations in
all arms. No full-model throughput inference from this experiment.
"""
from pathlib import Path
R=Path('/Users/jay/workspace/sglang');p=R/'benchmark/mlx/live_router_rootcause_20261104';s=(p/'production_route.py').read_text()
s=s.replace('    del operands, arrays','    del operands')
s=s.replace('    def project(','''    norm=nn.RMSNorm(a.hidden_size,eps=a.rms_norm_eps)
    norm.weight=arrays['post_attention_layernorm.weight']
    owner=SimpleNamespace(args=a,fuse_hc=True,compile_residual=True,preserve_residual_dtype=True,post_attention_layernorm=norm)
    for name in ('hc_ffn_fn','hc_ffn_base','hc_ffn_scale'):
        setattr(owner,name,arrays[name])
    mx.eval(norm.parameters(),[getattr(owner,name) for name in ('hc_ffn_fn','hc_ffn_base','hc_ffn_scale')])
    def project(''')
s=s.replace("    def forward(value, ids, weights, mode, scope, order, placement):\n        if scope.startswith('live'):","    def forward(value, ids, weights, mode, scope, order, placement):\n        post=residual=None\n        if scope.startswith('hc'):\n            value,post,residual=g.DecoderLayer._branch_inputs(owner,value,'ffn',owner.post_attention_layernorm)\n        if scope.startswith(('live','hc')):")
s=s.replace("elif scope == 'live_x_after_weights':", "elif scope in ('live_x_after_weights','hc_weights','hc_post_after_branch','hc_residual_after_branch','hc_both_after_branch'):")
s=s.replace("        return (down*weights[...,None]).sum(-2).astype(value.dtype) + shared_y", "        branch=(down*weights[...,None]).sum(-2).astype(value.dtype) + shared_y\n        return g.DecoderLayer._branch_output(owner,branch,post,residual) if scope.startswith('hc') else branch")
a=s.index('    specs = ');b=s.index('    fs = ',a)
s=s[:a]+"    specs=[(scope,mode,'up_first','after_shared') for scope in ('hc','hc_weights') for mode in ('native','reuse1')]\n    x=mx.repeat(x[...,None,:],4,axis=-2);mx.eval(x)\n"+s[b:]
s=s.replace('production_route_{args.layer}.json','hc_boundary_{args.layer}.json')
s=s.replace("        return g.DecoderLayer._branch_output","        if scope=='hc_post_after_branch':post=mx.depends(post,branch)\n        if scope=='hc_residual_after_branch':residual=mx.depends(residual,branch)\n        if scope=='hc_both_after_branch':post,residual=mx.depends([post,residual],branch)\n        return g.DecoderLayer._branch_output")
s=s.replace("('hc','hc_weights')","('hc','hc_weights','hc_post_after_branch','hc_residual_after_branch','hc_both_after_branch')")
s=s.replace('hc_boundary_{args.layer}.json','hc_output_dependencies_{args.layer}.json')
ns={'__name__':'__main__','__file__':str(Path(__file__))};exec(compile(s,str(Path(__file__)),'exec'),ns)
