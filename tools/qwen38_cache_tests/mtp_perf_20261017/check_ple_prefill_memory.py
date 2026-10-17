import ast,json
from pathlib import Path
from types import SimpleNamespace
import torch

def load(path):
    tree=ast.parse(Path(path).read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_short_conv")
    fn.decorator_list=[];fn.returns=None
    for arg in fn.args.args:arg.annotation=None
    namespace={"torch":torch,"F":torch.nn.functional,"get_req_to_token_pool":lambda:pool,"_ple_track_targets":lambda *_:None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),"exec"),namespace)
    return namespace["_short_conv"]

old=load("/prod/homelab-llm/models/sources/fp4-local-verify-mailbox-bba4aaf182a3/python/sglang/srt/models/qwen4_exp.py")
new=load("/workspace/sglang/python/sglang/srt/models/qwen4_exp.py")
width,channels=8192,10240
torch.manual_seed(17)
x=torch.randn(width,channels,dtype=torch.bfloat16,device="cuda")
state=torch.randn(1,channels,4,dtype=torch.bfloat16,device="cuda")
initial=state.clone()
pool=SimpleNamespace(short_conv_layer_cache=lambda _:state)
batch=SimpleNamespace(use_decode_fast_path=False,lengths=torch.tensor([width],device="cuda"),row_width=width,req_indices=torch.zeros(width,dtype=torch.long,device="cuda"),token_offsets=torch.arange(width,device="cuda"),state_indices=torch.tensor([0],device="cuda"),mode=SimpleNamespace(is_target_verify=lambda:False))
model=SimpleNamespace(layer_id=0,short_conv_dilation=2,short_conv_state_len=4,conv_channels=channels,conv1d=SimpleNamespace(weight=torch.randn(channels,1,3,device="cuda",dtype=torch.bfloat16)))
rows={};outputs={};states={}
with torch.inference_mode():
    for name,fn in [("old",old),("new",new)]:
        state.copy_(initial);torch.cuda.synchronize();torch.cuda.empty_cache()
        base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        y=fn(model,x,None,batch);torch.cuda.synchronize()
        rows[name]={"base_bytes":base,"peak_increment_bytes":torch.cuda.max_memory_allocated()-base}
        outputs[name]=y.cpu();states[name]=state.cpu();del y
assert torch.equal(outputs["old"],outputs["new"])
assert torch.equal(states["old"],states["new"])
assert rows["new"]["peak_increment_bytes"]<rows["old"]["peak_increment_bytes"]
result={"status":"passed","shape":[width,channels],"exact_output":True,"exact_recurrent_state":True,"memory":rows,"saved_peak_bytes":rows["old"]["peak_increment_bytes"]-rows["new"]["peak_increment_bytes"]}
Path("/workspace/sglang/tools/qwen38_cache_tests/mtp_perf_20261017/ple-prefill-memory.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
