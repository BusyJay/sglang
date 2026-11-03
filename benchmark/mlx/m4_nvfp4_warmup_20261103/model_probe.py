"""Real GLM model graph startup with small synthetic NVFP4 expert banks."""

import ast
import hashlib
import json
import time
import sys
import types

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from common import OUT, ROOT, bank, load, warmup

archive = OUT.parent / "throughput_20261103/baseline/python/sglang/srt/hardware_backend/mlx/models"
tree = ast.parse((archive / "glm5_fp4.py").read_text())
nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in
         ("NVFP4SwitchLinear", "NVFP4Linear", "FP8Linear", "FP8KDAInputs")]
fp4 = types.ModuleType("sglang.srt.hardware_backend.mlx.models.glm5_fp4")
fp4.__dict__.update(mx=mx, nn=nn)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "archived_fp4_classes", "exec"), fp4.__dict__)
sys.modules[fp4.__name__] = fp4
g = load(
    "sglang.srt.hardware_backend.mlx.models.glm5_next",
    OUT / "integration_check/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py",
)
config = json.loads((OUT.parent / "glm_fp4_checkpoint_audit_20261102/fp4/config.json").read_text())["text_config"]
config.update(
    vocab_size=128, hidden_size=512, intermediate_size=512, moe_intermediate_size=512,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
    n_routed_experts=8, num_experts_per_tok=8, q_lora_rank=128, kv_lora_rank=512,
    qk_nope_head_dim=128, qk_head_dim=128, v_head_dim=128,
    index_head_dim=128, index_n_heads=2, index_topk=32, mhc=False,
    first_k_dense_replace=0, layer_types=["linear_attention", "deepseek_sparse_attention"],
    mlp_layer_types=["sparse", "sparse"], indexer_types=["full", "full"],
    linear_attn_config=dict(num_heads=2, head_dim=128, short_conv_kernel_size=4,
                           gate_lower_bound=-5.0, kda_layers=[0], full_attn_layers=[1]),
)
mx.set_memory_limit(2 * 2**30)
mx.set_cache_limit(128 * 2**20)
mx.random.seed(205)
model = g.Model(g.ModelArgs(model_type="glm5_next", text_config=config))
model.set_dtype(mx.bfloat16)
for layer in model.layers:
    block = layer.mlp.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        p = bank(8, 512, 512)
        setattr(block, name, fp4.NVFP4SwitchLinear(p.weight, p.scales, p.weight_scale_2))
model.configure_optimizations("affine_fused")
model.configure_mtp_verification("ffn")
mx.eval(model.parameters())
cache = model.make_cache()
tokens = mx.array([[1, 2, 3]], mx.int32)
before = model.forward_last_logits(tokens, cache)
mx.eval(before, [c.state for c in cache])
offsets = [c.offset for c in cache]
arrays = [[(key, mx.array(v)) for key, v in tree_flatten(c.state)
           if isinstance(v, mx.array)] for c in cache]
start = time.perf_counter()
model.prepare_for_inference()
prepare_ms = (time.perf_counter() - start) * 1000
start = time.perf_counter()
runner_tree = ast.parse((OUT / "integration_check/python/sglang/srt/hardware_backend/mlx/model_runner.py").read_text())
runner_cls = next(n for n in runner_tree.body if isinstance(n, ast.ClassDef) and n.name == "MlxModelRunner")
method = next(n for n in runner_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_warmup_model_for_inference")
namespace = dict(time=time, logger=__import__("logging").getLogger("warmup_probe"))
exec(compile(ast.Module(body=[method], type_ignores=[]), "archived_runner_warmup", "exec"), namespace)
runner = types.SimpleNamespace(
    model=model, _mtp=types.SimpleNamespace(num_drafts=3, capture_layer_ids=(0,)),
    _dflash_draft_path="synthetic", _mtp_num_steps=0, _enable_sampling=False,
)
namespace["_warmup_model_for_inference"](runner)
warmup_ms = (time.perf_counter() - start) * 1000
assert [c.offset for c in cache] == offsets
for owner, snapshot in zip(cache, arrays):
    current = dict(tree_flatten(owner.state))
    for key, b in snapshot:
        assert mx.array_equal(current[key], b).item()

fresh = model.make_cache()
after = model.forward_last_logits(tokens, fresh)
mx.eval(after, [c.state for c in fresh])
assert mx.array_equal(before, after).item()
times = []
for _ in range(10):
    start = time.perf_counter()
    result = model.forward_last_logits(mx.array([[4]], mx.int32), fresh)
    mx.eval(result, [c.state for c in fresh])
    times.append((time.perf_counter() - start) * 1000)
data = dict(
    scope="M4 two-layer synthetic GLM with actual KDA/MLA/model graph owners; not checkpoint/HTTP",
    prepare_ms=prepare_ms, verification_warmup_ms=warmup_ms, first_decode_ms=times[0],
    decode_raw_ms=times, output_exact=True, existing_request_cache_unchanged=True,
    fixed_groups=len(model.model._fixed_groups), peak_bytes=mx.get_peak_memory(),
    source_sha256=hashlib.sha256((OUT / "integration_check/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py").read_bytes()).hexdigest(),
    warmup_sha256=hashlib.sha256((ROOT / "python/sglang/srt/hardware_backend/mlx/warmup.py").read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256((OUT / "integration_check/python/sglang/srt/hardware_backend/mlx/model_runner.py").read_bytes()).hexdigest(),
)
(OUT / "model_probe.json").write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2), flush=True)
