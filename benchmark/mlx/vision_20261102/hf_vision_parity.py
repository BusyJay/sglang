"""Real BF16 checkpoint comparison against installed Transformers vision."""
import json,time
from pathlib import Path
import numpy as np
import torch
import mlx.core as mx
from safetensors.torch import load_file
from PIL import Image
from transformers import AutoImageProcessor
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
from sglang.srt.hardware_backend.mlx.models.glm5_vision import load_glm5_vision
root=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e');torch.set_num_threads(8)
cfg=Glm5NextVisionConfig(**json.loads((root/'config.json').read_text())['vision_config']);cfg._attn_implementation='sdpa'
reference=Glm5NextVisionModel(cfg).to(dtype=torch.bfloat16).eval()
weights={k.removeprefix('model.visual.'):v for k,v in load_file(root/'model-00062-of-00062.safetensors').items() if k.startswith('model.visual.')}
print(reference.load_state_dict(weights,strict=True),flush=True)
model=load_glm5_vision(root);processor=AutoImageProcessor.from_pretrained(root);results=[]
for sizes in [[(112,112)],[(112,168)],[(112,112),(168,112)]]:
 images=[Image.fromarray(np.random.default_rng(i+19).integers(0,256,(h,w,3),dtype=np.uint8)) for i,(h,w) in enumerate(sizes)]
 data=processor(images=images,return_tensors='pt')
 with torch.inference_mode():expected=reference(data['pixel_values'].bfloat16(),data['image_grid_thw']).pooler_output.float().numpy()
 output=model(mx.array(data['pixel_values'].numpy()),data['image_grid_thw'].numpy());mx.eval(output);got=np.array(output.astype(mx.float32));diff=got-expected
 row=dict(sizes=sizes,relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(expected)),max_abs=float(np.abs(diff).max()),cosine=float((got*expected).sum()/np.sqrt((got*got).sum()*(expected*expected).sum())))
 print(row,flush=True);results.append(row)
Path('benchmark/mlx/vision_20261102/hf_vision_parity.json').write_text(json.dumps(results,indent=2)+'\n')
