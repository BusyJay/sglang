#!/usr/bin/env python3
"""Small real pinned FP4 PLE vs lazy storage, invalid IDs, streams and CUDA graph."""
import json
import tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader.qwen38_nvfp4_pack.convert import initialize_fixture
from sglang.srt.models.qwen38_fp4_ple import Qwen38Fp4PleEmbedding
from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index


def main():
    ctx, flags, parallel, backend, a2a = initialize_fixture()
    torch.manual_seed(791)
    prefix = 'model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.'
    weights = torch.randint(0,256,(128,80),dtype=torch.uint8)
    scales = torch.full((128,10),.5,dtype=torch.float8_e4m3fn)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'ple.safetensors'
        save_file({prefix+'weight':weights,prefix+'weight_scale':scales},path)
        desc=read_ple_checkpoint_index(str(path))
        embeddings=[]
        with ctx.override_server_args(model_path='dummy'), parallel.override(tp_size=1,tp_rank=0), torch.device('cuda'):
            for mode in ('pinned','lazy'):
                raw=VocabParallelEmbedding(128,160,params_dtype=torch.bfloat16)
                raw.register_buffer('weight_scale',torch.ones(1,device='cuda'))
                emb=Qwen38Fp4PleEmbedding(raw,backend=mode,checkpoint_rows=128)
                for kind in ('weight','weight_scale'):
                    emb.load_checkpoint_part(0,kind,desc[prefix+kind],1)
                emb.fp4_global_scale.fill_(.0123)
                emb.finalize_lazy();embeddings.append(emb)
        pin,lazy=embeddings
        assert pin.weight.is_pinned() and not pin._lazy and pin._lazy_cache is None
        assert pin.weight.shape==(128,90)
        torch.testing.assert_close(pin.weight[:,:80],weights,atol=0,rtol=0)
        torch.testing.assert_close(pin.weight[:,80:],scales.view(torch.uint8),atol=0,rtol=0)
        ids=torch.tensor([0,127,-1,128,42,0],device='cuda')
        expected=lazy.gather(ids)
        actual=pin.gather(ids)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        out=torch.empty_like(actual)
        assert pin.gather(ids,out=out) is out
        torch.testing.assert_close(out,expected,atol=0,rtol=0)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            cross=pin.gather(ids.flip(0))
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(cross,expected.flip(0),atol=0,rtol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=pin.gather(ids)
        ids.copy_(ids.flip(0));graph.replay()
        torch.testing.assert_close(captured,expected.flip(0),atol=0,rtol=0)
        lazy._lazy_cache.close()
    report={'status':'passed','pinned':True,'row_bytes':90,'parity_to_lazy':'exact','invalid_ids':True,'prefetch_output':True,'cross_stream':True,'cuda_graph_replay':True}
    output=Path('/workspace/sglang/tools/qwen38_cache_tests/fp4_local_20261017/pinned_ple.json');output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
