"""Give singleton experts wider native output tiles without wider pair arrays."""
from common import ROOT
p=ROOT/'python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py'
s=p.read_text()
a=s.index('if(SINGLES&&nmatch==1){');b=s.index('float a[RPS]',a)
block=s[a:b]
block=block.replace('float result[RPS]={0};','uint row=threadgroup_position_in_grid.y*16+sg*8;\n if(row>=N)return;\n float result[8]={0};').replace('r<RPS','r<8')
s=s[:a]+block+s[b:]
p.write_text(s)
