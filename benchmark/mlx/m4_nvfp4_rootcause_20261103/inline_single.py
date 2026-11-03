"""Inline native singleton arithmetic so MLX chooses each operand address space."""
from common import ROOT
from native_clone import native
path=ROOT/'python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py'
text=path.read_text()
start=text.index('if(SINGLES&&nmatch==1){')
end=text.index('float a[RPS]',start)
body=r'''if(SINGLES&&nmatch==1){
 float result[RPS]={0};
 for(int k=0;k<K;k+=512){
  float values[16];
  #pragma unroll
  for(int i=0;i<16;++i)values[i]=float(x[size_t(v0)*K+k+lane*16+i]);
  #pragma unroll
  for(int r=0;r<RPS;++r){
   const auto* w=(const device ushort*)(weight+(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2);
   uchar s=scales[(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16];
   float scale=float(*(thread fp8_e4m3*)(&s)),accum=0;
   #pragma unroll
   for(int j=0;j<4;++j){
    ushort packed=w[j];uchar c0=packed,c1=packed>>4,c2=packed>>8,c3=packed>>12;
    accum+=(values[4*j]*float(*(thread fp4_e2m1*)(&c0))+
            values[4*j+1]*float(*(thread fp4_e2m1*)(&c1))+
            values[4*j+2]*float(*(thread fp4_e2m1*)(&c2))+
            values[4*j+3]*float(*(thread fp4_e2m1*)(&c3)));
   }
   {
#pragma clang fp contract(off)
    result[r]+=scale*accum;
   }
  }
 }
 float global=global_scale[expert]/2688.0f;
 #pragma unroll
 for(int r=0;r<RPS;++r){
  float value=simd_sum(result[r]);
  if(lane==0)out[size_t(pick)*N+row+r]=O(value*global);
 }
 return;
}
'''
text=text[:start]+body+text[end:]
# Only decoders remain needed, no pointer-address-space assumptions.
from common import exact
start=text.index('_HEADER = r"""');end=text.index('_SOURCE =',start)
text=text[:start]+'_HEADER = r"""'+exact._HEADER+'"""\n\n'+text[end:]
path.write_text(text)
