from functools import cache
from pathlib import Path
import mlx.core as mx
import importlib
A=importlib.import_module('sglang.kernels.ops.gemm.mlx_affine_rows')
D=importlib.import_module('sglang.kernels.ops.moe.mlx_glm')
H=importlib.import_module('sglang.kernels.ops.gemm.mlx_dense_rows')
WORD='\ninline uint2 load_word_pair(const device uchar* base, size_t stride, size_t offset) {\n    const device uint* words=(const device uint*)base;\n    return uint2(words[(offset/4)*stride],words[(offset/4+1)*stride]);\n}\ntemplate <int VPT>\ninline float word_qdot4(uint2 packed, const thread float* x, float scale, float bias, float sum) {\n    float accum=0;\n    for(int i=0;i<VPT/4;++i){\n        uint ws=(i<2?packed.x:packed.y) >> ((i%2)*16);\n        accum+=(x[4*i]*(ws&0x000f)+x[4*i+1]*(ws&0x00f0)+x[4*i+2]*(ws&0x0f00)+x[4*i+3]*(ws&0xf000));\n    }\n    return scale*accum+sum*bias;\n}\n'

def gate_source(kind):
    source=A._GROUPED_GATE_UP
    if kind=='word':
        source=source.replace('uchar gp[2*BYTES], upack[2*BYTES];','uint2 gp=load_word_pair((const device uchar*)gw+gr,gw_strides[2],packed);\nuint2 upack=load_word_pair((const device uchar*)uw+ur,uw_strides[2],packed);')
        source=source.replace('glm_load_packed<2*BYTES>((const device uchar*)gw+gr,gw_strides[2],packed,gp);','').replace('glm_load_packed<2*BYTES>((const device uchar*)uw+ur,uw_strides[2],packed,upack);','')
        source=source.replace('glm_qdot_shared<VPT, BITS>(gp','word_qdot4<VPT>(gp').replace('glm_qdot_shared<VPT,BITS>(gp','word_qdot4<VPT>(gp')
        source=source.replace('glm_qdot_shared<VPT, BITS>(upack','word_qdot4<VPT>(upack').replace('glm_qdot_shared<VPT,BITS>(upack','word_qdot4<VPT>(upack')
    elif kind=='device':
        source=source.replace('glm_qdot_shared<VPT, BITS>(gp','glm_qdot<VPT, BITS>((const device uchar*)gw+gr+packed').replace('glm_qdot_shared<VPT,BITS>(gp','glm_qdot<VPT,BITS>((const device uchar*)gw+gr+packed')
        source=source.replace('glm_qdot_shared<VPT, BITS>(upack','glm_qdot<VPT, BITS>((const device uchar*)uw+ur+packed').replace('glm_qdot_shared<VPT,BITS>(upack','glm_qdot<VPT,BITS>((const device uchar*)uw+ur+packed')
    elif kind!='byte':raise ValueError(kind)
    return source
@cache
def gate_kernel(kind):
    return mx.fast.metal_kernel(name='review_gate_'+kind,input_names=['x','gw','gs','gb','uw','us','ub','indices','limit','grouped','counts'],output_names=['out'],source=gate_source(kind),header=A._SHARED_HEADER+A._PACKED_LOAD+WORD,ensure_row_contiguous=False)
def gate(x,ga,ua,ids,*,kind='word',grouped=None,rps=2,limit=7.,group_size=64):
    count,k=x.shape[1:];n=ga[0].shape[1];topk=ids.shape[-1]
    grouped=D._group_picks(ids,min(3,count)) if grouped is None else grouped
    return (coeff_gate_kernel() if kind=='coeff' else mask_gate_kernel() if kind=='masks' else gate_kernel(kind))(inputs=[x,*ga,*ua,ids,mx.array([limit or 0.],mx.float32),*grouped],template=[('T',x.dtype),('K',k),('N',n),('GS',group_size),('BITS',4),('TOPK',topk),('COUNT',count),('TILE',min(3,count)),('RPS',rps),('HAS_LIMIT',limit is not None)],grid=(128*(n//(4*rps)),count*topk,1),threadgroup=(128,1,1),output_shapes=[(1,count,topk,n)],output_dtypes=[x.dtype])[0]
DOWN_SOURCE='\n    const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;\n    const uint row=threadgroup_position_in_grid.x*4;\n    const uint start_pick=threadgroup_position_in_grid.y*TOPK*TILE;\n    const uint end_pick=min(start_pick+uint(TOPK*TILE),uint(COUNT*TOPK));\n    threadgroup float weighted[TOPK*TILE*4];\n    for(uint first=start_pick+sg;first<end_pick;first+=SGS){\n    const uint expert=indices[first];\n    const uint matches=counts[first];\n    const device uint* picks=grouped+first*(TOPK*TILE);\n    constexpr int PACK=glm_pack_factor<BITS>(),BYTES=glm_bytes_per_pack<BITS>();\n    constexpr int VPT=2*PACK,BLOCK=32*VPT;\n    for(uint start=0;start<matches;start+=TILE){\n        const uint rows=min(uint(TILE),matches-start);\nif(rows==1){\n        float value[1][4]={0};\n        for(uint k=lane*VPT;k<K;k+=BLOCK){\n            float input[1][VPT],sum[1];\n            for(uint t=0;t<1;++t)\n                sum[t]=glm_load_vector<T,VPT,BITS>(act+size_t(picks[start+t])*K+k,input[t]);\n            for(uint j=0;j<4;++j){\n                const size_t base=size_t(expert)*w_strides[0]*4+size_t(row+j)*w_strides[1]*4;\n                const size_t offset=k*BYTES/PACK;\n                uchar packed[2*BYTES];\n                if(w_strides[2]==1 && rows>1){\n                    const device uchar* ptr=(const device uchar*)w+base+offset;\n                    for(uint i=0;i<2*BYTES;++i)packed[i]=ptr[i];\n                }else if(w_strides[2]!=1){\n                    for(uint i=0;i<2*BYTES;++i)\n                        packed[i]=((const device uchar*)w)[base+((offset+i)/4)*w_strides[2]*4+(offset+i)%4];\n                }\n                const float scale=s[size_t(expert)*s_strides[0]+size_t(row+j)*s_strides[1]+size_t(k/GS)*s_strides[2]];\n                const float bias=b[size_t(expert)*b_strides[0]+size_t(row+j)*b_strides[1]+size_t(k/GS)*b_strides[2]];\n                for(uint t=0;t<1;++t)\n                    value[t][j]+=(w_strides[2]==1 && rows==1)\n                        ? glm_qdot<VPT,BITS>((const device uint8_t*)w+base+offset,input[t],scale,bias,sum[t])\n                        : glm_qdot_shared<VPT,BITS>(packed,input[t],scale,bias,sum[t]);\n            }\n        }\n        for(uint t=0;t<1;++t)\n            for(uint j=0;j<4;++j){\n                const T rounded=T(simd_sum(value[t][j]));\n                volatile float product=float(R(rounded))*scores[picks[start+t]];\n                if(lane==0)weighted[(picks[start+t]-start_pick)*4+j]=product;\n            }\n}\nelse if(rows==2){\n        float value[2][4]={0};\n        for(uint k=lane*VPT;k<K;k+=BLOCK){\n            float input[2][VPT],sum[2];\n            for(uint t=0;t<2;++t)\n                sum[t]=glm_load_vector<T,VPT,BITS>(act+size_t(picks[start+t])*K+k,input[t]);\n            for(uint j=0;j<4;++j){\n                const size_t base=size_t(expert)*w_strides[0]*4+size_t(row+j)*w_strides[1]*4;\n                const size_t offset=k*BYTES/PACK;\n                uchar packed[2*BYTES];\n                if(w_strides[2]==1 && rows>1){\n                    const device uchar* ptr=(const device uchar*)w+base+offset;\n                    for(uint i=0;i<2*BYTES;++i)packed[i]=ptr[i];\n                }else if(w_strides[2]!=1){\n                    for(uint i=0;i<2*BYTES;++i)\n                        packed[i]=((const device uchar*)w)[base+((offset+i)/4)*w_strides[2]*4+(offset+i)%4];\n                }\n                const float scale=s[size_t(expert)*s_strides[0]+size_t(row+j)*s_strides[1]+size_t(k/GS)*s_strides[2]];\n                const float bias=b[size_t(expert)*b_strides[0]+size_t(row+j)*b_strides[1]+size_t(k/GS)*b_strides[2]];\n                for(uint t=0;t<2;++t)\n                    value[t][j]+=(w_strides[2]==1 && rows==1)\n                        ? glm_qdot<VPT,BITS>((const device uint8_t*)w+base+offset,input[t],scale,bias,sum[t])\n                        : glm_qdot_shared<VPT,BITS>(packed,input[t],scale,bias,sum[t]);\n            }\n        }\n        for(uint t=0;t<2;++t)\n            for(uint j=0;j<4;++j){\n                const T rounded=T(simd_sum(value[t][j]));\n                volatile float product=float(R(rounded))*scores[picks[start+t]];\n                if(lane==0)weighted[(picks[start+t]-start_pick)*4+j]=product;\n            }\n}\nelse if(rows==3){\n        float value[3][4]={0};\n        for(uint k=lane*VPT;k<K;k+=BLOCK){\n            float input[3][VPT],sum[3];\n            for(uint t=0;t<3;++t)\n                sum[t]=glm_load_vector<T,VPT,BITS>(act+size_t(picks[start+t])*K+k,input[t]);\n            for(uint j=0;j<4;++j){\n                const size_t base=size_t(expert)*w_strides[0]*4+size_t(row+j)*w_strides[1]*4;\n                const size_t offset=k*BYTES/PACK;\n                uchar packed[2*BYTES];\n                if(w_strides[2]==1 && rows>1){\n                    const device uchar* ptr=(const device uchar*)w+base+offset;\n                    for(uint i=0;i<2*BYTES;++i)packed[i]=ptr[i];\n                }else if(w_strides[2]!=1){\n                    for(uint i=0;i<2*BYTES;++i)\n                        packed[i]=((const device uchar*)w)[base+((offset+i)/4)*w_strides[2]*4+(offset+i)%4];\n                }\n                const float scale=s[size_t(expert)*s_strides[0]+size_t(row+j)*s_strides[1]+size_t(k/GS)*s_strides[2]];\n                const float bias=b[size_t(expert)*b_strides[0]+size_t(row+j)*b_strides[1]+size_t(k/GS)*b_strides[2]];\n                for(uint t=0;t<3;++t)\n                    value[t][j]+=(w_strides[2]==1 && rows==1)\n                        ? glm_qdot<VPT,BITS>((const device uint8_t*)w+base+offset,input[t],scale,bias,sum[t])\n                        : glm_qdot_shared<VPT,BITS>(packed,input[t],scale,bias,sum[t]);\n            }\n        }\n        for(uint t=0;t<3;++t)\n            for(uint j=0;j<4;++j){\n                const T rounded=T(simd_sum(value[t][j]));\n                volatile float product=float(R(rounded))*scores[picks[start+t]];\n                if(lane==0)weighted[(picks[start+t]-start_pick)*4+j]=product;\n            }\n}\n\n    }\n\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    if(sg==0 && lane<TILE*4){\n        const uint tok=threadgroup_position_in_grid.y*TILE+lane/4;\n        const uint col=lane%4;\n        if(tok<COUNT){\n            float total=0;\n            for(uint p=0;p<TOPK;++p)total+=weighted[((lane/4)*TOPK+p)*4+col];\n            out[size_t(tok)*N+row+col]=R(total)+shared[size_t(tok)*N+row+col];\n        }\n    }\n'

def down_source(kind):
    source=DOWN_SOURCE
    if kind=='device':
        start=source.index('                uchar packed[2*BYTES];')
        end=source.index('                const float scale=',start)
        body=source[start:end]
        source=source.replace(body,'')
        start=source.index('value[t][j]+=')
        end=source.index(';',start)+1
        old=source[start:end]
        source=source.replace(old,'value[t][j]+=glm_qdot<VPT,BITS>((const device uint8_t*)w+base+offset,input[t],scale,bias,sum[t]);')
    elif kind=='word':
        start=source.index('                uchar packed[2*BYTES];');end=source.index('                const float scale=',start)
        body=source[start:end]
        source=source.replace(body,'                uint2 packed=load_word_pair((const device uchar*)w+base,w_strides[2],offset);\n')
        start=source.index('value[t][j]+=');end=source.index(';',start)+1;old=source[start:end]
        source=source.replace(old,'value[t][j]+=word_qdot4<VPT>(packed,input[t],scale,bias,sum[t]);')
    elif kind=='threadgroup':
        # Each SIMD owns a private packed lane tile; SIMD barriers retain
        # independent expert progress. No device partial or inverse buffer.
        source=source.replace('threadgroup float weighted[TOPK*TILE*4];','threadgroup float weighted[TOPK*TILE*4];\nthreadgroup uchar packed_tiles[SGS*32*16];')
        start=source.index('                uchar packed[2*BYTES];');end=source.index('                const float scale=',start)
        body=source[start:end]
        new="""                threadgroup uchar* tile=packed_tiles+(sg*32+lane)*16;
                for(uint i=0;i<2*BYTES;++i)tile[i]=((const device uchar*)w)[base+((offset+i)/4)*w_strides[2]*4+(offset+i)%4];
                simdgroup_barrier(mem_flags::mem_threadgroup);
"""
        source=source.replace(body,new)
        start=source.index('value[t][j]+=');end=source.index(';',start)+1;old=source[start:end]
        source=source.replace(old,'value[t][j]+=tg_qdot<VPT,BITS>(tile,input[t],scale,bias,sum[t]);')
    elif kind!='byte':raise ValueError(kind)
    return source
@cache
def down_kernel(kind):
    shared=D._HEADER[D._HEADER.index('template <int values_per_thread, int bits>'):D._HEADER.index('// qmv_fast_impl')].replace('glm_qdot(','glm_qdot_shared(').replace('const device','const thread')
    tg=D._HEADER[D._HEADER.index('template <int values_per_thread, int bits>'):D._HEADER.index('// qmv_fast_impl')].replace('glm_qdot(','tg_qdot(').replace('const device','const threadgroup')
    return mx.fast.metal_kernel(name='review_down_'+kind,input_names=['act','indices','w','s','b','scores','shared','grouped','counts'],output_names=['out'],source=down_source(kind),header=D._HEADER+shared+tg+WORD,ensure_row_contiguous=False)
def down(act,ids,scores,w,s,b,shared,*,bits,kind='word',grouped=None,simds=8,group_size=64):
    count,topk,k=act.shape[1:];n=w.shape[1];tile=min(3,count)
    if grouped is None:grouped=D._group_picks(ids,tile)
    return (fast_down_kernel('word') if kind=='masks' else down_kernel(kind))(inputs=[act,ids,w,s,b,scores,shared,*grouped],template=[('T',act.dtype),('R',shared.dtype),('K',k),('N',n),('TOPK',topk),('BITS',bits),('GS',group_size),('COUNT',count),('TILE',tile),('SGS',simds)],grid=(32*simds*(n//4),(count+tile-1)//tile,1),threadgroup=(32*simds,1,1),output_shapes=[shared.shape],output_dtypes=[shared.dtype])[0]
@cache
def head_kernel(vector):
    import inspect
    from types import SimpleNamespace
    ns=dict(vars(H))
    ns['mx']=SimpleNamespace(fast=SimpleNamespace(metal_kernel=lambda **kw:kw))
    factory=inspect.getsource(H._argmax_rows_kernel.__wrapped__).replace('def _argmax_rows_kernel():','def make():')
    exec(factory,ns)
    kw=ns['make']()
    if vector:
        old="for (int i = 0; i < 4; ++i)\n                values[i] = row + r < N ? float(w[size_t(row + r) * K + k + i]) : 0.0f;"
        new="float4 packed_w = row+r<N ? float4(*((const device vec<O,4>*)(w+size_t(row+r)*K+k))) : float4(0);\n            for(int i=0;i<4;++i)values[i]=packed_w[i];"
        assert old in kw['source'], 'vector loading replacement must change shader'
        kw['source']=kw['source'].replace(old,new)
    kw['name']='review_head_'+str(vector)
    return mx.fast.metal_kernel(**kw)
def head(x,w,*,bm=8,rps=4,vector=False):
    count,k=x.shape[1:];n=w.shape[0];rows=min(count,4);blocks=(n+bm*rps-1)//(bm*rps)
    scores,ids=head_kernel(vector)(inputs=[x,w],template=[('O',x.dtype),('K',k),('N',n),('BM',bm),('RPS',rps),('ROWS',rows),('COUNT',count),('TAIL',count%rows!=0),('BLOCKS',blocks)],grid=(32*bm*blocks,1,(count+rows-1)//rows),threadgroup=(32*bm,1,1),output_shapes=[(count,blocks),(count,blocks)],output_dtypes=[x.dtype,mx.uint32])
    return mx.take_along_axis(ids,mx.argmax(scores,axis=-1,keepdims=True),axis=-1).reshape(1,count)

# Unpack each packed lane exactly once, reuse mask coefficients across vectors.
MASK = r"""
inline uint4 load_masks4(const device uchar* base, size_t stride, size_t offset){
    const device uint* ptr=(const device uint*)base;
    uint a=ptr[(offset/4)*stride],b=ptr[(offset/4+1)*stride];
    return uint4(a&65535,a>>16,b&65535,b>>16);
}
inline float mask_qdot4(uint4 ws,const thread float* x,float scale,float bias,float sum){
    float acc=0;
    for(int i=0;i<4;++i)acc+=(x[4*i]*(ws[i]&0xf)+x[4*i+1]*(ws[i]&0xf0)+x[4*i+2]*(ws[i]&0xf00)+x[4*i+3]*(ws[i]&0xf000));
    return scale*acc+sum*bias;
}
"""
@cache
def mask_gate_kernel():
    source=gate_source('word').replace('uint2 gp=load_word_pair','uint4 gp=load_masks4').replace('uint2 upack=load_word_pair','uint4 upack=load_masks4').replace('word_qdot4<VPT>','mask_qdot4')
    return mx.fast.metal_kernel(name='review_gate_masks',input_names=['x','gw','gs','gb','uw','us','ub','indices','limit','grouped','counts'],output_names=['out'],source=source,header=A._SHARED_HEADER+A._PACKED_LOAD+MASK,ensure_row_contiguous=False)
@cache
def fast_down_kernel(kind):
    source=down_source(kind)
    # Compile-time specialize matched row count, avoiding dynamic loops and
    # keeping native device qdot for singleton experts.
    if kind=='word':source=source.replace('uint2 packed=load_word_pair','uint4 packed=load_masks4').replace('word_qdot4<VPT>','mask_qdot4')
    shared=D._HEADER[D._HEADER.index('template <int values_per_thread, int bits>'):D._HEADER.index('// qmv_fast_impl')].replace('glm_qdot(','glm_qdot_shared(').replace('const device','const thread')
    return mx.fast.metal_kernel(name='review_down_masks_'+kind,input_names=['act','indices','w','s','b','scores','shared','grouped','counts'],output_names=['out'],source=source,header=D._HEADER+shared+MASK,ensure_row_contiguous=False)

# Parallel per-pick qdots retain original four-output projection parallelism.
# Packed lane tiles are shared only where an expert repeats across tokens.
PARALLEL_DOWN = r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint row=threadgroup_position_in_grid.x*4;
const uint start=threadgroup_position_in_grid.y*TOPK*TILE;
const uint pick=min(start+sg,uint(COUNT*TOPK-1));
const uint expert=indices[pick];
uint owner=pick;
for(uint p=start;p<pick;++p)if(indices[p]==expert){owner=p;break;}
threadgroup uchar packed_tiles[TOPK*TILE*32*16];
threadgroup float weighted[TOPK*TILE*4];
float values[4]={0};
constexpr int PACK=glm_pack_factor<BITS>(),BYTES=glm_bytes_per_pack<BITS>(),VPT=2*PACK,BLOCK=32*VPT;
for(uint k=lane*VPT;k<K;k+=BLOCK){
 float input[VPT];
 const float sum=glm_load_vector<T,VPT,BITS>(act+size_t(pick)*K+k,input);
 for(uint j=0;j<4;++j){
  const size_t base=size_t(expert)*w_strides[0]*4+size_t(row+j)*w_strides[1]*4;
  const size_t offset=k*BYTES/PACK;
  threadgroup uchar* dst=packed_tiles+(sg*32+lane)*16;
  if(pick==owner){for(uint i=0;i<2*BYTES;++i)dst[i]=((const device uchar*)w)[base+((offset+i)/4)*w_strides[2]*4+(offset+i)%4];}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale=s[size_t(expert)*s_strides[0]+size_t(row+j)*s_strides[1]+size_t(k/GS)*s_strides[2]];
  const float bias=b[size_t(expert)*b_strides[0]+size_t(row+j)*b_strides[1]+size_t(k/GS)*b_strides[2]];
  values[j]+=tg_qdot<VPT,BITS>(packed_tiles+((owner-start)*32+lane)*16,input,scale,bias,sum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
for(uint j=0;j<4;++j){
 T rounded=T(simd_sum(values[j]));volatile float product=float(R(rounded))*scores[pick];
 if(lane==0)weighted[sg*4+j]=product;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg%TOPK==0 && lane<4 && start+sg<COUNT*TOPK){
 float total=0;for(uint p=0;p<TOPK;++p)total+=weighted[(sg+p)*4+lane];
 uint tok=(start+sg)/TOPK;out[size_t(tok)*N+row+lane]=R(total)+shared[size_t(tok)*N+row+lane];
}
"""
@cache
def parallel_down_kernel(kind):
    tg=D._HEADER[D._HEADER.index('template <int values_per_thread, int bits>'):D._HEADER.index('// qmv_fast_impl')].replace('glm_qdot(','tg_qdot(').replace('const device','const threadgroup')
    source=PARALLEL_DOWN
    if kind=='word':
        source=source.replace('values[j]+=tg_qdot<VPT,BITS>(packed_tiles+((owner-start)*32+lane)*16,input,scale,bias,sum);','values[j]+=word_qdot4<VPT>(load_word_pair((const device uchar*)w+base,w_strides[2],offset),input,scale,bias,sum);')
        # A control with identical projection parallelism but no barriers/shared loads.
        a=source.index('  threadgroup uchar* dst=');b=source.index('  const float scale=',a)
        source=source[:a]+source[b:]
        source=source.replace('  threadgroup_barrier(mem_flags::mem_threadgroup);','')
    return mx.fast.metal_kernel(name='review_parallel_down_'+kind,input_names=['act','indices','w','s','b','scores','shared'],output_names=['out'],source=source,header=D._HEADER+tg+WORD,ensure_row_contiguous=False)
def parallel_down(a,i,sc,w,s,b,sh,*,bits,kind='tg',tile=3):
    count,topk,k=a.shape[1:];n=w.shape[1];threads=32*topk*tile
    return parallel_down_kernel(kind)(inputs=[a,i,w,s,b,sc,sh],template=[('T',a.dtype),('R',sh.dtype),('K',k),('N',n),('TOPK',topk),('BITS',bits),('GS',64),('COUNT',count),('TILE',tile)],grid=(threads*(n//4),(count+tile-1)//tile,1),threadgroup=(threads,1,1),output_shapes=[sh.shape],output_dtypes=[sh.dtype])[0]

COEFF = r"""
template <int VPT>
inline float coeff_qdot4(const thread float* c,const thread float* x,float scale,float bias,float sum){
 float accum=0;
 for(int i=0;i<VPT/4;++i)accum+=(x[4*i]*c[4*i]+x[4*i+1]*c[4*i+1]+x[4*i+2]*c[4*i+2]+x[4*i+3]*c[4*i+3]);
 return scale*accum+sum*bias;
}
inline void unpack_coeff4(uint2 w,thread float* c){
 for(int i=0;i<4;++i){uint q=(i<2?w.x:w.y)>>((i%2)*16);c[4*i]=q&0xf;c[4*i+1]=q&0xf0;c[4*i+2]=q&0xf00;c[4*i+3]=q&0xf000;}
}
"""
@cache
def coeff_gate_kernel():
 source=gate_source('word')
 source=source.replace('uint2 gp=load_word_pair((const device uchar*)gw+gr,gw_strides[2],packed);','float gp[VPT];unpack_coeff4(load_word_pair((const device uchar*)gw+gr,gw_strides[2],packed),gp);')
 source=source.replace('uint2 upack=load_word_pair((const device uchar*)uw+ur,uw_strides[2],packed);','float upack[VPT];unpack_coeff4(load_word_pair((const device uchar*)uw+ur,uw_strides[2],packed),upack);')
 source=source.replace('word_qdot4<VPT>','coeff_qdot4<VPT>')
 return mx.fast.metal_kernel(name='review_gate_coeff',input_names=['x','gw','gs','gb','uw','us','ub','indices','limit','grouped','counts'],output_names=['out'],source=source,header=A._SHARED_HEADER+A._PACKED_LOAD+WORD+COEFF,ensure_row_contiguous=False)

COMPACT_GROUP = r"""
const uint first=thread_position_in_grid.x;
if(first>=COUNT*TOPK)return;
const uint begin=(first/(TOPK*TILE))*(TOPK*TILE),end=min(begin+uint(TOPK*TILE),uint(COUNT*TOPK));
const uint expert=indices[first];bool owner=true;
for(uint p=begin;p<first;++p)if(indices[p]==expert)owner=false;
uint n=0;
if(owner)for(uint p=first;p<end;++p)if(indices[p]==expert)picks[first*(TOPK*TILE)+n++]=p;
counts[first]=n;
uint offset=0;
for(uint p=0;p<first;++p){
 uint b=(p/(TOPK*TILE))*(TOPK*TILE);bool unique=true;
 for(uint q=b;q<p;++q)if(indices[q]==indices[p])unique=false;
 offset+=unique;
}
if(owner)compact[offset]=first;
if(first==COUNT*TOPK-1)total[0]=offset+uint(owner);
"""
@cache
def compact_group_kernel():
 return mx.fast.metal_kernel(name='review_compact_groups',input_names=['indices'],output_names=['picks','counts','compact','total'],source=COMPACT_GROUP)
def compact_group(ids,tile=3):
 count,topk=ids.shape[1:]
 return compact_group_kernel()(inputs=[ids],template=[('COUNT',count),('TOPK',topk),('TILE',tile)],grid=(count*topk,1,1),threadgroup=(32,1,1),output_shapes=[(count*topk,topk*tile),(count*topk,),(count*topk,),(1,)],output_dtypes=[mx.uint32]*4)
@cache
def compact_gate_kernel():
 source=mask_gate_kernel().source if False else gate_source('word').replace('uint2 gp=load_word_pair','uint4 gp=load_masks4').replace('uint2 upack=load_word_pair','uint4 upack=load_masks4').replace('word_qdot4<VPT>','mask_qdot4')
 source=source.replace('const uint first = threadgroup_position_in_grid.y;', 'if(threadgroup_position_in_grid.y>=total[0])return;\nconst uint first=compact[threadgroup_position_in_grid.y];')
 return mx.fast.metal_kernel(name='review_compact_gate',input_names=['x','gw','gs','gb','uw','us','ub','indices','limit','grouped','counts','compact','total'],output_names=['out'],source=source,header=A._SHARED_HEADER+A._PACKED_LOAD+MASK,ensure_row_contiguous=False)
def compact_gate(x,g,u,i,meta,*,limit=10.,group_size=64):
 count,k=x.shape[1:];n=g[0].shape[1];topk=i.shape[-1]
 return compact_gate_kernel()(inputs=[x,*g,*u,i,mx.array([limit],mx.float32),*meta],template=[('T',x.dtype),('K',k),('N',n),('GS',group_size),('BITS',4),('TOPK',topk),('COUNT',count),('TILE',min(3,count)),('RPS',2),('HAS_LIMIT',True)],grid=(128*(n//8),count*topk,1),threadgroup=(128,1,1),output_shapes=[(1,count,topk,n)],output_dtypes=[x.dtype])[0]
