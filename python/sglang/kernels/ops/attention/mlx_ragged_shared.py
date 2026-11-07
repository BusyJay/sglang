# SPDX-License-Identifier: Apache-2.0
"""Shared selected-history tiles, retaining MLX's MPP accumulation trees."""

NAX_SCORES = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint bank = sg / 4, part = sg % 4, h0 = bank * 32;
    const uint length = lengths[r];
    uint col = threadgroup_position_in_grid.x * 32;
    if (col >= length) {
        for (uint i = tid; i < 64 * 32; i += 256) {
            uint p = col + i % 32;
            if (p < WIDTH) scores[(size_t(r)*H+i/32)*WIDTH+p] = T(-INFINITY);
        }
        return;
    }
    if (col+32 > length && length >= 32) col = length-32;
    // Tile storage and partial sums have disjoint lifetimes. Reuse 24 KiB
    // rather than keeping both buffers resident at the same time.
    threadgroup float scratch[6*32*32];
    threadgroup T* tile = (threadgroup T*)scratch;
    threadgroup uint slots[32];
    if (tid < 32) {
        uint p=col+tid;
        slots[tid]=p<length ? indices[size_t(r) * WIDTH + p] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32,32,32,false,true,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
    auto a=op.template get_left_input_cooperative_tensor<T,T,float>();
    auto b=op.template get_right_input_cooperative_tensor<T,T,float>();
    auto c=op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
    for (uint i=0;i<c.get_capacity();++i)c[i]=0;
    for (uint base=0;base<D;base+=256) {
        for (uint i=tid;i<32*256/4;i+=256) {
            const uint p=col+i/64, k=base+(i%64)*4;
            const uint slot=slots[i/64];
            const vec<T,4> values=p<length ? vec<T,4>(LOAD_HISTORY) : vec<T,4>(0);
            for (uint j=0;j<4;++j)tile[(i/64)*256+(i%64)*4+j]=values[j];
            STASH_DECODED
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step=0;step<64;step+=32) {
            const uint k=base+part*64+step;
            for (uint i=0;i<a.get_capacity();++i) {
                auto coord=a.get_multidimensional_index(i);
                a[i]=queries[(size_t(r)*H+h0+coord[1])*D+k+coord[0]];
            }
            for (uint i=0;i<b.get_capacity();++i) {
                auto coord=b.get_multidimensional_index(i);
                b[i]=tile[coord[1]*256+part*64+step+coord[0]];
            }
            op.run(a,b,c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part) {
        for (uint i=0;i<c.get_capacity();++i) {
            auto coord=c.get_multidimensional_index(i);
            scratch[(bank*3+part-1)*1024+coord[1]*32+coord[0]]=c[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!part) {
        for (uint i=0;i<c.get_capacity();++i) {
            auto coord=c.get_multidimensional_index(i);
            float total=c[i];
            for (uint s=0;s<3;++s)total+=scratch[(bank*3+s)*1024+coord[1]*32+coord[0]];
            const uint p=col+coord[0];
            if (p>=threadgroup_position_in_grid.x*32 && p<length)
                scores[(size_t(r)*H+h0+coord[1])*WIDTH+p]=T(total);
        }
    }
"""

NAX_VALUES = r"""
    const uint sg=simdgroup_index_in_threadgroup;
    const uint tid=thread_position_in_threadgroup.x;
    const uint r=threadgroup_position_in_grid.y;
    const uint h0=(sg/2)*16;
    const uint col0=threadgroup_position_in_grid.x*64;
    const uint col=col0+(sg%2)*32;
    const uint length=lengths[r];
    threadgroup T history[16*64];
    threadgroup uint slots[16];
    constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(
        16,32,16,false,false,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
    auto a=op.template get_left_input_cooperative_tensor<T,T,float>();
    auto b=op.template get_right_input_cooperative_tensor<T,T,float>();
    auto total=op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
    auto c=total;
    for (uint i=0;i<total.get_capacity();++i)total[i]=0;
    uint partition_size=length>2048?2048:1024;
    uint partitions=length>1024?(length+partition_size-1)/partition_size:1;
    if(partitions==1)partition_size=length;
    for(uint part=0;part<partitions;++part) {
        for(uint i=0;i<c.get_capacity();++i)c[i]=0;
        uint end=min(length,(part+1)*partition_size);
        for(uint k=part*partition_size;k<end;k+=16) {
            if(tid<16) {
                const uint p=k+tid;
                slots[tid]=p<end ? indices[size_t(r) * WIDTH + p] : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint p=k+tid/16, cv=(tid%16)*4;
            const uint slot=slots[tid/16];
            const auto* row=VALUE_ROW;
            const vec<T,4> values=p<end ? vec<T,4>(*((const device vec<V,4>*)(row+col0+cv))) : vec<T,4>(0);
            for(uint j=0;j<4;++j)history[(tid/16)*64+cv+j]=values[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint i=0;i<a.get_capacity();++i) {
                auto coord=a.get_multidimensional_index(i);
                uint p=k+coord[0];
                a[i]=p<end ? probs[(size_t(r)*H+h0+coord[1])*WIDTH+p] : T(0);
            }
            for(uint i=0;i<b.get_capacity();++i) {
                auto coord=b.get_multidimensional_index(i);
                b[i]=history[coord[1]*64+(sg%2)*32+coord[0]];
            }
            op.run(a,b,c);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint i=0;i<c.get_capacity();++i)total[i]+=c[i];
    }
    for(uint i=0;i<total.get_capacity();++i) {
        auto coord=total.get_multidimensional_index(i);
        out[(size_t(r)*H+h0+coord[1])*D+col+coord[0]]=T(total[i]);
    }
"""

# Share the full common prefix of two implicit rows. Two SIMD-group
# banks keep their own accumulators and original partition boundaries.
NAX_PAIRED_VALUES = r"""
    const uint sg=simdgroup_index_in_threadgroup;
    const uint tid=thread_position_in_threadgroup.x;
    const uint first=threadgroup_position_in_grid.y*2;
    const uint position=sg/4, r=first+position, h0=(sg%4)*16;
    const uint col=threadgroup_position_in_grid.x*32;
    uint lengths_pair[2]={lengths[first], first+1<ROWS?lengths[first+1]:0};
    uint sizes[2], parts[2];
    for(uint pos=0;pos<2;++pos) {
        uint length=lengths_pair[pos];
        sizes[pos]=length>2048?2048:1024;
        parts[pos]=length>1024?(length+sizes[pos]-1)/sizes[pos]:1;
        if(parts[pos]==1)sizes[pos]=length;
    }
    threadgroup T history[16*32];
    constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(
        16,32,16,false,false,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
    auto a=op.template get_left_input_cooperative_tensor<T,T,float>();
    auto b=op.template get_right_input_cooperative_tensor<T,T,float>();
    auto total=op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
    auto c=total;
    for(uint i=0;i<total.get_capacity();++i)total[i]=0;
    for(uint part=0;part<max(parts[0],parts[1]);++part) {
        for(uint i=0;i<c.get_capacity();++i)c[i]=0;
        uint start[2]={part*sizes[0],part*sizes[1]};
        uint end[2]={min(lengths_pair[0],start[0]+sizes[0]),min(lengths_pair[1],start[1]+sizes[1])};
        uint shared_end = start[0];
        if(start[0]==start[1]) {
            uint end_min=min(end[0],end[1]);
            if(end_min>start[0]) shared_end=start[0]+((end_min-start[0])/16)*16;
        }
        for(uint k=start[0];k<shared_end;k+=16) {
            if(tid<128) {
                uint p=k+tid/8, cv=(tid%8)*4, source_r=first, slot=p;
                const auto* row=PAIRED_VALUE_ROW;
                const vec<T,4> values=vec<T,4>(*((const device vec<V,4>*)(row+col+cv)));
                for(uint j=0;j<4;++j)history[(tid/8)*32+cv+j]=values[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint i=0;i<a.get_capacity();++i) {
                auto coord=a.get_multidimensional_index(i);
                a[i]=probs[(size_t(r)*H+h0+coord[1])*WIDTH+k+coord[0]];
            }
            for(uint i=0;i<b.get_capacity();++i) {
                auto coord=b.get_multidimensional_index(i);
                b[i]=history[coord[1]*32+coord[0]];
            }
            op.run(a,b,c);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint source=0;source<2;++source) {
            uint begin=start[0]==start[1]?shared_end:start[source];
            for(uint k=begin;k<end[source];k+=16) {
                if(tid<128) {
                    uint p=k+tid/8, cv=(tid%8)*4, source_r=first+source, slot=p;
                    const auto* row=PAIRED_VALUE_ROW;
                    const vec<T,4> values=p<end[source] ? vec<T,4>(*((const device vec<V,4>*)(row+col+cv))) : vec<T,4>(0);
                    for(uint j=0;j<4;++j)history[(tid/8)*32+cv+j]=values[j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(position==source) {
                    for(uint i=0;i<a.get_capacity();++i) {
                        auto coord=a.get_multidimensional_index(i);
                        uint p=k+coord[0];
                        a[i]=p<end[source] ? probs[(size_t(r)*H+h0+coord[1])*WIDTH+p] : T(0);
                    }
                    for(uint i=0;i<b.get_capacity();++i) {
                        auto coord=b.get_multidimensional_index(i);
                        b[i]=history[coord[1]*32+coord[0]];
                    }
                    op.run(a,b,c);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if(part<parts[position])
            for(uint i=0;i<c.get_capacity();++i)total[i]+=c[i];
    }
    if(r<ROWS)for(uint i=0;i<total.get_capacity();++i) {
        auto coord=total.get_multidimensional_index(i);
        out[(size_t(r)*H+h0+coord[1])*D+col+coord[0]]=T(total[i]);
    }
"""
