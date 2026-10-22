// Diagnostic common harness. Links Strata's unchanged NVFP4 row implementation.
#include "/workspace/sglang/python/sglang/kernels/ops/moe/cpu_nvfp4/rows.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
struct Pool {
  std::atomic<unsigned> epoch{0};
  alignas(64) std::atomic<int> pending{0};
  alignas(64) std::atomic<bool> stop{false};
  std::function<void(int)> task;
  std::vector<std::thread> workers;
  Pool(int n,const int* cores) {
    for(int i=0;i<n;++i) workers.emplace_back([=,this] {
      cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cores[i],&mask);pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask);
      unsigned seen=0;
      while(!stop.load(std::memory_order_relaxed)) {
        unsigned e=epoch.load(std::memory_order_acquire);
        if(e==seen){_mm_pause();continue;}
        seen=e;task(i);pending.fetch_sub(1,std::memory_order_release);
      }
    });
  }
  void run(std::function<void(int)> f) {
    task=std::move(f);pending.store(workers.size(),std::memory_order_relaxed);epoch.fetch_add(1,std::memory_order_release);
    while(pending.load(std::memory_order_acquire)) _mm_pause();
  }
  ~Pool(){stop.store(true);for(auto& t:workers)t.join();}
};
extern "C" void* row_pool_create(int n,const int* cores){return new Pool(n,cores);}
extern "C" void row_pool_destroy(void* p){delete static_cast<Pool*>(p);}
template<int NT> void bf_rows(const uint8_t*w,const uint8_t*s,const uint16_t*a,float*out,int n,int rows,int lo,int hi) {
 static const a16global::Tables unit(1.f);const uint16_t* ap[NT];for(int t=0;t<NT;++t)ap[t]=a+t*n;
 for(int r=lo;r<hi;r+=8){int rr[8];for(int j=0;j<8;++j)rr[j]=std::min(r+j,rows-1);float v[8][NT];
   a16global::project<NT,8>(w,s,rr,ap,n,unit,v);
   for(int j=0;j<8&&r+j<hi;++j)for(int t=0;t<NT;++t)out[t*rows+r+j]=v[j][t];
 }
}
extern "C" void row_run(void*p,int mode,const uint8_t*w,const uint8_t*s,const uint8_t*g,const uint16_t*a,const uint8_t*q,float*out,int rows,int n,int nt) {
 auto&pool=*static_cast<Pool*>(p);int threads=pool.workers.size();const void* qa[3];float* op[3];
 for(int t=0;t<nt;++t){qa[t]=q+t*(n/32)*34;op[t]=out+t*rows;}
 pool.run([&](int id){int lo=rows*id/threads,hi=rows*(id+1)/threads;
   if(mode==0){if(nt==1)bf_rows<1>(w,s,a,out,n,rows,lo,hi);else if(nt==2)bf_rows<2>(w,s,a,out,n,rows,lo,hi);else bf_rows<3>(w,s,a,out,n,rows,lo,hi);}
   else strata::kernels::cpu::nvfp4_512_rows(g,(n/64)*36,n,qa,nt,op,lo,hi);
 });
}
extern "C" void flush_range(const void* p,size_t bytes){uintptr_t a=reinterpret_cast<uintptr_t>(p);for(uintptr_t k=a&~uintptr_t(63);k<a+bytes;k+=64)_mm_clflushopt(reinterpret_cast<void*>(k));_mm_sfence();}
