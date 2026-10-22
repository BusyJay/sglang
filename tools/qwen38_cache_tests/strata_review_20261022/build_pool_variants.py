"""Diagnostic copies only: compare current sleeping worker pool with bounded spin.
Widen the native test-only core bound to 16; production code remains unchanged.
"""
from pathlib import Path
import subprocess
src=Path('/workspace/sglang/python/sglang/kernels/ops/moe/cpu_nvfp4');root=Path('/results')
original=(src/'kernel.cpp').read_text().replace('threads > 4','threads > 16')
append=r'''
extern "C" int bench_batch(const uint8_t *payload,uint64_t bytes,const uint64_t *offsets,int experts,int h,int i,const uint16_t*x,const int32_t*ids,const float*weights,int nt,int topk,int threads,const int*cores,int nc,float*out){
 return run(payload,bytes,offsets,experts,h,i,x,ids,weights,nt,topk,threads,cores,nc,out,true,nullptr,nullptr,0,true);
}
extern "C" void bench_noop(const int*cores,int nc,int iterations,int gap_us,double*out){
 auto&w=pool(cores,nc);
 for(int k=0;k<iterations;++k){
  if(gap_us) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
  auto t=std::chrono::steady_clock::now();w.dispatch([](int){});
  out[k]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
 }
}
'''
for mode in ['cv','spin']:
 d=root/('pool-'+mode);d.mkdir(exist_ok=True)
 for name in ['rows.hpp','base.hpp']:(d/name).write_bytes((src/name).read_bytes())
 text=original
 if mode=='spin':
  text=text.replace('size_t generation = 0;','std::atomic<size_t> generation{0};').replace('int pending = 0, ready = 0;','std::atomic<int> pending{0};\n  int ready = 0;').replace('bool stopping = false;','std::atomic<bool> stopping{false};')
  needle='          work.wait(lock, [&] { return stopping || generation != seen; });'
  assert needle in text
  text=text.replace(needle,'''          lock.unlock();
          const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
          unsigned polls=0;
          while (!stopping.load(std::memory_order_relaxed) && generation.load(std::memory_order_acquire)==seen) {
            _mm_pause();
            if ((++polls & 1023)==0 && std::chrono::steady_clock::now()>=deadline) break;
          }
          lock.lock();
'''+needle)
  needle='    finished.wait(lock, [&] { return pending == 0; });'
  assert needle in text
  text=text.replace(needle,'''    lock.unlock();
    while (pending.load(std::memory_order_acquire)!=0) _mm_pause();
    lock.lock();''')
 (d/'kernel.cpp').write_text(text+append)
 subprocess.run(['g++','-O3','-std=c++17','-mavx512f','-mavx512bw','-mavx512vl','-mavx512bf16','-fPIC','-shared','-pthread',str(d/'kernel.cpp'),'-o',str(d/'kernel.so')],check=True)
 print('built',mode,flush=True)
