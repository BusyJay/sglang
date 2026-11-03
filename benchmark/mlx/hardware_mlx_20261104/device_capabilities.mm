#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main() {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) return 2;
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"name"] = d.name;
    out[@"registry_id"] = @(d.registryID);
    out[@"unified_memory"] = @(d.hasUnifiedMemory);
    out[@"recommended_working_set_bytes"] = @(d.recommendedMaxWorkingSetSize);
    out[@"max_buffer_bytes"] = @(d.maxBufferLength);
    out[@"max_threadgroup_memory_bytes"] = @(d.maxThreadgroupMemoryLength);
    MTLSize threads = d.maxThreadsPerThreadgroup;
    out[@"max_threads_per_threadgroup"] = @[@(threads.width), @(threads.height), @(threads.depth)];
    NSMutableArray *families = [NSMutableArray array];
    for (NSUInteger family = 1001; family <= 1011; family++) {
      if ([d supportsFamily:(MTLGPUFamily)family]) [families addObject:@(family)];
    }
    out[@"supported_apple_gpu_families"] = families;
    NSMutableArray *sets = [NSMutableArray array];
    for (id<MTLCounterSet> set in d.counterSets) {
      NSMutableArray *counters = [NSMutableArray array];
      for (id<MTLCounter> counter in set.counters) [counters addObject:counter.name];
      [sets addObject:@{@"name": set.name, @"counters": counters}];
    }
    out[@"public_counter_sets"] = sets;
    out[@"counter_sampling"] = @{
      @"stage": @([d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),
      @"dispatch": @([d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),
      @"blit": @([d supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary])
    };
    out[@"scope"] = @"Device/API metadata only; no allocation, GPU dispatch or performance measurement";
    NSError *err = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingPrettyPrinted error:&err];
    if (!json) return 3;
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
  }
  return 0;
}
