// Private-process encoder-stage timestamps and pipeline inventory. No resources capture.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <atomic>
#include <mach/mach_time.h>
#include <mutex>

static id<MTLDevice> device;
static NSMutableArray* counterPages;
static NSMutableArray* passes;
static NSMutableDictionary* labels;
static std::mutex guard;
static std::atomic<bool> active{false};
static std::atomic<uint32_t> nextSample{0};
static MTLTimestamp cpu0,gpu0,cpu1,gpu1;
static char passKey,pipelineKey;
static id(*makeEncoder)(id,SEL,MTLDispatchType);
static void(*setPipeline)(id,SEL,id);
static void(*dispatchGroups)(id,SEL,MTLSize,MTLSize);
static void(*dispatchThreads)(id,SEL,MTLSize,MTLSize);

static id(*makePipeline)(id,SEL,id,NSError**) __attribute__((ns_returns_retained));
static id wrappedMakePipeline(id self,SEL sel,id<MTLFunction> f,NSError** error) __attribute__((ns_returns_retained));
static id wrappedMakePipeline(id self,SEL sel,id<MTLFunction> f,NSError** error){
    id result=makePipeline(self,sel,f,error);
    if(result){std::lock_guard<std::mutex> lock(guard);labels[[NSValue valueWithPointer:(__bridge const void*)result]]=f.name;}
    return result;
}

static id wrappedEncoder(id cb,SEL sel,MTLDispatchType type){
    if(!active)return makeEncoder(cb,sel,type);
    uint32_t index=nextSample.fetch_add(2);
    if(index+1>=counterPages.count*4096)return makeEncoder(cb,sel,type);
    MTLComputePassDescriptor* desc=[MTLComputePassDescriptor computePassDescriptor];
    desc.dispatchType=type;
    desc.sampleBufferAttachments[0].sampleBuffer=counterPages[index/4096];
    desc.sampleBufferAttachments[0].startOfEncoderSampleIndex=index%4096;
    desc.sampleBufferAttachments[0].endOfEncoderSampleIndex=index%4096+1;
    id enc=[cb computeCommandEncoderWithDescriptor:desc];
    NSMutableDictionary* record=[@{@"sample":@(index),@"ops":[NSMutableArray new]} mutableCopy];
    {std::lock_guard<std::mutex> lock(guard);[passes addObject:record];}
    objc_setAssociatedObject(enc,&passKey,record,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return enc;
}

static void wrappedPipeline(id enc,SEL sel,id<MTLComputePipelineState> p){
    setPipeline(enc,sel,p);
    NSString* name=p.label;
    if(!name){
        NSValue* key=[NSValue valueWithPointer:(__bridge const void*)p];
        std::lock_guard<std::mutex> lock(guard);
        name=labels[key];
        if(!name){name=p.description;labels[key]=name;}
    }
    objc_setAssociatedObject(enc,&pipelineKey,name,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

static void note(id enc,MTLSize grid,MTLSize threads,NSString* mode){
    if(!active)return;
    NSMutableDictionary* record=objc_getAssociatedObject(enc,&passKey);
    if(!record)return;
    NSString* name=objc_getAssociatedObject(enc,&pipelineKey)?:@"unknown";
    NSDictionary* op=@{@"name":name,@"mode":mode,@"grid":@[@(grid.width),@(grid.height),@(grid.depth)],@"threads":@[@(threads.width),@(threads.height),@(threads.depth)]};
    std::lock_guard<std::mutex> lock(guard);[record[@"ops"] addObject:op];
}

static void wrappedGroups(id enc,SEL sel,MTLSize grid,MTLSize threads){note(enc,grid,threads,@"groups");dispatchGroups(enc,sel,grid,threads);}
static void wrappedThreads(id enc,SEL sel,MTLSize grid,MTLSize threads){note(enc,grid,threads,@"threads");dispatchThreads(enc,sel,grid,threads);}

static IMP replace(Class cls,SEL selector,IMP impl){
    Method m=class_getInstanceMethod(cls,selector);IMP prior=method_getImplementation(m);
    if(!class_addMethod(cls,selector,impl,method_getTypeEncoding(m)))class_replaceMethod(cls,selector,impl,method_getTypeEncoding(m));
    return prior;
}

extern "C" int pipeline_install(){@autoreleasepool{
    if(device)return 0;
    device=MTLCreateSystemDefaultDevice();
    if(![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])return 1;
    MTLCounterSampleBufferDescriptor* desc=[MTLCounterSampleBufferDescriptor new];
    for(id<MTLCounterSet> set in device.counterSets)if([set.name isEqualToString:MTLCommonCounterSetTimestamp])desc.counterSet=set;
    desc.storageMode=MTLStorageModeShared;desc.sampleCount=4096;
    counterPages=[NSMutableArray new];
    for(int i=0;i<32;i++){NSError* error=nil;id buffer=[device newCounterSampleBufferWithDescriptor:desc error:&error];if(!buffer){NSLog(@"Counter page %d: %@",i,error);if(i==0)return 2;break;}[counterPages addObject:buffer];}
    passes=[NSMutableArray new];labels=[NSMutableDictionary new];
    makePipeline=(decltype(makePipeline))replace(object_getClass(device),@selector(newComputePipelineStateWithFunction:error:),(IMP)wrappedMakePipeline);
    id<MTLCommandBuffer> cb=[[device newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    Class ecls=object_getClass(enc),ccls=object_getClass(cb);
    makeEncoder=(id(*)(id,SEL,MTLDispatchType))replace(ccls,@selector(computeCommandEncoderWithDispatchType:),(IMP)wrappedEncoder);
    setPipeline=(void(*)(id,SEL,id))replace(ecls,@selector(setComputePipelineState:),(IMP)wrappedPipeline);
    dispatchGroups=(void(*)(id,SEL,MTLSize,MTLSize))replace(ecls,@selector(dispatchThreadgroups:threadsPerThreadgroup:),(IMP)wrappedGroups);
    dispatchThreads=(void(*)(id,SEL,MTLSize,MTLSize))replace(ecls,@selector(dispatchThreads:threadsPerThreadgroup:),(IMP)wrappedThreads);
    [enc endEncoding];return 0;
}}

extern "C" void pipeline_begin(){
    {std::lock_guard<std::mutex> lock(guard);[passes removeAllObjects];}
    nextSample=0;[device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];active=true;
}
extern "C" void pipeline_end(){active=false;[device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];}
extern "C" void pipeline_write(const char* path){@autoreleasepool{
    NSUInteger count=MIN(NSUInteger(nextSample.load()),counterPages.count*4096);
    NSMutableArray* data=[NSMutableArray new];
    for(NSUInteger page=0;page<(count+4095)/4096;page++)[data addObject:[counterPages[page] resolveCounterRange:NSMakeRange(0,MIN(4096,count-page*4096))]];
    std::lock_guard<std::mutex> lock(guard);
    for(NSMutableDictionary* pass in passes){NSUInteger i=[pass[@"sample"] unsignedIntegerValue];NSData* bytes=data[i/4096];const MTLCounterResultTimestamp* stamps=(const MTLCounterResultTimestamp*)bytes.bytes;NSUInteger j=i%4096;if((j+2)*sizeof(*stamps)<=bytes.length){pass[@"start"]=@(stamps[j].timestamp);pass[@"end"]=@(stamps[j+1].timestamp);}}
    mach_timebase_info_data_t clock;mach_timebase_info(&clock);
    NSDictionary* result=@{@"cpu0":@(cpu0),@"gpu0":@(gpu0),@"cpu1":@(cpu1),@"gpu1":@(gpu1),@"timestamp_frequency":@([device queryTimestampFrequency]),@"sample_count":@(count),@"cpu_numer":@(clock.numer),@"cpu_denom":@(clock.denom),@"passes":passes};
    [[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil]writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
}}
