// Standalone Metal submission probe, portable to another macOS host by copying
// this file and compiling there. No MLX headers/libs, model, or downloads.
// clang-format off
/* Build from repository root, using the existing macOS SDK:
xcrun clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra -Werror \
  -isysroot "$(xcrun --show-sdk-path)" -mmacosx-version-min=11.0 \
  benchmark/mlx/ar_optimization_20261103/icb_probe.mm \
  -framework Metal -framework Foundation \
  -o benchmark/mlx/ar_optimization_20261103/icb_probe
For host syntax checking, replace the framework/output arguments with -fsyntax-only.
Metal source is compiled at runtime; a host syntax check alone does not check it.
*/
// clang-format on
// Run: ./benchmark/mlx/ar_optimization_20261103/icb_probe [warm_rounds=64]
// One command buffer/encoder per sample; all paths use concurrent encoders,
// 127 dependency barriers, tracked shared buffers, and identical arithmetic.
// ICBs and two parity argument records are CPU-encoded once, never rebuilt.
// CPU submission ends after commit; completion includes wait, excludes reset
// and verification. GPU time is command-buffer GPUStartTime..GPUEndTime, not
// kernel-only counter time. First execution is after library/pipeline setup,
// and first paths necessarily share device/driver warming. Order rotates and
// reverses each round. Stable scratch addresses, changing inputs, no overlap
// between samples. This measures replay of a fixed schedule, not graph capture,
// dynamic shapes, model throughput, or an MPSGraph implementation.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr NSUInteger kSteps = 128;
static constexpr NSUInteger kWidth = 4096;
static constexpr NSUInteger kNarrow = 256;
static constexpr MTLResourceOptions kOptions =
    MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
static void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
static std::string description(NSError *error) {
  return error ? std::string(error.localizedDescription.UTF8String)
               : "no NSError provided";
}

static const char *kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint n; uint k; float bias; float scale; };
struct Args {
  device const float *x [[id(0)]];
  device float *y [[id(1)]];
  device const uchar *w [[id(2)]];
  constant Params *p [[id(3)]];
};
inline float e4m3(uchar code) {
  uint mag = uint(code) & 127;
  if (mag == 127) return NAN;
  uint exponent = mag >> 3, mantissa = mag & 7;
  float v = exponent == 0 ? float(mantissa) * 0.001953125f
      : as_type<float>(((exponent + 120) << 23) | (mantissa << 20));
  return code & 128 ? -v : v;
}
inline float point(float x, uint i, constant Params &p) {
  return x * p.scale + p.bias * float(1 + (i & 7));
}
kernel void vector_direct(device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]], device const uchar *w [[buffer(2)]],
    constant Params &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < p.n) y[i] = point(x[i], i, p);
}
kernel void vector_args(constant Args &a [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
  if (i < a.p->n) a.y[i] = point(a.x[i], i, *a.p);
}
// One SIMD group per row, decoded E4M3 with uniform synthetic weight scale.
// Explicit residual-free bounded epilogue avoids growth/underflow in 128 layers.
// This is a decode + reduction workload, not a tuned production FP8 GEMV.
inline void gemv(device const float *x, device float *y,
    device const uchar *w, constant Params &p, uint row, uint lane) {
  if (row >= p.n) return;
  float sum = 0;
  for (uint col = lane; col < p.k; col += 32)
    sum += (e4m3(w[row * p.k + col]) * p.scale) * x[col];
  sum = simd_sum(sum);
  if (lane == 0) y[row] = p.bias + 0.125f * tanh(sum);
}
kernel void gemv_direct(device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]], device const uchar *w [[buffer(2)]],
    constant Params &p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) { gemv(x, y, w, p, row, lane); }
kernel void gemv_args(constant Args &a [[buffer(0)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  gemv(a.x, a.y, a.w, *a.p, row, lane);
}
)METAL";

struct Params {
  uint32_t n, k;
  float bias, scale;
};
struct Reference {
  std::vector<float> scratch[2];
};
struct Case {
  bool gemv;
  id<MTLBuffer> scratch[2], weights[2], params[2], arguments[2];
  id<MTLComputePipelineState> pipelines[2];
  id<MTLIndirectCommandBuffer> icbs[2];
  Params p[2];
  Reference references[3];
  double pipelineMs = 0, argumentMs = 0, icbMs[2] = {};
  NSUInteger allocatedBytes = 0;
};

static id<MTLBuffer> buffer(id<MTLDevice> device, NSUInteger bytes, Case &c) {
  id<MTLBuffer> result = [device newBufferWithLength:bytes options:kOptions];
  require(result != nil, "buffer allocation failed");
  require(result.hazardTrackingMode == MTLHazardTrackingModeTracked,
          "buffer is not tracked");
  c.allocatedBytes += result.allocatedSize;
  return result;
}
static float input(NSUInteger i, int seed) {
  return float(int((i * 17 + NSUInteger(seed) * 31) % 257) - 128) / 256.0f;
}
static float decode(uint8_t code) {
  unsigned mag = code & 127, exp = mag >> 3, mant = mag & 7;
  float v = exp == 0 ? float(mant) / 512.0f
                     : std::ldexp(1.0f + float(mant) / 8.0f, int(exp) - 7);
  return code & 128 ? -v : v;
}
static void makeReferences(Case &c) {
  for (int seed = 0; seed < 3; ++seed) {
    Reference &r = c.references[seed];
    for (auto &s : r.scratch)
      s.assign(kWidth, 0.0f);
    for (NSUInteger i = 0; i < kWidth; ++i)
      r.scratch[0][i] = input(i, seed);
    for (NSUInteger step = 0; step < kSteps; ++step) {
      const NSUInteger parity = step & 1;
      const auto &x = r.scratch[parity];
      auto &y = r.scratch[parity ^ 1];
      const Params &p = c.p[parity];
      const uint8_t *w =
          static_cast<const uint8_t *>(c.weights[parity].contents);
      for (NSUInteger row = 0; row < p.n; ++row) {
        if (!c.gemv) {
          y[row] = x[row] * p.scale + p.bias * float(1 + (row & 7));
        } else {
          double sum = 0;
          for (NSUInteger col = 0; col < p.k; ++col)
            sum +=
                double(decode(w[row * p.k + col]) * p.scale) * double(x[col]);
          y[row] = float(double(p.bias) + 0.125 * std::tanh(sum));
        }
      }
    }
  }
}
static MTLSize groups(const Case &c, NSUInteger parity) {
  return MTLSizeMake(c.gemv ? c.p[parity].n : (c.p[parity].n + 255) / 256, 1,
                     1);
}
static MTLSize threads(const Case &c) {
  return MTLSizeMake(c.gemv ? 32 : 256, 1, 1);
}

static void setup(Case &c, id<MTLDevice> device, id<MTLLibrary> library) {
  auto start = Clock::now();
  id<MTLFunction> functions[2];
  for (NSUInteger ab = 0; ab < 2; ++ab) {
    NSString *name = c.gemv ? (ab ? @"gemv_args" : @"gemv_direct")
                            : (ab ? @"vector_args" : @"vector_direct");
    functions[ab] = [library newFunctionWithName:name];
    require(functions[ab] != nil, "missing Metal function");
    MTLComputePipelineDescriptor *descriptor =
        [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = functions[ab];
    descriptor.supportIndirectCommandBuffers = YES;
    NSError *error = nil;
    c.pipelines[ab] =
        [device newComputePipelineStateWithDescriptor:descriptor
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&error];
    require(c.pipelines[ab] != nil, "pipeline creation: " + description(error));
    require(c.pipelines[ab].supportIndirectCommandBuffers,
            "pipeline did not enable ICB");
    require(c.pipelines[ab].maxTotalThreadsPerThreadgroup >= threads(c).width,
            "requested threadgroup exceeds pipeline limit");
    if (c.gemv)
      require(c.pipelines[ab].threadExecutionWidth == 32,
              "GEMV probe requires SIMD width 32");
  }
  c.pipelineMs = ms(start);
  for (NSUInteger parity = 0; parity < 2; ++parity) {
    c.p[parity] =
        c.gemv
            ? Params{uint32_t(parity ? kWidth : kNarrow),
                     uint32_t(parity ? kNarrow : kWidth), 0.5f, 1.0f / 128.0f}
            : Params{uint32_t(kWidth), uint32_t(kWidth),
                     parity ? -0.0009765625f : 0.001953125f, 0.9990234375f};
    c.scratch[parity] = buffer(device, kWidth * sizeof(float), c);
    NSUInteger weightBytes = c.gemv ? kWidth * kNarrow : 1;
    c.weights[parity] = buffer(device, weightBytes, c);
    auto *w = static_cast<uint8_t *>(c.weights[parity].contents);
    uint32_t state = 0x12345678u + uint32_t(parity);
    for (NSUInteger i = 0; i < weightBytes; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      // Finite E4M3 magnitudes [0, 1.875], signed; no altered checkpoint.
      w[i] = uint8_t((state & 0x80) | ((state >> 8) % 64));
    }
    c.params[parity] = buffer(device, sizeof(Params), c);
    std::memcpy(c.params[parity].contents, &c.p[parity], sizeof(Params));
  }
  start = Clock::now();
  id<MTLArgumentEncoder> args =
      [functions[1] newArgumentEncoderWithBufferIndex:0];
  require(args != nil, "argument encoder creation failed");
  for (NSUInteger parity = 0; parity < 2; ++parity) {
    c.arguments[parity] = buffer(device, args.encodedLength, c);
    [args setArgumentBuffer:c.arguments[parity] offset:0];
    [args setBuffer:c.scratch[parity] offset:0 atIndex:0];
    [args setBuffer:c.scratch[parity ^ 1] offset:0 atIndex:1];
    [args setBuffer:c.weights[parity] offset:0 atIndex:2];
    [args setBuffer:c.params[parity] offset:0 atIndex:3];
  }
  c.argumentMs = ms(start);
  for (NSUInteger ab = 0; ab < 2; ++ab) {
    start = Clock::now();
    MTLIndirectCommandBufferDescriptor *descriptor =
        [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritBuffers = NO;
    descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = ab ? 1 : 4;
    c.icbs[ab] = [device newIndirectCommandBufferWithDescriptor:descriptor
                                                maxCommandCount:kSteps
                                                        options:kOptions];
    require(c.icbs[ab] != nil, "ICB creation failed");
    for (NSUInteger step = 0; step < kSteps; ++step) {
      NSUInteger parity = step & 1;
      id<MTLIndirectComputeCommand> command =
          [c.icbs[ab] indirectComputeCommandAtIndex:step];
      [command setComputePipelineState:c.pipelines[ab]];
      if (ab) {
        [command setKernelBuffer:c.arguments[parity] offset:0 atIndex:0];
      } else {
        [command setKernelBuffer:c.scratch[parity] offset:0 atIndex:0];
        [command setKernelBuffer:c.scratch[parity ^ 1] offset:0 atIndex:1];
        [command setKernelBuffer:c.weights[parity] offset:0 atIndex:2];
        [command setKernelBuffer:c.params[parity] offset:0 atIndex:3];
      }
      [command concurrentDispatchThreadgroups:groups(c, parity)
                        threadsPerThreadgroup:threads(c)];
      // The ICB barrier precedes this command (unlike encoder barriers
      // emitted after a direct dispatch). Include commands 1..127;
      // omitting the final dependency caused intermittent scratch errors
      // in the local M4 Pro repeat test. No relaxed correctness threshold.
      if (step > 0)
        [command setBarrier];
    }
    c.icbMs[ab] = ms(start);
    c.allocatedBytes += c.icbs[ab].allocatedSize;
  }
  makeReferences(c);
}

static NSDictionary *sample(Case &c, id<MTLCommandQueue> queue, NSUInteger mode,
                            NSString *phase, int round, NSUInteger position,
                            int seed) {
  // Host writes only after previous sample completed, before any encoding.
  auto resetStart = Clock::now();
  for (NSUInteger b = 0; b < 2; ++b)
    std::memset(c.scratch[b].contents, 0, kWidth * sizeof(float));
  auto *x = static_cast<float *>(c.scratch[0].contents);
  for (NSUInteger i = 0; i < kWidth; ++i)
    x[i] = input(i, seed);
  double resetMs = ms(resetStart);
  const NSUInteger ab = mode & 1;
  const bool indirect = mode >= 2;
  auto start = Clock::now();
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  require(cb != nil, "command buffer creation failed");
  id<MTLComputeCommandEncoder> encoder =
      [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  require(encoder != nil, "compute encoder creation failed");
  // Declare every indirectly referenced leaf resource, plus argument storage.
  // Identical declarations in direct paths keep residency overhead comparable.
  for (NSUInteger b = 0; b < 2; ++b) {
    [encoder useResource:c.scratch[b]
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:c.weights[b] usage:MTLResourceUsageRead];
    [encoder useResource:c.params[b] usage:MTLResourceUsageRead];
    if (ab)
      [encoder useResource:c.arguments[b] usage:MTLResourceUsageRead];
  }
  if (indirect) {
    [encoder useResource:c.icbs[ab] usage:MTLResourceUsageRead];
    [encoder executeCommandsInBuffer:c.icbs[ab]
                           withRange:NSMakeRange(0, kSteps)];
  } else {
    [encoder setComputePipelineState:c.pipelines[ab]];
    for (NSUInteger step = 0; step < kSteps; ++step) {
      NSUInteger parity = step & 1;
      if (ab) {
        [encoder setBuffer:c.arguments[parity] offset:0 atIndex:0];
      } else {
        [encoder setBuffer:c.scratch[parity] offset:0 atIndex:0];
        [encoder setBuffer:c.scratch[parity ^ 1] offset:0 atIndex:1];
        [encoder setBuffer:c.weights[parity] offset:0 atIndex:2];
        [encoder setBuffer:c.params[parity] offset:0 atIndex:3];
      }
      [encoder dispatchThreadgroups:groups(c, parity)
              threadsPerThreadgroup:threads(c)];
      if (step + 1 < kSteps)
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }
  [encoder endEncoding];
  double encodeMs = ms(start);
  auto commitStart = Clock::now();
  [cb commit];
  double commitMs = ms(commitStart), submitMs = ms(start);
  [cb waitUntilCompleted];
  double completionMs = ms(start);
  require(cb.status == MTLCommandBufferStatusCompleted,
          "GPU command failed: " + description(cb.error));
  double gpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1000;
  require(cb.GPUStartTime > 0 && gpuMs > 0 && std::isfinite(gpuMs),
          "invalid GPU timestamps");
  auto verifyStart = Clock::now();
  double maxAbs = 0, maxRel = 0, checksum = 0;
  NSUInteger failures = 0;
  // Independent double-accumulation reference for GEMV, FP32 for pointwise.
  const double atol = c.gemv ? 2e-5 : 1e-6, rtol = c.gemv ? 2e-5 : 1e-6;
  for (NSUInteger b = 0; b < 2; ++b) {
    const auto *actual = static_cast<const float *>(c.scratch[b].contents);
    const auto &expected = c.references[seed].scratch[b];
    for (NSUInteger i = 0; i < kWidth; ++i) {
      double error = std::abs(double(actual[i]) - double(expected[i]));
      maxAbs = std::max(maxAbs, error);
      maxRel = std::max(maxRel,
                        error / std::max(1e-12, std::abs(double(expected[i]))));
      if (!std::isfinite(actual[i]) ||
          error > atol + rtol * std::abs(double(expected[i])))
        ++failures;
      checksum += double(actual[i]) * double(1 + i % 13);
    }
  }
  NSArray *names = @[
    @"direct", @"direct_argument_buffer", @"icb_preencoded",
    @"icb_argument_buffer"
  ];
  return @{
    @"mode" : names[mode],
    @"phase" : phase,
    @"round" : @(round),
    @"order_position" : @(position),
    @"seed" : @(seed),
    @"reset_ms" : @(resetMs),
    @"cpu_encode_ms" : @(encodeMs),
    @"cpu_commit_ms" : @(commitMs),
    @"cpu_submission_ms" : @(submitMs),
    @"cpu_completion_ms" : @(completionMs),
    @"gpu_elapsed_ms" : @(gpuMs),
    @"verification_ms" : @(ms(verifyStart)),
    @"correct" : @(failures == 0),
    @"mismatches" : @(failures),
    @"max_abs_error" : @(maxAbs),
    @"max_relative_error" : @(maxRel),
    @"checksum_both_scratch" : @(checksum),
    @"output_head" : @[ @(x[0]), @(x[1]), @(x[2]), @(x[3]) ],
    @"atol" : @(atol),
    @"rtol" : @(rtol)
  };
}

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  return n & 1 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
static NSDictionary *runCase(bool gemv, id<MTLDevice> device,
                             id<MTLLibrary> library, id<MTLCommandQueue> queue,
                             int rounds, bool &correct) {
  Case c{};
  c.gemv = gemv;
  auto start = Clock::now();
  setup(c, device, library);
  double setupMs = ms(start);
  NSMutableArray *samples = [NSMutableArray array];
  for (int round = -3; round < rounds; ++round) {
    int ordinal = round + 3;
    NSString *phase =
        round == -3 ? @"first" : (round < 0 ? @"warmup" : @"warm");
    int seed = ordinal % 3;
    for (NSUInteger position = 0; position < 4; ++position) {
      @autoreleasepool {
        NSUInteger mode = (NSUInteger(ordinal / 2) +
                           (ordinal & 1 ? 3 - position : position)) %
                          4;
        NSDictionary *s = sample(c, queue, mode, phase, round, position, seed);
        correct &= [s[@"correct"] boolValue];
        [samples addObject:s];
      }
    }
  }
  NSMutableDictionary *summary = [NSMutableDictionary dictionary];
  for (NSString *mode in @[
         @"direct", @"direct_argument_buffer", @"icb_preencoded",
         @"icb_argument_buffer"
       ]) {
    NSMutableDictionary *fields = [NSMutableDictionary dictionary];
    for (NSString *field in
         @[ @"cpu_submission_ms", @"gpu_elapsed_ms", @"cpu_completion_ms" ]) {
      std::vector<double> values;
      for (NSDictionary *s in samples)
        if ([s[@"mode"] isEqual:mode] && [s[@"phase"] isEqual:@"warm"])
          values.push_back([s[field] doubleValue]);
      fields[field] = @(median(values));
    }
    summary[mode] = fields;
  }
  return @{
    @"name" : gemv ? @"decoded_e4m3_gemv_chain" : @"vector_pointwise_chain",
    @"dispatches" : @(kSteps),
    @"barriers" : @(kSteps - 1),
    @"dimensions" : gemv
        ? @[ @[ @(kNarrow), @(kWidth) ], @[ @(kWidth), @(kNarrow) ] ]
        : @[ @(kWidth) ],
    @"weight_bytes" : @(gemv ? 2 * kWidth * kNarrow : 2),
    @"resource_allocated_bytes_including_icbs" : @(c.allocatedBytes),
    @"setup_including_cpu_reference_ms" : @(setupMs),
    @"pipeline_setup_ms" : @(c.pipelineMs),
    @"argument_buffer_setup_ms" : @(c.argumentMs),
    @"icb_preencode_ms" : @[ @(c.icbMs[0]), @(c.icbMs[1]) ],
    @"warm_medians_ms" : summary,
    @"samples" : samples
  };
}

static void emit(NSDictionary *result) {
  NSError *error = nil;
  NSData *json = [NSJSONSerialization
      dataWithJSONObject:result
                 options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                   error:&error];
  require(json != nil, "JSON serialization: " + description(error));
  std::fwrite(json.bytes, 1, json.length, stdout);
  std::fputc('\n', stdout);
}
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      auto start = Clock::now();
      int rounds = 64;
      if (argc == 2) {
        char *end = nullptr;
        long value = std::strtol(argv[1], &end, 10);
        require(end != argv[1] && *end == '\0' && value >= 1 && value <= 512,
                "warm_rounds must be an integer in [1,512]");
        rounds = int(value);
      }
      require(argc <= 2, "usage: icb_probe [warm_rounds=64]");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      require(device != nil, "MTLCreateSystemDefaultDevice returned nil");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      require(queue != nil, "command queue creation failed");
      auto libraryStart = Clock::now();
      MTLCompileOptions *options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError *error = nil;
      id<MTLLibrary> library =
          [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                               options:options
                                 error:&error];
      require(library != nil,
              "Metal source compilation: " + description(error));
      double libraryMs = ms(libraryStart);
      bool correct = true;
      NSMutableArray *cases = [NSMutableArray array];
      [cases addObject:runCase(false, device, library, queue, rounds, correct)];
      [cases addObject:runCase(true, device, library, queue, rounds, correct)];
      emit(@{
        @"schema_version" : @1,
        @"device" : device.name,
        @"os" : NSProcessInfo.processInfo.operatingSystemVersionString,
        @"physical_memory_bytes" : @(NSProcessInfo.processInfo.physicalMemory),
        @"argument_buffer_tier" : @(device.argumentBuffersSupport),
        @"recommended_max_working_set_bytes" :
            @(device.recommendedMaxWorkingSetSize),
        @"warm_rounds" : @(rounds),
        @"warmup_rounds" : @2,
        @"metal_debug_layer_requested" :
            @(std::getenv("MTL_DEBUG_LAYER") != nullptr),
        @"library_compile_ms" : @(libraryMs),
        @"total_elapsed_ms" : @(ms(start)),
        @"all_correct" : @(correct),
        @"cases" : cases,
        @"scope" : @"Local fixed-schedule microbenchmark; synthetic FP8 "
                   @"weights; no model or MPSGraph",
        @"timing" : @"CPU encode includes command-buffer/encoder creation and "
                    @"resource declarations; submission includes commit; "
                    @"completion includes wait; GPU elapsed is command-buffer "
                    @"interval; reset and correctness excluded",
        @"ordering" : @"Four modes rotate and reverse each round; first is "
                      @"post-setup and shares prior device warming",
        @"resources" : @"Shared tracked buffers; explicit leaf declarations; "
                       @"stable ping-pong scratch; 127 explicit barriers; CPU "
                       @"reset only after completion"
      });
      return correct ? 0 : 2;
    } catch (const std::exception &e) {
      emit(@{
        @"all_correct" : @NO,
        @"error" : [NSString stringWithUTF8String:e.what()]
      });
      return 1;
    }
  }
}
