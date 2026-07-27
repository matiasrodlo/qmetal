// zerocheck.mm — is the measured bandwidth inflated by all-zero data?
//
// bw_probe initialized the state to |0...0> and applied a real-valued gate.
// A gate maps zero amplitudes to zero amplitudes, so every timed pass in that
// tool ran over a buffer that stayed entirely zero. If Apple Silicon elides or
// compresses zero traffic anywhere in the path, the reported ~485 GB/s is an
// artifact rather than the real budget.
//
// Same kernel, same n, same traffic. Only the data differs.
//
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework Foundation \
//       zerocheck.mm -o zerocheck

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

static NSString *const kSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Gate2 { float2 m00, m01, m10, m11; };

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// mode 0: |0...0>  (all amplitudes zero except index 0)
// mode 1: dense non-zero
kernel void init_state(device float2 *state [[buffer(0)]],
                       constant uint &grid_w [[buffer(1)]],
                       constant uint &mode [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    ulong t = (ulong)gid.y * (ulong)grid_w + (ulong)gid.x;
    if (mode == 0u) {
        state[t] = float2(t == 0 ? 1.0f : 0.0f, 0.0f);
    } else {
        float p = (float)(t & 1023u) * 0.0009765625f + 0.001f;
        state[t] = float2(p, 1.0f - p);
    }
}

kernel void gate1q_inplace(device float2 *state [[buffer(0)]],
                           constant Gate2 &g [[buffer(1)]],
                           constant uint &q [[buffer(2)]],
                           constant uint &grid_w [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    ulong t = (ulong)gid.y * (ulong)grid_w + (ulong)gid.x;
    ulong stride = 1ul << q;
    ulong i = ((t >> q) << (q + 1)) | (t & (stride - 1));
    ulong j = i | stride;
    float2 a = state[i];
    float2 b = state[j];
    state[i] = cmul(g.m00, a) + cmul(g.m01, b);
    state[j] = cmul(g.m10, a) + cmul(g.m11, b);
}
)METAL";

struct Complex64 { float re, im; };
struct Gate2 { Complex64 m00, m01, m10, m11; };

static void grid_for(uint64_t total, uint32_t *w, uint64_t *h) {
  uint64_t ww = std::min<uint64_t>(total, 1ull << 26);
  *w = (uint32_t)ww;
  *h = total / ww;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
    if (!lib) {
      fprintf(stderr, "compile: %s\n", [[err description] UTF8String]);
      return 1;
    }
    id<MTLComputePipelineState> init_pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"init_state"]
                                           error:&err];
    id<MTLComputePipelineState> gate_pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gate1q_inplace"]
                                           error:&err];

    printf("device: %s\n\n", [dev.name UTF8String]);
    printf("n,data,seconds,GB/s\n");

    for (int n : {26, 28, 30}) {
      id<MTLBuffer> buf = [dev newBufferWithLength:(1ull << n) * 8ull
                                           options:MTLResourceStorageModeShared];
      for (uint32_t mode : {0u, 1u}) {
        uint32_t gw;
        uint64_t gh;

        // init
        grid_for(1ull << n, &gw, &gh);
        {
          id<MTLCommandBuffer> cb = [queue commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:init_pso];
          [e setBuffer:buf offset:0 atIndex:0];
          [e setBytes:&gw length:4 atIndex:1];
          [e setBytes:&mode length:4 atIndex:2];
          [e dispatchThreads:MTLSizeMake(gw, gh, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, gw), 1, 1)];
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
        }

        // timed gate, min of 3 after a warmup
        grid_for(1ull << (n - 1), &gw, &gh);
        Gate2 g = {{0.6f, 0.0f}, {0.8f, 0.0f}, {0.8f, 0.0f}, {-0.6f, 0.0f}};
        uint32_t q = 3;
        double best = 1e9;
        for (int r = 0; r < 4; r++) {
          id<MTLCommandBuffer> cb = [queue commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:gate_pso];
          [e setBuffer:buf offset:0 atIndex:0];
          [e setBytes:&g length:sizeof(g) atIndex:1];
          [e setBytes:&q length:4 atIndex:2];
          [e setBytes:&gw length:4 atIndex:3];
          [e dispatchThreads:MTLSizeMake(gw, gh, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, gw), 1, 1)];
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          if (r > 0) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
        }

        double bytes = (double)(1ull << n) * 16.0;
        printf("%d,%s,%.6f,%.1f\n", n, mode == 0 ? "all-zero" : "dense", best,
               bytes / best / 1e9);
        fflush(stdout);
      }
    }
    return 0;
  }
}
