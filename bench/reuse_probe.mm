// reuse_probe.mm — does cross-gate blocking create reuse?
//
// bw_probe established that a single gate has no stride sensitivity: at any
// target qubit, one thread per amplitude pair reads two large contiguous runs
// and the memory system handles them equally well. So locality pressure in
// this workload is not *within* a gate. It is *across* gates — the whole state
// is re-read for every gate and nothing survives in cache between them.
//
// This tool tests the fix directly. Take K single-qubit gates that all target
// qubits below b, so every amplitude pair they touch stays inside a contiguous
// 2^b block. Then compare:
//
//   unblocked : K dispatches, each streaming the whole state.
//               traffic = K * 2^n * 16 bytes
//   blocked   : ONE dispatch. Each threadgroup stages a 2^b block into
//               threadgroup memory, applies all K gates there, writes back.
//               traffic = 2^n * 16 bytes, independent of K
//
// The ideal speedup is therefore K. What we actually get is the reuse effect,
// and it is the S0 decision gate: if it is flat, cache blocking is not in this
// architecture and the research question changes shape.
//
// Also measures empty-kernel dispatch cost, since the unblocked arm pays K
// launches and that must not be mistaken for a memory effect.
//
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework Foundation \
//       reuse_probe.mm -o reuse_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static NSString *const kSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Gate2 { float2 m00, m01, m10, m11; };

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void empty_kernel(device float2 *state [[buffer(0)]]) { }

kernel void init_state(device float2 *state [[buffer(0)]],
                       constant uint &grid_w [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    ulong t = (ulong)gid.y * (ulong)grid_w + (ulong)gid.x;
    // A spread-out, deterministic state. A basis state would leave most of the
    // buffer zero and let the compiler and memory system off too easily.
    float p = (float)(t & 1023u) * 0.0009765625f;
    state[t] = float2(p, 1.0f - p);
}

// Unblocked arm: one full-state pass per gate.
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

// Blocked arm: one threadgroup owns one contiguous 2^b block for the whole
// gate sequence. Every gate targets q < b, so all its pairs are in-block and
// the state vector is read once and written once no matter how large K is.
kernel void gates_blocked(device float2 *state [[buffer(0)]],
                          constant Gate2 *gates [[buffer(1)]],
                          constant uint *targets [[buffer(2)]],
                          constant uint &K [[buffer(3)]],
                          constant uint &b [[buffer(4)]],
                          threadgroup float2 *tile [[threadgroup(0)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tgsize [[threads_per_threadgroup]]) {
    uint block = 1u << b;
    ulong base = (ulong)tgid * (ulong)block;

    for (uint i = tid; i < block; i += tgsize) tile[i] = state[base + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pairs = block >> 1;
    for (uint k = 0; k < K; k++) {
        uint q = targets[k];
        uint stride = 1u << q;
        Gate2 g = gates[k];
        for (uint t = tid; t < pairs; t += tgsize) {
            uint i = ((t >> q) << (q + 1)) | (t & (stride - 1));
            uint j = i | stride;
            float2 a = tile[i];
            float2 c = tile[j];
            tile[i] = cmul(g.m00, a) + cmul(g.m01, c);
            tile[j] = cmul(g.m10, a) + cmul(g.m11, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < block; i += tgsize) state[base + i] = tile[i];
}
)METAL";

static const double kGiB = 1073741824.0;

struct Complex64 {
  float re, im;
};
struct Gate2 {
  Complex64 m00, m01, m10, m11;
};

struct Ctx {
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> empty_pso, init_pso, gate_pso, blocked_pso;
  id<MTLBuffer> state;
};

static void grid_for(uint64_t total, uint32_t *grid_w, uint64_t *grid_h) {
  uint64_t w = std::min<uint64_t>(total, 1ull << 26);
  *grid_w = (uint32_t)w;
  *grid_h = total / w;
}

// A rotation by a k-dependent angle: unitary, so repeated application cannot
// drift the norm, and distinct per gate so nothing collapses to identity.
static Gate2 gate_for(int k) {
  double th = 0.31 + 0.17 * k;
  float c = (float)cos(th), s = (float)sin(th);
  return Gate2{{c, 0.0f}, {-s, 0.0f}, {s, 0.0f}, {c, 0.0f}};
}

static double time_unblocked(Ctx &ctx, int n, int K, const std::vector<uint32_t> &targets) {
  uint32_t grid_w;
  uint64_t grid_h;
  grid_for(1ull << (n - 1), &grid_w, &grid_h);

  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  for (int k = 0; k < K; k++) {
    Gate2 g = gate_for(k);
    uint32_t q = targets[k];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.gate_pso];
    [enc setBuffer:ctx.state offset:0 atIndex:0];
    [enc setBytes:&g length:sizeof(g) atIndex:1];
    [enc setBytes:&q length:sizeof(q) atIndex:2];
    [enc setBytes:&grid_w length:sizeof(grid_w) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(grid_w, grid_h, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, grid_w), 1, 1)];
    [enc endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];
  return cb.GPUEndTime - cb.GPUStartTime;
}

static double time_blocked(Ctx &ctx, int n, int K, int b,
                           const std::vector<uint32_t> &targets) {
  std::vector<Gate2> gates(K);
  for (int k = 0; k < K; k++) gates[k] = gate_for(k);

  uint32_t block = 1u << b;
  uint64_t nblocks = 1ull << (n - b);
  NSUInteger T = std::min<NSUInteger>(256, block >> 1);
  uint32_t KK = (uint32_t)K, bb = (uint32_t)b;

  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.blocked_pso];
  [enc setBuffer:ctx.state offset:0 atIndex:0];
  [enc setBytes:gates.data() length:sizeof(Gate2) * K atIndex:1];
  [enc setBytes:targets.data() length:sizeof(uint32_t) * K atIndex:2];
  [enc setBytes:&KK length:sizeof(KK) atIndex:3];
  [enc setBytes:&bb length:sizeof(bb) atIndex:4];
  [enc setThreadgroupMemoryLength:block * sizeof(Complex64) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  return cb.GPUEndTime - cb.GPUStartTime;
}

static void reset_state(Ctx &ctx, int n) {
  uint32_t grid_w;
  uint64_t grid_h;
  grid_for(1ull << n, &grid_w, &grid_h);
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.init_pso];
  [enc setBuffer:ctx.state offset:0 atIndex:0];
  [enc setBytes:&grid_w length:sizeof(grid_w) atIndex:1];
  [enc dispatchThreads:MTLSizeMake(grid_w, grid_h, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, grid_w), 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

int main(int argc, char **argv) {
  @autoreleasepool {
    int n = (argc > 1) ? atoi(argv[1]) : 28;

    Ctx ctx{};
    ctx.dev = MTLCreateSystemDefaultDevice();
    ctx.queue = [ctx.dev newCommandQueue];

    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.dev newLibraryWithSource:kSource options:nil error:&err];
    if (!lib) {
      fprintf(stderr, "shader compile failed: %s\n", [[err description] UTF8String]);
      return 1;
    }
    auto pso = [&](NSString *name) {
      return [ctx.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name]
                                                   error:&err];
    };
    ctx.empty_pso = pso(@"empty_kernel");
    ctx.init_pso = pso(@"init_state");
    ctx.gate_pso = pso(@"gate1q_inplace");
    ctx.blocked_pso = pso(@"gates_blocked");
    if (!ctx.blocked_pso) {
      fprintf(stderr, "pipeline failed: %s\n", [[err description] UTF8String]);
      return 1;
    }

    printf("device                     : %s\n", [ctx.dev.name UTF8String]);
    printf("maxThreadgroupMemoryLength : %lu bytes\n",
           (unsigned long)ctx.dev.maxThreadgroupMemoryLength);
    printf("blocked-arm threadgroup occ: %lu threads max\n",
           (unsigned long)ctx.blocked_pso.maxTotalThreadsPerThreadgroup);
    printf("n                          : %d (%.2f GiB)\n", n,
           (double)(1ull << n) * 8.0 / kGiB);

    ctx.state = [ctx.dev newBufferWithLength:(1ull << n) * 8ull
                                     options:MTLResourceStorageModeShared];
    reset_state(ctx, n);

    // ---------------------------------------------------------------
    printf("\n=== 0. empty-kernel dispatch cost ===\n");
    {
      const int N = 200;
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      for (int i = 0; i < N; i++) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx.empty_pso];
        [enc setBuffer:ctx.state offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
      double per = (cb.GPUEndTime - cb.GPUStartTime) / N;
      printf("per dispatch: %.3f us\n", per * 1e6);
      printf("break-even  : a launch costs as much as %.0f MiB of traffic\n",
             per * 485e9 / (1024.0 * 1024.0));
    }

    // ---------------------------------------------------------------
    // Correctness first. A performance number from a wrong kernel is noise.
    printf("\n=== 1. correctness: blocked vs unblocked ===\n");
    {
      int nc = 20, Kc = 8, bc = 10;
      std::vector<uint32_t> targets(Kc);
      for (int k = 0; k < Kc; k++) targets[k] = k % bc;

      id<MTLBuffer> saved = ctx.state;
      ctx.state = [ctx.dev newBufferWithLength:(1ull << nc) * 8ull
                                       options:MTLResourceStorageModeShared];

      reset_state(ctx, nc);
      time_unblocked(ctx, nc, Kc, targets);
      std::vector<Complex64> ref((size_t)1 << nc);
      memcpy(ref.data(), ctx.state.contents, ref.size() * sizeof(Complex64));

      reset_state(ctx, nc);
      time_blocked(ctx, nc, Kc, bc, targets);
      const Complex64 *got = (const Complex64 *)ctx.state.contents;

      double maxdiff = 0.0;
      for (size_t i = 0; i < ref.size(); i++) {
        maxdiff = std::max(maxdiff, (double)std::hypot(ref[i].re - got[i].re,
                                                       ref[i].im - got[i].im));
      }
      printf("n=%d K=%d b=%d  max amplitude diff: %.3e  %s\n", nc, Kc, bc,
             maxdiff, maxdiff < 1e-5 ? "PASS" : "FAIL");
      ctx.state = saved;
      if (maxdiff >= 1e-5) return 1;
    }

    // ---------------------------------------------------------------
    printf("\n=== 2. reuse vs gate count (b=12) ===\n");
    printf("K,unblocked_s,blocked_s,speedup,ideal\n");
    {
      int b = 12;
      for (int K : {1, 2, 4, 8, 16, 32}) {
        std::vector<uint32_t> targets(K);
        for (int k = 0; k < K; k++) targets[k] = k % b;
        reset_state(ctx, n);
        time_unblocked(ctx, n, K, targets);  // warmup
        double tu = time_unblocked(ctx, n, K, targets);
        time_blocked(ctx, n, K, b, targets);
        double tb = time_blocked(ctx, n, K, b, targets);
        printf("%d,%.6f,%.6f,%.2f,%d\n", K, tu, tb, tu / tb, K);
        fflush(stdout);
      }
    }

    // ---------------------------------------------------------------
    printf("\n=== 3. reuse vs block size (K=16) ===\n");
    printf("b,block_amps,tg_mem_bytes,unblocked_s,blocked_s,speedup\n");
    {
      int K = 16;
      for (int b : {6, 8, 10, 12}) {
        std::vector<uint32_t> targets(K);
        for (int k = 0; k < K; k++) targets[k] = k % b;
        reset_state(ctx, n);
        time_unblocked(ctx, n, K, targets);
        double tu = time_unblocked(ctx, n, K, targets);
        time_blocked(ctx, n, K, b, targets);
        double tb = time_blocked(ctx, n, K, b, targets);
        printf("%d,%d,%d,%.6f,%.6f,%.2f\n", b, 1 << b, (1 << b) * 8, tu, tb,
               tu / tb);
        fflush(stdout);
      }
    }

    printf("\ndone\n");
    return 0;
  }
}
