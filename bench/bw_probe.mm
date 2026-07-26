// bw_probe.mm — memory-system characterization for statevector simulation on
// Apple Silicon.
//
// Answers three questions the architecture depends on:
//
//   1. What streaming bandwidth does a Metal kernel actually reach, and does it
//      hold as the working set grows past every cache boundary up to 64 GiB?
//   2. Where is the stride/locality cliff? An in-place 1-qubit gate on target
//      qubit q touches amplitude pairs 2^q apart. Sweeping q at fixed n walks
//      the access stride from 1 element to half the state, crossing L2 and the
//      SLC on the way. The knee locates the cache the way `sysctl` will not.
//   3. Does a 64 GiB working set stay resident under sustained load, or does
//      macOS start compressing?
//
// Conventions (pinned, see README): LSB-first, qubit q has stride 1 << q.
// complex64 amplitudes as float2. All index math is 64-bit — at n = 33 a
// 1-qubit gate needs 2^32 threads, which overflows a 32-bit grid index.
//
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework Foundation \
//       bw_probe.mm -o bw_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

static NSString *const kSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Gate2 { float2 m00, m01, m10, m11; };

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Flatten a 2-D grid position into a 64-bit linear index. grid_w is a power of
// two, so this is exact for every dispatch this tool issues.
static inline ulong lin(uint2 gid, uint grid_w) {
    return (ulong)gid.y * (ulong)grid_w + (ulong)gid.x;
}

// Write |0...0>. Also serves as the first-touch pass that commits pages for a
// lazily-allocated shared buffer.
kernel void init_state(device float2 *state [[buffer(0)]],
                       constant uint &grid_w [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    state[t] = float2(t == 0 ? 1.0f : 0.0f, 0.0f);
}

// Pure sequential streaming: one read + one write per amplitude, no strided
// access. This is the bandwidth ceiling any gate kernel is measured against.
kernel void stream_scale(device float2 *dst [[buffer(0)]],
                         const device float2 *src [[buffer(1)]],
                         constant float2 &s [[buffer(2)]],
                         constant uint &grid_w [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    dst[t] = cmul(src[t], s);
}

// In-place single-qubit gate. One thread owns the amplitude pair (i, j) with
// j = i + 2^q. Total traffic is identical to stream_scale — one read and one
// write per amplitude — so any difference is pure access-pattern cost.
kernel void gate1q_inplace(device float2 *state [[buffer(0)]],
                           constant Gate2 &g [[buffer(1)]],
                           constant uint &q [[buffer(2)]],
                           constant uint &grid_w [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    ulong stride = 1ul << q;
    ulong i = ((t >> q) << (q + 1)) | (t & (stride - 1));
    ulong j = i | stride;
    float2 a = state[i];
    float2 b = state[j];
    state[i] = cmul(g.m00, a) + cmul(g.m01, b);
    state[j] = cmul(g.m10, a) + cmul(g.m11, b);
}
)METAL";

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

static const double kGiB = 1073741824.0;

// Plain aggregates rather than raw arrays: blocks cannot capture array types,
// and these mirror the kernel-side layouts exactly.
struct Complex64 {
  float re, im;
};
struct Gate2 {
  Complex64 m00, m01, m10, m11;
};

struct Ctx {
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> init_pso;
  id<MTLComputePipelineState> stream_pso;
  id<MTLComputePipelineState> gate_pso;
  id<MTLBuffer> a;  // primary state buffer
  id<MTLBuffer> b;  // destination for the out-of-place stream test
};

// Split `total` threads into a 2-D grid so the x extent never exceeds 2^26.
// `total` is always a power of two here.
static void grid_for(uint64_t total, uint32_t *grid_w, uint64_t *grid_h) {
  uint64_t w = std::min<uint64_t>(total, 1ull << 26);
  *grid_w = (uint32_t)w;
  *grid_h = total / w;
}

// Encode one dispatch and return GPU-side elapsed seconds.
static double time_dispatch(Ctx &ctx, id<MTLComputePipelineState> pso,
                            uint64_t total_threads,
                            void (^bind)(id<MTLComputeCommandEncoder>, uint32_t)) {
  uint32_t grid_w;
  uint64_t grid_h;
  grid_for(total_threads, &grid_w, &grid_h);

  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  bind(enc, grid_w);

  NSUInteger tg = std::min<NSUInteger>(256, grid_w);
  [enc dispatchThreads:MTLSizeMake(grid_w, grid_h, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  if (cb.error) {
    fprintf(stderr, "command buffer error: %s\n",
            [[cb.error description] UTF8String]);
    exit(1);
  }
  return cb.GPUEndTime - cb.GPUStartTime;
}

static double run_stream(Ctx &ctx, int n, int reps) {
  uint64_t total = 1ull << n;
  Complex64 s = {0.99999f, 0.00001f};
  double best = 1e9;
  for (int r = 0; r < reps + 1; r++) {
    double t = time_dispatch(ctx, ctx.stream_pso, total,
                             ^(id<MTLComputeCommandEncoder> enc, uint32_t gw) {
                               [enc setBuffer:ctx.b offset:0 atIndex:0];
                               [enc setBuffer:ctx.a offset:0 atIndex:1];
                               [enc setBytes:&s length:sizeof(s) atIndex:2];
                               [enc setBytes:&gw length:sizeof(gw) atIndex:3];
                             });
    if (r > 0) best = std::min(best, t);  // discard warmup
  }
  return best;
}

static double run_gate(Ctx &ctx, int n, int q, int reps) {
  uint64_t total = 1ull << (n - 1);
  // A real orthogonal 2x2 with no zero entries, so no arithmetic path is
  // skipped and the kernel stays norm-preserving across repeated application.
  Gate2 g = {{0.6f, 0.0f}, {0.8f, 0.0f}, {0.8f, 0.0f}, {-0.6f, 0.0f}};
  uint32_t qq = (uint32_t)q;
  double best = 1e9;
  // reps == 0 means "one timed run, no warmup discard" — what the residency
  // test wants, since it is looking for drift rather than a best case.
  int first_measured = (reps == 0) ? 0 : 1;
  for (int r = 0; r < reps + 1; r++) {
    double t = time_dispatch(ctx, ctx.gate_pso, total,
                             ^(id<MTLComputeCommandEncoder> enc, uint32_t gw) {
                               [enc setBuffer:ctx.a offset:0 atIndex:0];
                               [enc setBytes:&g length:sizeof(g) atIndex:1];
                               [enc setBytes:&qq length:sizeof(qq) atIndex:2];
                               [enc setBytes:&gw length:sizeof(gw) atIndex:3];
                             });
    if (r >= first_measured) best = std::min(best, t);
  }
  return best;
}

// Both kernels move exactly one read + one write per amplitude.
static double gbps(int n, double seconds) {
  double bytes = (double)(1ull << n) * 8.0 * 2.0;
  return bytes / seconds / 1e9;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    int nmax = (argc > 1) ? atoi(argv[1]) : 33;

    Ctx ctx{};
    ctx.dev = MTLCreateSystemDefaultDevice();
    if (!ctx.dev) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    ctx.queue = [ctx.dev newCommandQueue];

    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.dev newLibraryWithSource:kSource
                                               options:nil
                                                 error:&err];
    if (!lib) {
      fprintf(stderr, "shader compile failed: %s\n",
              [[err description] UTF8String]);
      return 1;
    }
    ctx.init_pso = [ctx.dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"init_state"]
                                      error:&err];
    ctx.stream_pso = [ctx.dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream_scale"]
                                      error:&err];
    ctx.gate_pso = [ctx.dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gate1q_inplace"]
                                      error:&err];
    if (!ctx.init_pso || !ctx.stream_pso || !ctx.gate_pso) {
      fprintf(stderr, "pipeline creation failed: %s\n",
              [[err description] UTF8String]);
      return 1;
    }

    uint64_t bytes_a = (1ull << nmax) * 8ull;
    // The out-of-place destination is only needed for the stream test, which we
    // cap one qubit below nmax so both buffers fit the working set.
    int nstream_max = nmax - 1;
    uint64_t bytes_b = (1ull << nstream_max) * 8ull;

    printf("device                : %s\n", [ctx.dev.name UTF8String]);
    printf("maxBufferLength       : %.2f GiB\n", ctx.dev.maxBufferLength / kGiB);
    printf("recommendedMaxWorkSet : %.2f GiB\n",
           ctx.dev.recommendedMaxWorkingSetSize / kGiB);
    printf("state buffer          : n=%d, %.2f GiB\n", nmax, bytes_a / kGiB);
    printf("stream dst buffer     : n=%d, %.2f GiB\n", nstream_max,
           bytes_b / kGiB);
    fflush(stdout);

    if (bytes_a > (uint64_t)ctx.dev.maxBufferLength) {
      fprintf(stderr, "n=%d exceeds maxBufferLength\n", nmax);
      return 1;
    }

    NSDate *t0 = [NSDate date];
    ctx.a = [ctx.dev newBufferWithLength:bytes_a options:MTLResourceStorageModeShared];
    ctx.b = [ctx.dev newBufferWithLength:bytes_b options:MTLResourceStorageModeShared];
    if (!ctx.a || !ctx.b) {
      fprintf(stderr, "allocation failed\n");
      return 1;
    }
    printf("allocation (lazy)     : %.3f s\n", -[t0 timeIntervalSinceNow]);
    fflush(stdout);

    // First touch: commit every page before timing anything.
    t0 = [NSDate date];
    time_dispatch(ctx, ctx.init_pso, 1ull << nmax,
                  ^(id<MTLComputeCommandEncoder> enc, uint32_t gw) {
                    [enc setBuffer:ctx.a offset:0 atIndex:0];
                    [enc setBytes:&gw length:sizeof(gw) atIndex:1];
                  });
    printf("first touch (%.1f GiB) : %.3f s\n", bytes_a / kGiB,
           -[t0 timeIntervalSinceNow]);

    // -----------------------------------------------------------------
    printf("\n=== 1. streaming bandwidth vs qubit count ===\n");
    printf("n,bytes_moved,seconds,GB/s\n");
    for (int n = 20; n <= nstream_max; n++) {
      int reps = (n >= 30) ? 2 : 5;
      double t = run_stream(ctx, n, reps);
      printf("%d,%.0f,%.6f,%.1f\n", n, (double)(1ull << n) * 16.0, t,
             gbps(n, t));
      fflush(stdout);
    }

    // -----------------------------------------------------------------
    printf("\n=== 2. in-place 1q gate: time vs target qubit (stride sweep) ===\n");
    printf("n,q,stride_bytes,seconds,GB/s\n");
    std::vector<int> sweep_ns = {26, 28, 30};
    if (nmax >= 32) sweep_ns.push_back(32);
    for (int n : sweep_ns) {
      if (n > nmax) continue;
      int reps = (n >= 30) ? 2 : 4;
      for (int q = 0; q < n; q++) {
        double t = run_gate(ctx, n, q, reps);
        printf("%d,%d,%.0f,%.6f,%.1f\n", n, q, (double)(1ull << q) * 8.0, t,
               gbps(n, t));
        fflush(stdout);
      }
    }

    // -----------------------------------------------------------------
    printf("\n=== 3. residency under sustained load at n=%d ===\n", nmax);
    printf("iter,q,seconds,GB/s\n");
    for (int it = 0; it < 16; it++) {
      int q = it % 4;  // low q: contiguous, should be bandwidth-limited
      double t = run_gate(ctx, nmax, q, 0);
      printf("%d,%d,%.6f,%.1f\n", it, q, t, gbps(nmax, t));
      fflush(stdout);
    }

    printf("\ndone\n");
    return 0;
  }
}
