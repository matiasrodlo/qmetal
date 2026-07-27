#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "qmetal/reference.h"  // gate_matrix_1q, in double
#include "qmetal/simulator.h"

// Kernel source, embedded at build time from src/metal/kernels.metal.
#include "kernels_source.h"

namespace qmetal {
namespace {

// Blocks cannot capture raw array types, so every kernel argument that reaches
// a dispatch block is a named aggregate. These mirror the .metal layouts.
struct GateF32 {
  float m00[2], m01[2], m10[2], m11[2];
};
struct PhaseF32 {
  float re, im;
};

// Gate matrices are computed in double and narrowed only at assignment. Apple
// GPUs have no fp64, so this is the only place precision can be preserved.
GateF32 narrow(const cdouble m[4]) {
  GateF32 g;
  auto put = [](float *dst, const cdouble &z) {
    dst[0] = static_cast<float>(z.real());
    dst[1] = static_cast<float>(z.imag());
  };
  put(g.m00, m[0]);
  put(g.m01, m[1]);
  put(g.m10, m[2]);
  put(g.m11, m[3]);
  return g;
}

// Flush after this many encodes. Metal tolerates far more, but bounding the
// command buffer keeps peak driver memory predictable on deep circuits.
constexpr uint32_t kFlushEvery = 4096;

// Reduction geometry. Threadgroups are fixed rather than derived from the state
// size, so the dispatch stays 1-D and inside 32 bits at any n. Enough groups to
// saturate 40 GPU cores, few enough that the per-thread float run stays short.
constexpr uint32_t kReduceGroups = 4096;
constexpr uint32_t kReduceThreads = 256;

// Amplitudes per block for the sampler. A block is 16 KiB, one page-ish, so
// resolving a shot inside a block touches very little of the buffer.
constexpr uint32_t kSampleBlock = 1024;

// splitmix64: same seed gives the same stream on any platform. Do not swap in
// a libc RNG -- reproducibility is a requirement, not a nicety.
inline uint64_t splitmix64(uint64_t &s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

}  // namespace

struct Simulator::Impl {
  uint32_t n = 0;
  size_t dim = 0;
  uint64_t dispatches = 0;

  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLBuffer> state = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

  id<MTLCommandBuffer> cmd = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  uint32_t encoded = 0;

  // Bump-allocated pool for diagonal-layer terms.
  //
  // Dispatches are encoded lazily and executed later, so a buffer written at
  // encode time must not be rewritten before the GPU reads it. Handing each
  // dispatch its own region and resetting only on flush is what makes batching
  // and per-op parameter buffers coexist. Reusing one region for every layer
  // silently corrupts all but the last -- which is exactly what it did.
  id<MTLBuffer> term_pool = nil;
  size_t pool_capacity = 0;
  size_t pool_offset = 0;

  void grow_pool(size_t bytes) {
    size_t cap = pool_capacity ? pool_capacity : (size_t(1) << 16);
    while (cap < bytes) cap *= 2;
    term_pool = [device newBufferWithLength:cap
                                    options:MTLResourceStorageModeShared];
    pool_capacity = cap;
    pool_offset = 0;
  }

  // Metal requires buffer offsets to be 256-byte aligned.
  size_t pool_alloc(size_t bytes) {
    size_t aligned = (bytes + 255) & ~size_t(255);
    if (pool_offset + aligned > pool_capacity) {
      flush(true);              // drain anything still referencing the pool
      pool_offset = 0;
      if (aligned > pool_capacity) grow_pool(aligned);
    }
    size_t off = pool_offset;
    pool_offset += aligned;
    return off;
  }

  // Reduction scratch, allocated on first use and reused thereafter.
  id<MTLBuffer> partials = nil;   // kReduceGroups * float2
  id<MTLBuffer> block_sums = nil; // one float per sampler block

  // Run a reduction to completion and return the per-threadgroup partials.
  // Reductions are synchronous by nature: the caller wants a number.
  template <typename Bind>
  const void *reduce(const char *kernel, id<MTLBuffer> out, Bind bind) {
    flush(true);  // the state must be settled before it is read
    id<MTLCommandBuffer> c = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
    [e setComputePipelineState:pipeline(kernel)];
    [e setBuffer:state offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    uint64_t cnt = dim;
    [e setBytes:&cnt length:sizeof(cnt) atIndex:2];
    bind(e);
    [e dispatchThreadgroups:MTLSizeMake(kReduceGroups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    [e endEncoding];
    [c commit];
    [c waitUntilCompleted];
    if (c.error)
      throw std::runtime_error(std::string("reduction failed: ") +
                               [[c.error description] UTF8String]);
    dispatches++;
    return out.contents;
  }

  id<MTLComputePipelineState> pipeline(const char *name) {
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;
    throw std::runtime_error(std::string("missing pipeline: ") + name);
  }

  // Open a command buffer/encoder if none is live.
  id<MTLComputeCommandEncoder> encoder() {
    if (enc) return enc;
    cmd = [queue commandBuffer];
    enc = [cmd computeCommandEncoder];
    encoded = 0;
    return enc;
  }

  void flush(bool wait) {
    if (!enc) return;
    [enc endEncoding];
    [cmd commit];
    if (wait) {
      [cmd waitUntilCompleted];
      if (cmd.error) {
        NSString *d = [cmd.error description];
        enc = nil;
        cmd = nil;
        throw std::runtime_error(std::string("GPU error: ") + [d UTF8String]);
      }
    }
    enc = nil;
    cmd = nil;
    encoded = 0;
    pool_offset = 0;  // nothing references the pool once the buffer has run
  }

  // Dispatch `threads` work items, splitting into a 2-D grid so the x extent
  // stays inside 32 bits. `threads` is always a power of two here.
  void dispatch(const char *kernel, uint64_t threads,
                void (^bind)(id<MTLComputeCommandEncoder>, uint32_t)) {
    if (threads == 0) return;
    uint64_t w = std::min<uint64_t>(threads, 1ull << 26);
    uint64_t h = threads / w;
    uint32_t grid_w = static_cast<uint32_t>(w);

    id<MTLComputeCommandEncoder> e = encoder();
    [e setComputePipelineState:pipeline(kernel)];
    [e setBuffer:state offset:0 atIndex:0];
    bind(e, grid_w);
    NSUInteger tg = std::min<NSUInteger>(256, grid_w);
    [e dispatchThreads:MTLSizeMake(w, h, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

    dispatches++;
    if (++encoded >= kFlushEvery) flush(false);
  }
};

// ---------------------------------------------------------------------------

bool Simulator::available() { return MTLCreateSystemDefaultDevice() != nil; }

std::string Simulator::device_name() {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  return d ? std::string([d.name UTF8String]) : std::string("none");
}

uint64_t Simulator::max_buffer_length() {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  return d ? (uint64_t)d.maxBufferLength : 0;
}

uint64_t Simulator::recommended_working_set() {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  return d ? (uint64_t)d.recommendedMaxWorkingSetSize : 0;
}

Simulator::Simulator(uint32_t num_qubits) : impl_(new Impl) {
  if (num_qubits == 0 || num_qubits > kMaxQubits)
    throw std::runtime_error("Simulator: num_qubits must be 1.." +
                             std::to_string(kMaxQubits));

  impl_->device = MTLCreateSystemDefaultDevice();
  if (!impl_->device) throw std::runtime_error("Simulator: no Metal device");
  impl_->queue = [impl_->device newCommandQueue];

  NSError *err = nil;
  id<MTLLibrary> lib = [impl_->device newLibraryWithSource:@(kKernelSource)
                                                   options:nil
                                                     error:&err];
  if (!lib)
    throw std::runtime_error(std::string("shader compile failed: ") +
                             [[err description] UTF8String]);

  for (const char *name :
       {"init_basis", "apply_1q", "apply_controlled_1q", "apply_diagonal_2q",
        "apply_swap", "apply_ccx", "apply_diagonal_layer",
        "apply_1q_group2", "apply_1q_group3", "apply_1q_group4",
        "reduce_abs2", "reduce_pauli",
        "block_abs2"}) {
    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
    if (!fn)
      throw std::runtime_error(std::string("kernel not found: ") + name);
    id<MTLComputePipelineState> pso =
        [impl_->device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
      throw std::runtime_error(std::string("pipeline failed for ") + name +
                               ": " + [[err description] UTF8String]);
    impl_->pipelines[name] = pso;
  }

  impl_->n = num_qubits;
  impl_->dim = size_t(1) << num_qubits;

  // Preflight before allocating. Passing only means the device's own limits do
  // not already rule it out -- free system memory can still refuse the request.
  uint64_t bytes = uint64_t(impl_->dim) * 8ull;
  if (bytes > (uint64_t)impl_->device.maxBufferLength)
    throw std::runtime_error(
        std::to_string(num_qubits) + " qubits needs " +
        std::to_string(bytes >> 30) + " GiB, past maxBufferLength " +
        std::to_string((uint64_t)impl_->device.maxBufferLength >> 30) + " GiB");

  impl_->state = [impl_->device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
  if (!impl_->state)
    throw std::runtime_error("Simulator: state allocation failed (" +
                             std::to_string(bytes >> 30) + " GiB)");
  reset();
}

Simulator::~Simulator() {
  try {
    if (impl_) impl_->flush(true);
  } catch (...) {
    // A destructor must not throw. The buffer is released either way.
  }
}

uint32_t Simulator::num_qubits() const { return impl_->n; }
size_t Simulator::dim() const { return impl_->dim; }
uint64_t Simulator::dispatch_count() const { return impl_->dispatches; }

void Simulator::reset() {
  Impl *s = impl_.get();
  s->dispatch("init_basis", s->dim,
              ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                [e setBytes:&gw length:sizeof(gw) atIndex:1];
              });
}

void Simulator::apply(const Gate &g) {
  Impl *s = impl_.get();
  for (int k = 0; k < g.num_qubits; k++)
    if (g.qubits[k] >= s->n)
      throw std::runtime_error("qubit index out of range");

  if (g.num_qubits == 1) {
    cdouble m[4];
    gate_matrix_1q(g.kind, g.params, m);  // throws if not a 1q gate
    GateF32 gm = narrow(m);
    uint32_t q = g.qubits[0];
    s->dispatch("apply_1q", s->dim >> 1,
                ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                  [e setBytes:&gm length:sizeof(gm) atIndex:1];
                  [e setBytes:&q length:sizeof(q) atIndex:2];
                  [e setBytes:&gw length:sizeof(gw) atIndex:3];
                });
    return;
  }

  if (g.num_qubits == 2) {
    if (g.qubits[0] == g.qubits[1])
      throw std::runtime_error("2-qubit gate on identical wires");
    uint32_t a = g.qubits[0], b = g.qubits[1];

    switch (g.kind) {
      case G_CX:
      case G_CY: {
        cdouble m[4];
        gate_matrix_1q(g.kind == G_CX ? G_X : G_Y, nullptr, m);
        GateF32 gm = narrow(m);
        s->dispatch("apply_controlled_1q", s->dim >> 2,
                    ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                      [e setBytes:&gm length:sizeof(gm) atIndex:1];
                      [e setBytes:&a length:sizeof(a) atIndex:2];
                      [e setBytes:&b length:sizeof(b) atIndex:3];
                      [e setBytes:&gw length:sizeof(gw) atIndex:4];
                    });
        return;
      }
      case G_CZ:
      case G_CP: {
        cdouble ph = (g.kind == G_CZ) ? cdouble(-1, 0)
                                      : std::polar(1.0, g.params[0]);
        PhaseF32 p{static_cast<float>(ph.real()),
                   static_cast<float>(ph.imag())};
        s->dispatch("apply_diagonal_2q", s->dim >> 2,
                    ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                      [e setBytes:&p length:sizeof(p) atIndex:1];
                      [e setBytes:&a length:sizeof(a) atIndex:2];
                      [e setBytes:&b length:sizeof(b) atIndex:3];
                      [e setBytes:&gw length:sizeof(gw) atIndex:4];
                    });
        return;
      }
      case G_SWAP:
        s->dispatch("apply_swap", s->dim >> 2,
                    ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                      [e setBytes:&a length:sizeof(a) atIndex:1];
                      [e setBytes:&b length:sizeof(b) atIndex:2];
                      [e setBytes:&gw length:sizeof(gw) atIndex:3];
                    });
        return;
      default:
        break;
    }
  }

  if (g.num_qubits == 3 && g.kind == G_CCX) {
    if (s->n < 3) throw std::runtime_error("CCX needs at least 3 qubits");
    uint32_t c0 = g.qubits[0], c1 = g.qubits[1], t = g.qubits[2];
    if (c0 == c1 || c0 == t || c1 == t)
      throw std::runtime_error("CCX on repeated wires");
    s->dispatch("apply_ccx", s->dim >> 3,
                ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                  [e setBytes:&c0 length:sizeof(c0) atIndex:1];
                  [e setBytes:&c1 length:sizeof(c1) atIndex:2];
                  [e setBytes:&t length:sizeof(t) atIndex:3];
                  [e setBytes:&gw length:sizeof(gw) atIndex:4];
                });
    return;
  }

  // Fail closed. An unsupported gate is an error, never a silent identity.
  throw std::runtime_error(std::string("unsupported gate: ") +
                           gate_name(g.kind));
}

void Simulator::run(const Circuit &c) { run(c, FusionOptions{}); }

void Simulator::run(const Circuit &c, const FusionOptions &opts) {
  if (c.num_qubits != impl_->n)
    throw std::runtime_error("circuit/simulator size mismatch");
  run(build_plan(c, opts));
}

void Simulator::run(const Plan &p) {
  if (p.num_qubits != impl_->n)
    throw std::runtime_error("plan/simulator size mismatch");
  for (const PlanOp &op : p.ops) apply(op);
}

void Simulator::apply(const PlanOp &op) {
  Impl *s = impl_.get();
  switch (op.kind) {
    case PlanOp::Kind::Dense1q: {
      GateF32 gm = narrow(op.matrix);
      uint32_t q = op.qubits[0];
      s->dispatch("apply_1q", s->dim >> 1,
                  ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                    [e setBytes:&gm length:sizeof(gm) atIndex:1];
                    [e setBytes:&q length:sizeof(q) atIndex:2];
                    [e setBytes:&gw length:sizeof(gw) atIndex:3];
                  });
      return;
    }
    case PlanOp::Kind::Dense1qGroup: {
      const size_t k = op.group_qubits.size();
      if (k < 2 || k > 4) throw std::runtime_error("group size must be 2..4");
      // Parameters go through the same bump pool as diagonal terms, for the
      // same reason: dispatches are encoded now and executed later.
      size_t goff = s->pool_alloc(k * sizeof(GateF32));
      size_t qoff = s->pool_alloc(k * sizeof(uint32_t));
      auto *base = static_cast<uint8_t *>(s->term_pool.contents);
      auto *gp = reinterpret_cast<GateF32 *>(base + goff);
      auto *qp = reinterpret_cast<uint32_t *>(base + qoff);
      for (size_t j = 0; j < k; j++) {
        gp[j] = narrow(&op.group_matrices[4 * j]);
        qp[j] = op.group_qubits[j];
      }
      const char *kernel = (k == 2)   ? "apply_1q_group2"
                           : (k == 3) ? "apply_1q_group3"
                                      : "apply_1q_group4";
      id<MTLBuffer> pool = s->term_pool;
      s->dispatch(kernel, s->dim >> k,
                  ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                    [e setBuffer:pool offset:goff atIndex:1];
                    [e setBuffer:pool offset:qoff atIndex:2];
                    [e setBytes:&gw length:sizeof(gw) atIndex:3];
                  });
      return;
    }
    case PlanOp::Kind::Controlled1q: {
      GateF32 gm = narrow(op.matrix);
      uint32_t a = op.qubits[0], b = op.qubits[1];
      s->dispatch("apply_controlled_1q", s->dim >> 2,
                  ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                    [e setBytes:&gm length:sizeof(gm) atIndex:1];
                    [e setBytes:&a length:sizeof(a) atIndex:2];
                    [e setBytes:&b length:sizeof(b) atIndex:3];
                    [e setBytes:&gw length:sizeof(gw) atIndex:4];
                  });
      return;
    }
    case PlanOp::Kind::Swap: {
      uint32_t a = op.qubits[0], b = op.qubits[1];
      s->dispatch("apply_swap", s->dim >> 2,
                  ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                    [e setBytes:&a length:sizeof(a) atIndex:1];
                    [e setBytes:&b length:sizeof(b) atIndex:2];
                    [e setBytes:&gw length:sizeof(gw) atIndex:3];
                  });
      return;
    }
    case PlanOp::Kind::CCX: {
      uint32_t c0 = op.qubits[0], c1 = op.qubits[1], t = op.qubits[2];
      s->dispatch("apply_ccx", s->dim >> 3,
                  ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                    [e setBytes:&c0 length:sizeof(c0) atIndex:1];
                    [e setBytes:&c1 length:sizeof(c1) atIndex:2];
                    [e setBytes:&t length:sizeof(t) atIndex:3];
                    [e setBytes:&gw length:sizeof(gw) atIndex:4];
                  });
      return;
    }
    case PlanOp::Kind::DiagonalLayer: {
      const size_t nt = op.masks.size();
      if (nt == 0) return;
      // Sort by ascending |angle| so the kernel's float accumulation adds small
      // contributions together before they meet large ones. Without this the
      // tail of a phase ladder is lost to round-off entirely.
      std::vector<size_t> idx(nt);
      for (size_t i = 0; i < nt; i++) idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return std::abs(op.angles[a]) < std::abs(op.angles[b]);
      });
      size_t mask_off = s->pool_alloc(nt * sizeof(uint64_t));
      size_t angle_off = s->pool_alloc(nt * sizeof(float));
      auto *base = static_cast<uint8_t *>(s->term_pool.contents);
      auto *mp = reinterpret_cast<uint64_t *>(base + mask_off);
      auto *ap = reinterpret_cast<float *>(base + angle_off);
      for (size_t i = 0; i < nt; i++) {
        mp[i] = op.masks[idx[i]];
        ap[i] = static_cast<float>(op.angles[idx[i]]);
      }
      uint32_t count = static_cast<uint32_t>(nt);
      PhaseF32 gp{static_cast<float>(std::cos(op.global_phase)),
                  static_cast<float>(std::sin(op.global_phase))};
      id<MTLBuffer> pool = s->term_pool;
      s->dispatch("apply_diagonal_layer", s->dim,
                  ^(id<MTLComputeCommandEncoder> e, uint32_t gw) {
                    [e setBuffer:pool offset:mask_off atIndex:1];
                    [e setBuffer:pool offset:angle_off atIndex:2];
                    [e setBytes:&count length:sizeof(count) atIndex:3];
                    [e setBytes:&gp length:sizeof(gp) atIndex:4];
                    [e setBytes:&gw length:sizeof(gw) atIndex:5];
                  });
      return;
    }
  }
  throw std::runtime_error("Simulator: unknown PlanOp kind");
}

void Simulator::synchronize() { impl_->flush(true); }

const float *Simulator::raw() {
  synchronize();
  return static_cast<const float *>(impl_->state.contents);
}

std::vector<cdouble> Simulator::amplitudes() {
  const float *p = raw();
  std::vector<cdouble> out(impl_->dim);
  for (size_t i = 0; i < impl_->dim; i++)
    out[i] = cdouble(p[2 * i], p[2 * i + 1]);
  return out;
}

// ---------------------------------------------------------------------------
// Reductions

double Simulator::abs2sum() {
  Impl *s = impl_.get();
  if (!s->partials)
    s->partials = [s->device newBufferWithLength:kReduceGroups * sizeof(float) * 2
                                         options:MTLResourceStorageModeShared];
  const float *p = static_cast<const float *>(
      s->reduce("reduce_abs2", s->partials, ^(id<MTLComputeCommandEncoder>) {}));
  double total = 0.0;  // widen here; the GPU stage had no fp64 available
  for (uint32_t i = 0; i < kReduceGroups; i++) total += p[i];
  return total;
}

double Simulator::norm() { return std::sqrt(abs2sum()); }

double Simulator::expectation(const PauliString &pauli) {
  Impl *s = impl_.get();
  if (!pauli.valid(s->n))
    throw std::runtime_error("expectation: invalid Pauli string for this width");
  if (!s->partials)
    s->partials = [s->device newBufferWithLength:kReduceGroups * sizeof(float) * 2
                                         options:MTLResourceStorageModeShared];

  const uint64_t flip = pauli.x_mask | pauli.y_mask;
  // Y contributes both a per-index sign and a global i per Y.
  const uint64_t sign_mask = pauli.y_mask | pauli.z_mask;
  const uint32_t ny = __builtin_popcountll(pauli.y_mask);

  const float *p = static_cast<const float *>(
      s->reduce("reduce_pauli", s->partials, ^(id<MTLComputeCommandEncoder> e) {
        [e setBytes:&flip length:sizeof(flip) atIndex:3];
        [e setBytes:&sign_mask length:sizeof(sign_mask) atIndex:4];
      }));

  double re = 0.0, im = 0.0;
  for (uint32_t i = 0; i < kReduceGroups; i++) {
    re += p[2 * i];
    im += p[2 * i + 1];
  }
  // Multiply by i^ny exactly, then take the real part: P is Hermitian, so the
  // imaginary part is round-off and discarding it is not an approximation.
  switch (ny & 3u) {
    case 0: return re;
    case 1: return -im;
    case 2: return -re;
    default: return im;
  }
}

double Simulator::expectation(const Hamiltonian &h) {
  double total = 0.0;
  for (const PauliTerm &t : h.terms) total += t.coefficient * expectation(t.pauli);
  return total;
}

// ---------------------------------------------------------------------------
// Sampling
//
// Two levels. The GPU reduces |amp|^2 into per-block sums; the host builds a
// double-precision CDF over those blocks, binary-searches it per shot, then
// walks inside the chosen block reading directly from the shared buffer. The
// statevector is never copied and only one block is touched per shot.

std::vector<uint64_t> Simulator::sample(size_t shots, uint64_t seed) {
  Impl *s = impl_.get();
  std::vector<uint64_t> out;
  if (shots == 0) return out;

  const uint64_t nblocks = (s->dim + kSampleBlock - 1) / kSampleBlock;
  if (!s->block_sums)
    s->block_sums =
        [s->device newBufferWithLength:nblocks * sizeof(float)
                               options:MTLResourceStorageModeShared];

  s->flush(true);
  {
    uint64_t w = std::min<uint64_t>(nblocks, 1ull << 26);
    uint64_t hgt = (nblocks + w - 1) / w;
    uint32_t grid_w = static_cast<uint32_t>(w);
    uint64_t cnt = s->dim;
    uint32_t blk = kSampleBlock;
    id<MTLCommandBuffer> c = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
    [e setComputePipelineState:s->pipeline("block_abs2")];
    [e setBuffer:s->state offset:0 atIndex:0];
    [e setBuffer:s->block_sums offset:0 atIndex:1];
    [e setBytes:&cnt length:sizeof(cnt) atIndex:2];
    [e setBytes:&blk length:sizeof(blk) atIndex:3];
    [e setBytes:&grid_w length:sizeof(grid_w) atIndex:4];
    [e dispatchThreads:MTLSizeMake(w, hgt, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, w), 1, 1)];
    [e endEncoding];
    [c commit];
    [c waitUntilCompleted];
    if (c.error)
      throw std::runtime_error("sampling reduction failed");
    s->dispatches++;
  }

  const float *bs = static_cast<const float *>(s->block_sums.contents);
  std::vector<double> cdf(nblocks);
  double run = 0.0;
  for (uint64_t b = 0; b < nblocks; b++) {
    run += bs[b];
    cdf[b] = run;
  }
  const double total = run;
  const float *amps = static_cast<const float *>(s->state.contents);

  out.resize(shots);
  if (total <= 0.0) {
    std::fill(out.begin(), out.end(), 0ull);
    return out;
  }

  uint64_t rng = seed;
  for (size_t k = 0; k < shots; k++) {
    // 53-bit uniform in [0,1), scaled by the ACTUAL total. The float32 gate
    // accumulation leaves the norm slightly off 1, and assuming 1 would bias
    // the tail.
    double u = double(splitmix64(rng) >> 11) * (1.0 / 9007199254740992.0) * total;

    uint64_t lo = 0, hi = nblocks - 1;
    while (lo < hi) {
      uint64_t mid = (lo + hi) >> 1;
      if (cdf[mid] > u) hi = mid; else lo = mid + 1;
    }
    const uint64_t b = lo;
    double rem = u - (b > 0 ? cdf[b - 1] : 0.0);

    uint64_t start = b * kSampleBlock;
    uint64_t end = std::min<uint64_t>(start + kSampleBlock, s->dim);
    double acc = 0.0;
    uint64_t chosen = start;
    int64_t last_nonzero = -1;
    bool found = false;
    for (uint64_t i = start; i < end; i++) {
      double re = amps[2 * i], im = amps[2 * i + 1];
      double pr = re * re + im * im;
      if (pr > 0.0) last_nonzero = int64_t(i);
      acc += pr;
      if (acc > rem) { chosen = i; found = true; break; }
    }
    // Round-off can leave the running sum just short; clamp to the last state
    // that actually had probability rather than to the block edge.
    if (!found) chosen = (last_nonzero >= 0) ? uint64_t(last_nonzero) : end - 1;
    out[k] = chosen;
  }
  return out;
}

std::vector<std::pair<std::string, uint64_t>> Simulator::counts(size_t shots,
                                                                uint64_t seed) {
  auto idx = sample(shots, seed);
  std::unordered_map<std::string, uint64_t> tally;
  const uint32_t n = impl_->n;
  for (uint64_t v : idx) {
    std::string bits(n, '0');
    for (uint32_t q = 0; q < n; q++)
      if (v & (1ull << q)) bits[n - 1 - q] = '1';  // qubit 0 rightmost
    tally[bits]++;
  }
  std::vector<std::pair<std::string, uint64_t>> out(tally.begin(), tally.end());
  std::sort(out.begin(), out.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  return out;
}

}  // namespace qmetal
