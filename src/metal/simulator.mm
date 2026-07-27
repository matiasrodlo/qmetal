#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

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

  for (const char *name : {"init_basis", "apply_1q", "apply_controlled_1q",
                           "apply_diagonal_2q", "apply_swap", "apply_ccx"}) {
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

void Simulator::run(const Circuit &c) {
  if (c.num_qubits != impl_->n)
    throw std::runtime_error("circuit/simulator size mismatch");
  for (const Gate &g : c.ops) apply(g);
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

double Simulator::norm() {
  const float *p = raw();
  // Accumulate in double: a float32 sum over 2^n terms loses far too much.
  double s = 0.0;
  for (size_t i = 0; i < impl_->dim; i++) {
    double re = p[2 * i], im = p[2 * i + 1];
    s += re * re + im * im;
  }
  return std::sqrt(s);
}

}  // namespace qmetal
