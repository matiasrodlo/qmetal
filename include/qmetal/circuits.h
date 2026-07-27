// circuits.h — the frozen benchmark set.
//
// Fixed before any optimization exists, so that later tuning cannot drift
// toward whatever happens to be measured. Every generator is deterministic:
// same arguments produce the same gate stream, on any machine, forever.
//
// Changing a generator invalidates every prior measurement. Add a new one
// instead.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qmetal/circuit.h"

namespace qmetal {

// Entangles every qubit with a single Hadamard and a CX chain. Minimal gate
// count, maximal structure: the easiest thing to fuse and the first thing to
// get right.
Circuit ghz(uint32_t n);

// Textbook QFT: per stage a Hadamard then an exact-angle controlled-phase
// ladder. No trailing swap layer -- the bit order is reversed on output, and
// every consumer must agree on that.
Circuit qft(uint32_t n);

// Ring QAOA, p layers. Each layer is a ZZ coupling on every ring bond followed
// by a transverse RX. ZZ is emitted as CX / RZ / CX so the gate set stays small.
Circuit qaoa_ring(uint32_t n, uint32_t p);

// First-order Trotterized transverse-field Ising evolution on a chain.
Circuit tfim_trotter(uint32_t n, uint32_t steps);

// Grover-shaped workload: a Hadamard layer, then iterations of a phase oracle
// and a diffusion operator. The oracle marks a single basis state using a CZ
// rather than a full multi-controlled Z, so this reproduces Grover's *gate
// structure* and cost profile without its exact unitary. Named accordingly.
Circuit grover_proxy(uint32_t n, uint32_t iters);

// Dense random circuit: a random single-qubit gate on every wire, then a CX
// layer whose pairing shifts each round. Deliberately hostile to fusion.
Circuit random_circuit(uint32_t n, uint32_t depth, uint64_t seed = 0x5eed);

// Hardware-efficient ansatz: RY/RZ on every wire then a CX chain, per layer.
// The shape most variational workloads actually run.
Circuit hardware_efficient(uint32_t n, uint32_t depth, uint64_t seed = 0x5eed);

// A subroutine acting repeatedly on a narrow work register, weakly coupled to
// the rest -- an ancilla-based oracle, a block-encoded subroutine, a local
// Hamiltonian patch. Every other circuit in the frozen set is full-width: each
// layer touches every wire, so there is no locality in *wire* space for cache
// blocking to exploit, no matter how the gates are fused. This one has it by
// construction, which is what makes the two cases separable.
Circuit local_patch(uint32_t n, uint32_t width, uint32_t rounds,
                    uint64_t seed = 0x10ca1);

struct BenchmarkSpec {
  std::string name;
  Circuit (*build)(uint32_t n);
};

// The frozen set, at whatever qubit count the caller asks for. Depth-carrying
// circuits use the fixed depths recorded here.
const std::vector<BenchmarkSpec> &frozen_benchmarks();

// The frozen set plus circuits added later. Appending is allowed; editing an
// existing generator is not, since it would invalidate prior measurements.
// Geometric means over the frozen seven stay comparable across the whole
// project history; means over this list do not.
const std::vector<BenchmarkSpec> &extended_benchmarks();

}  // namespace qmetal
