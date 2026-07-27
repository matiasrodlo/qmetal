// plan.h — the fused, scheduled program an executor runs.
//
// A Circuit is what the caller wrote. A Plan is what the GPU is asked to do.
// Fusion is the translation, and it is a pipeline of small passes over a typed
// IR rather than pattern matching buried in one long function. Each pass has a
// name, a precondition, and a test.
//
// The whole pipeline is pure C++ with no Metal dependency, so it is verified
// against the CPU oracle before a kernel ever runs.
//
// The quantity every pass exists to reduce is Plan::size(): this workload is
// memory-bound, so runtime tracks the number of full-state passes, not the
// amount of arithmetic.

#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "qmetal/circuit.h"

namespace qmetal {

using cdouble = std::complex<double>;

struct PlanOp {
  enum class Kind : uint8_t {
    Dense1q,        // arbitrary 2x2 on one wire
    Dense1qGroup,   // K independent 2x2s, one pass, amplitudes held in registers
    Blocked1q,      // a run of 2x2s on low wires, staged in threadgroup memory
    Controlled1q,   // arbitrary 2x2 on `target`, gated by `control`
    DiagonalLayer,  // multiply amplitude i by exp(i * phi(i)); see below
    Swap,
    CCX,
  };

  Kind kind;
  uint32_t qubits[3] = {0, 0, 0};
  cdouble matrix[4] = {1, 0, 0, 1};  // Dense1q / Controlled1q, row-major

  // DiagonalLayer. The phase applied to basis index i is
  //
  //     phi(i) = global + sum_t angle[t] over terms with (i & mask[t]) == mask[t]
  //
  // which covers every diagonal gate this project supports: Z/S/T/P/RZ on one
  // wire (single-bit mask) and CZ/CP on two (two-bit mask). An arbitrary run of
  // them collapses into ONE pass over the state regardless of how many gates it
  // contained, because they all commute and their phases simply add.
  std::vector<uint64_t> masks;
  std::vector<double> angles;
  double global_phase = 0.0;

  // Dense1qGroup. Wires ascending; four matrix entries per wire, row-major.
  // A layer of single-qubit gates on distinct wires is the one shape window
  // collapse cannot help -- each wire carries exactly one gate, so there is
  // nothing to multiply. Holding 2^K amplitudes in registers and applying K
  // gates to them turns K full-state passes into one, at identical traffic.
  std::vector<uint32_t> group_qubits;
  std::vector<cdouble> group_matrices;

  // Blocked1q reuses the same two vectors -- it is a group with more wires,
  // executed through threadgroup memory instead of registers. block_bits is the
  // log2 of the staged block size.
  uint32_t block_bits = 0;

  size_t term_count() const { return masks.size(); }
};

struct Plan {
  uint32_t num_qubits = 0;
  std::vector<PlanOp> ops;
  size_t size() const { return ops.size(); }
};

struct FusionOptions {
  // Every optimization is ablatable. An optimization that cannot be turned off
  // cannot be measured, and the ablation grid is the point of the project.
  bool collapse_1q_windows = true;  // per-wire 2x2 products within a 1q window
  bool fuse_diagonals = true;       // runs of commuting diagonal gates
  bool recognize_zz = true;         // CX . RZ . CX is a diagonal ZZ coupling
  uint32_t group_1q = 4;            // wires per register group; 1 disables
  uint32_t block_bits = 0;          // cache-blocking window, log2; 0 disables

  static FusionOptions none() {
    return FusionOptions{false, false, false, 1, 0};
  }
  // Fusion on, blocking on -- the fourth cell of the ablation grid.
  static FusionOptions blocked(uint32_t bits) {
    FusionOptions o;
    o.block_bits = bits;
    return o;
  }
};

// Translate a circuit into a plan. With FusionOptions::none() this is a
// one-to-one lowering and Plan::size() equals Circuit::size(), which is the
// baseline arm of the ablation.
Plan build_plan(const Circuit &c, const FusionOptions &opts = {});

// True if the gate is diagonal in the computational basis.
bool is_diagonal(const Gate &g);

// Diagonal contribution of a gate: appends (mask, angle) and returns the global
// phase it also carries. Precondition: is_diagonal(g).
double diagonal_term(const Gate &g, std::vector<uint64_t> &masks,
                     std::vector<double> &angles);

}  // namespace qmetal
