// circuit.h — core gate and circuit types.
//
// Deliberately a flat, C-ABI-friendly aggregate: no virtuals, no heap per gate,
// no inheritance. A circuit is a contiguous array of fixed-size records that can
// be handed to a Metal kernel, a C caller, or the CPU reference unchanged.
//
// Convention, pinned: LSB-first. Qubit q occupies bit q of the basis index, so
// its amplitude stride is 1 << q. Do not revisit this.

#pragma once

#include <cstdint>
#include <vector>

namespace qmetal {

enum GateKind : uint8_t {
  // 1 qubit, no parameters
  G_X = 0,
  G_Y,
  G_Z,
  G_H,
  G_S,
  G_SDG,
  G_T,
  G_TDG,
  // 1 qubit, one parameter
  G_RX,
  G_RY,
  G_RZ,
  G_P,
  // 2 qubits, no parameters
  G_CX,
  G_CY,
  G_CZ,
  G_SWAP,
  // 2 qubits, one parameter
  G_CP,
  // 3 qubits
  G_CCX,

  G_KIND_COUNT
};

struct Gate {
  GateKind kind;
  uint8_t num_qubits;
  uint8_t num_params;
  uint32_t qubits[3];
  double params[3];
};

// Constructors. Multi-qubit ordering is (control..., target) for controlled
// gates and (a, b) for SWAP.
Gate g1(GateKind k, uint32_t q);
Gate g1p(GateKind k, uint32_t q, double theta);
Gate g2(GateKind k, uint32_t a, uint32_t b);
Gate g2p(GateKind k, uint32_t a, uint32_t b, double theta);
Gate g3(GateKind k, uint32_t a, uint32_t b, uint32_t c);

struct Circuit {
  uint32_t num_qubits = 0;
  std::vector<Gate> ops;

  void add(const Gate &g) { ops.push_back(g); }
  size_t size() const { return ops.size(); }

  // Gate count by arity, for reporting and for sizing decisions.
  size_t count_1q() const;
  size_t count_2q() const;
};

const char *gate_name(GateKind k);

}  // namespace qmetal
