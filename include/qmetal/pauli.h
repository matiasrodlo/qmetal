// pauli.h — Pauli strings and Hamiltonians.
//
// A Pauli string is stored as three bitmasks rather than a per-qubit array, so
// a term of any weight fits in 24 bytes and reaches a kernel as three scalars.
// Qubit q carries X if bit q of x_mask is set, and so on; a qubit set in more
// than one mask is invalid.
//
// Acting on a basis state, P|i> = c(i) |i ^ flip> with flip = x_mask | y_mask.
// The coefficient follows from Y|b> = i (-1)^b |1-b>, X|b> = |1-b>, and
// Z|b> = (-1)^b |b>:
//
//     c(i) = i^ny * (-1)^(popcount(i & y_mask) + popcount(i & z_mask))
//
// so the whole expectation value is a single fused pass over the state with no
// scratch buffer and no intermediate |P psi>.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qmetal {

struct PauliString {
  uint64_t x_mask = 0;
  uint64_t y_mask = 0;
  uint64_t z_mask = 0;

  bool valid(uint32_t num_qubits) const;
  uint32_t weight() const;  // number of non-identity qubits

  // "IXYZ" style, leftmost character is qubit 0. Throws on a bad character or
  // a length exceeding num_qubits.
  static PauliString parse(const std::string &s);
  std::string to_string(uint32_t num_qubits) const;

  static PauliString identity() { return PauliString{}; }
  static PauliString single(char axis, uint32_t q);
};

struct PauliTerm {
  double coefficient = 1.0;
  PauliString pauli;
};

// A Hermitian observable as a real-weighted sum of Pauli strings.
struct Hamiltonian {
  std::vector<PauliTerm> terms;
  void add(double c, const PauliString &p) { terms.push_back({c, p}); }
  size_t size() const { return terms.size(); }
};

}  // namespace qmetal
