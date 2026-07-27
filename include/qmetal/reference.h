// reference.h — double-precision CPU statevector reference.
//
// This is the correctness oracle. It exists to be obviously right rather than
// fast: complex128 throughout, one gate at a time, no fusion, no blocking, no
// clever indexing. Every optimization in the GPU path is validated against it.
//
// Apple GPUs have no fp64, so the oracle can only live on the CPU. That is the
// point: an independent implementation in a different precision on different
// hardware, so a shared bug is unlikely.

#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "qmetal/circuit.h"
#include "qmetal/pauli.h"

namespace qmetal {

using cdouble = std::complex<double>;

class Reference {
 public:
  explicit Reference(uint32_t num_qubits);

  void reset();                       // |0...0>
  void apply(const Gate &g);
  void run(const Circuit &c);

  const std::vector<cdouble> &state() const { return state_; }
  uint32_t num_qubits() const { return n_; }
  size_t dim() const { return state_.size(); }

  double norm() const;                 // sqrt(sum |a|^2), should stay 1
  std::vector<double> probabilities() const;

  // <psi|P|psi> in double, the oracle for the GPU's fused Pauli reduction.
  double expectation(const PauliString &p) const;
  double expectation(const Hamiltonian &h) const;

 private:
  void apply_1q(uint32_t q, const cdouble m[4]);
  void apply_controlled_1q(uint32_t c, uint32_t t, const cdouble m[4]);
  void apply_swap(uint32_t a, uint32_t b);
  void apply_ccx(uint32_t c0, uint32_t c1, uint32_t t);
  void apply_diagonal_2q(uint32_t a, uint32_t b, cdouble phase_11);

  uint32_t n_;
  std::vector<cdouble> state_;
};

// Fill m with the 2x2 matrix of a single-qubit gate, row-major.
void gate_matrix_1q(GateKind k, const double *params, cdouble m[4]);

// Largest |a_ref - a_test| over all amplitudes. Sizes must match.
double max_amplitude_diff(const std::vector<cdouble> &a,
                          const std::vector<cdouble> &b);

// Compare against a single-precision interleaved buffer (re, im, re, im, ...),
// which is the layout the GPU path uses.
double max_amplitude_diff_f32(const std::vector<cdouble> &ref,
                              const float *interleaved, size_t count);

}  // namespace qmetal
