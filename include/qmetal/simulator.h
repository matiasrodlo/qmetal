// simulator.h — GPU statevector simulator.
//
// Owns its Metal device, command queue, and state buffer. Gates are encoded
// lazily into an open command buffer and flushed on synchronize() or on any
// read, so a circuit costs one submission rather than one per gate.
//
// The public header is pure C++: no Metal types leak out, so callers do not
// need Objective-C. The implementation is Objective-C++ behind a pimpl.

#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "qmetal/circuit.h"
#include "qmetal/pauli.h"

namespace qmetal {

using cdouble = std::complex<double>;

// Largest state this build will attempt. 33 qubits is 64 GiB, which fits the
// 80.64 GiB maxBufferLength measured on an M4 Max; 34 does not fit anywhere.
constexpr uint32_t kMaxQubits = 33;

class Simulator {
 public:
  // Throws if Metal is unavailable, num_qubits is out of range, or the device
  // reports it cannot hold the state.
  explicit Simulator(uint32_t num_qubits);
  ~Simulator();

  Simulator(const Simulator &) = delete;
  Simulator &operator=(const Simulator &) = delete;

  void reset();                    // |0...0>
  void apply(const Gate &g);       // throws on an unsupported gate
  void run(const Circuit &c);
  void synchronize();              // flush and wait

  uint32_t num_qubits() const;
  size_t dim() const;

  // Amplitudes widened to complex128, for comparison against the reference.
  // Synchronizes first.
  std::vector<cdouble> amplitudes();

  // Zero-copy view of the interleaved float32 state (re, im, re, im, ...).
  // Valid because the buffer is in unified memory. Synchronizes first.
  const float *raw();

  // --- reductions, all GPU-resident -------------------------------------
  // The state never leaves the device for any of these. At 33 qubits a host
  // loop over 8.6 billion amplitudes is not an option.

  double abs2sum();  // sum |amp|^2
  double norm();     // sqrt(abs2sum())

  // <psi|P|psi> for a Pauli string, in one fused pass: no |P psi> is
  // materialised and no scratch state buffer is allocated. Real by
  // construction, since P is Hermitian.
  double expectation(const PauliString &p);
  double expectation(const Hamiltonian &h);  // one pass per term

  // --- sampling ----------------------------------------------------------
  // Draw measurement outcomes without collapsing the state and without reading
  // it back. Identical seeds give identical samples.
  std::vector<uint64_t> sample(size_t shots, uint64_t seed = 0);

  // Bitstring counts, little-endian (qubit 0 is the rightmost character).
  std::vector<std::pair<std::string, uint64_t>> counts(size_t shots,
                                                       uint64_t seed = 0);

  // Device facts, for reporting and preflight.
  static bool available();
  static std::string device_name();
  static uint64_t max_buffer_length();
  static uint64_t recommended_working_set();

  // Total kernel dispatches encoded since construction. The quantity every
  // optimization in this project exists to reduce.
  uint64_t dispatch_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qmetal
