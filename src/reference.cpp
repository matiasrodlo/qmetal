#include "qmetal/reference.h"

#include <cmath>
#include <stdexcept>

namespace qmetal {

Gate g1(GateKind k, uint32_t q) { return Gate{k, 1, 0, {q, 0, 0}, {0, 0, 0}}; }
Gate g1p(GateKind k, uint32_t q, double t) {
  return Gate{k, 1, 1, {q, 0, 0}, {t, 0, 0}};
}
Gate g2(GateKind k, uint32_t a, uint32_t b) {
  return Gate{k, 2, 0, {a, b, 0}, {0, 0, 0}};
}
Gate g2p(GateKind k, uint32_t a, uint32_t b, double t) {
  return Gate{k, 2, 1, {a, b, 0}, {t, 0, 0}};
}
Gate g3(GateKind k, uint32_t a, uint32_t b, uint32_t c) {
  return Gate{k, 3, 0, {a, b, c}, {0, 0, 0}};
}

size_t Circuit::count_1q() const {
  size_t n = 0;
  for (const auto &g : ops)
    if (g.num_qubits == 1) n++;
  return n;
}
size_t Circuit::count_2q() const {
  size_t n = 0;
  for (const auto &g : ops)
    if (g.num_qubits == 2) n++;
  return n;
}

const char *gate_name(GateKind k) {
  switch (k) {
    case G_X: return "x";
    case G_Y: return "y";
    case G_Z: return "z";
    case G_H: return "h";
    case G_S: return "s";
    case G_SDG: return "sdg";
    case G_T: return "t";
    case G_TDG: return "tdg";
    case G_RX: return "rx";
    case G_RY: return "ry";
    case G_RZ: return "rz";
    case G_P: return "p";
    case G_CX: return "cx";
    case G_CY: return "cy";
    case G_CZ: return "cz";
    case G_SWAP: return "swap";
    case G_CP: return "cp";
    case G_CCX: return "ccx";
    default: return "?";
  }
}

// Row-major: m[0] m[1] / m[2] m[3]
void gate_matrix_1q(GateKind k, const double *p, cdouble m[4]) {
  const double s2 = 1.0 / std::sqrt(2.0);
  switch (k) {
    case G_X: m[0] = 0; m[1] = 1; m[2] = 1; m[3] = 0; return;
    case G_Y: m[0] = 0; m[1] = cdouble(0, -1); m[2] = cdouble(0, 1); m[3] = 0; return;
    case G_Z: m[0] = 1; m[1] = 0; m[2] = 0; m[3] = -1; return;
    case G_H: m[0] = s2; m[1] = s2; m[2] = s2; m[3] = -s2; return;
    case G_S: m[0] = 1; m[1] = 0; m[2] = 0; m[3] = cdouble(0, 1); return;
    case G_SDG: m[0] = 1; m[1] = 0; m[2] = 0; m[3] = cdouble(0, -1); return;
    case G_T:
      m[0] = 1; m[1] = 0; m[2] = 0;
      m[3] = std::polar(1.0, M_PI / 4);
      return;
    case G_TDG:
      m[0] = 1; m[1] = 0; m[2] = 0;
      m[3] = std::polar(1.0, -M_PI / 4);
      return;
    case G_RX: {
      double c = std::cos(p[0] / 2), s = std::sin(p[0] / 2);
      m[0] = c; m[1] = cdouble(0, -s); m[2] = cdouble(0, -s); m[3] = c;
      return;
    }
    case G_RY: {
      double c = std::cos(p[0] / 2), s = std::sin(p[0] / 2);
      m[0] = c; m[1] = -s; m[2] = s; m[3] = c;
      return;
    }
    case G_RZ: {
      m[0] = std::polar(1.0, -p[0] / 2); m[1] = 0; m[2] = 0;
      m[3] = std::polar(1.0, p[0] / 2);
      return;
    }
    case G_P:
      m[0] = 1; m[1] = 0; m[2] = 0; m[3] = std::polar(1.0, p[0]);
      return;
    default:
      throw std::runtime_error("gate_matrix_1q: not a 1-qubit gate");
  }
}

Reference::Reference(uint32_t n) : n_(n) {
  if (n == 0 || n > 30) throw std::runtime_error("Reference: n out of range");
  state_.resize(size_t(1) << n);
  reset();
}

void Reference::reset() {
  std::fill(state_.begin(), state_.end(), cdouble(0, 0));
  state_[0] = cdouble(1, 0);
}

// LSB-first: qubit q has stride 1 << q. Iterate over the 2^(n-1) pairs by
// inserting a zero bit at position q, exactly as the GPU kernel does.
void Reference::apply_1q(uint32_t q, const cdouble m[4]) {
  const size_t stride = size_t(1) << q;
  const size_t pairs = state_.size() >> 1;
  for (size_t t = 0; t < pairs; t++) {
    size_t i = ((t >> q) << (q + 1)) | (t & (stride - 1));
    size_t j = i | stride;
    cdouble a = state_[i], b = state_[j];
    state_[i] = m[0] * a + m[1] * b;
    state_[j] = m[2] * a + m[3] * b;
  }
}

void Reference::apply_controlled_1q(uint32_t c, uint32_t t, const cdouble m[4]) {
  const size_t cmask = size_t(1) << c;
  const size_t tstride = size_t(1) << t;
  for (size_t i = 0; i < state_.size(); i++) {
    if ((i & cmask) == 0) continue;
    if ((i & tstride) != 0) continue;  // visit each pair once, from the 0 side
    size_t j = i | tstride;
    cdouble a = state_[i], b = state_[j];
    state_[i] = m[0] * a + m[1] * b;
    state_[j] = m[2] * a + m[3] * b;
  }
}

void Reference::apply_diagonal_2q(uint32_t a, uint32_t b, cdouble phase_11) {
  const size_t ma = size_t(1) << a, mb = size_t(1) << b;
  for (size_t i = 0; i < state_.size(); i++)
    if ((i & ma) && (i & mb)) state_[i] *= phase_11;
}

void Reference::apply_swap(uint32_t a, uint32_t b) {
  if (a == b) return;
  const size_t ma = size_t(1) << a, mb = size_t(1) << b;
  for (size_t i = 0; i < state_.size(); i++) {
    // Swap only the |10> partner, once, to avoid double-swapping.
    if ((i & ma) && !(i & mb)) std::swap(state_[i], state_[(i ^ ma) | mb]);
  }
}

void Reference::apply_ccx(uint32_t c0, uint32_t c1, uint32_t t) {
  const size_t m0 = size_t(1) << c0, m1 = size_t(1) << c1,
               mt = size_t(1) << t;
  for (size_t i = 0; i < state_.size(); i++) {
    if ((i & m0) && (i & m1) && !(i & mt)) std::swap(state_[i], state_[i | mt]);
  }
}

void Reference::apply(const Gate &g) {
  for (int k = 0; k < g.num_qubits; k++)
    if (g.qubits[k] >= n_) throw std::runtime_error("qubit index out of range");

  if (g.num_qubits == 1) {
    cdouble m[4];
    gate_matrix_1q(g.kind, g.params, m);
    apply_1q(g.qubits[0], m);
    return;
  }
  if (g.num_qubits == 2) {
    if (g.qubits[0] == g.qubits[1])
      throw std::runtime_error("2-qubit gate on identical wires");
    switch (g.kind) {
      case G_CX: {
        cdouble m[4]; gate_matrix_1q(G_X, nullptr, m);
        apply_controlled_1q(g.qubits[0], g.qubits[1], m); return;
      }
      case G_CY: {
        cdouble m[4]; gate_matrix_1q(G_Y, nullptr, m);
        apply_controlled_1q(g.qubits[0], g.qubits[1], m); return;
      }
      case G_CZ:
        apply_diagonal_2q(g.qubits[0], g.qubits[1], cdouble(-1, 0)); return;
      case G_CP:
        apply_diagonal_2q(g.qubits[0], g.qubits[1],
                          std::polar(1.0, g.params[0])); return;
      case G_SWAP:
        apply_swap(g.qubits[0], g.qubits[1]); return;
      default:
        throw std::runtime_error("unsupported 2-qubit gate");
    }
  }
  if (g.num_qubits == 3 && g.kind == G_CCX) {
    apply_ccx(g.qubits[0], g.qubits[1], g.qubits[2]);
    return;
  }
  // Fail closed. Never silently apply identity.
  throw std::runtime_error(std::string("unsupported gate: ") +
                           gate_name(g.kind));
}

void Reference::run(const Circuit &c) {
  if (c.num_qubits != n_) throw std::runtime_error("circuit/reference size mismatch");
  for (const auto &g : c.ops) apply(g);
}

double Reference::norm() const {
  double s = 0.0;
  for (const auto &a : state_) s += std::norm(a);
  return std::sqrt(s);
}

std::vector<double> Reference::probabilities() const {
  std::vector<double> p(state_.size());
  for (size_t i = 0; i < state_.size(); i++) p[i] = std::norm(state_[i]);
  return p;
}

double max_amplitude_diff(const std::vector<cdouble> &a,
                          const std::vector<cdouble> &b) {
  if (a.size() != b.size()) throw std::runtime_error("size mismatch");
  double m = 0.0;
  for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double max_amplitude_diff_f32(const std::vector<cdouble> &ref,
                              const float *interleaved, size_t count) {
  if (ref.size() != count) throw std::runtime_error("size mismatch");
  double m = 0.0;
  for (size_t i = 0; i < count; i++) {
    cdouble got(interleaved[2 * i], interleaved[2 * i + 1]);
    m = std::max(m, std::abs(ref[i] - got));
  }
  return m;
}

}  // namespace qmetal
