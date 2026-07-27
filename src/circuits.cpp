#include "qmetal/circuits.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace qmetal {
namespace {

// exp(-i*theta*Z_a Z_b) as CX / RZ / CX, so the gate set stays minimal.
void zz(Circuit &c, uint32_t a, uint32_t b, double theta) {
  c.add(g2(G_CX, a, b));
  c.add(g1p(G_RZ, b, 2.0 * theta));
  c.add(g2(G_CX, a, b));
}

void require(bool ok, const char *msg) {
  if (!ok) throw std::runtime_error(msg);
}

}  // namespace

Circuit ghz(uint32_t n) {
  require(n >= 2, "ghz: need at least 2 qubits");
  Circuit c;
  c.num_qubits = n;
  c.add(g1(G_H, 0));
  for (uint32_t i = 0; i + 1 < n; i++) c.add(g2(G_CX, i, i + 1));
  return c;
}

Circuit qft(uint32_t n) {
  require(n >= 1, "qft: need at least 1 qubit");
  Circuit c;
  c.num_qubits = n;
  // Descending, because indexing is LSB-first: the textbook ascending loop
  // treats qubit 0 as the most significant bit, which under this convention
  // computes QFT composed with a bit reversal on the *input* rather than the
  // QFT. Verified by hand at n=2: ascending sends |1> to (1,-1,1,-1)/2, which
  // is QFT|2>, not QFT|1>.
  for (uint32_t j = n; j-- > 0;) {
    c.add(g1(G_H, j));
    for (uint32_t k = j; k-- > 0;)
      c.add(g2p(G_CP, k, j, M_PI / double(uint64_t(1) << (j - k))));
  }
  return c;
}

Circuit qaoa_ring(uint32_t n, uint32_t p) {
  require(n >= 3, "qaoa_ring: need at least 3 qubits");
  Circuit c;
  c.num_qubits = n;
  for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
  for (uint32_t layer = 0; layer < p; layer++) {
    // Angles vary per layer so no two layers are accidentally identical.
    double gamma = 0.3 + 0.11 * layer;
    double beta = 0.7 - 0.05 * layer;
    for (uint32_t i = 0; i < n; i++) zz(c, i, (i + 1) % n, gamma);
    for (uint32_t i = 0; i < n; i++) c.add(g1p(G_RX, i, 2.0 * beta));
  }
  return c;
}

Circuit tfim_trotter(uint32_t n, uint32_t steps) {
  require(n >= 2, "tfim_trotter: need at least 2 qubits");
  Circuit c;
  c.num_qubits = n;
  const double dt = 0.1, J = 1.0, h = 0.8;
  for (uint32_t s = 0; s < steps; s++) {
    for (uint32_t i = 0; i + 1 < n; i++) zz(c, i, i + 1, J * dt);
    for (uint32_t i = 0; i < n; i++) c.add(g1p(G_RX, i, 2.0 * h * dt));
  }
  return c;
}

Circuit grover_proxy(uint32_t n, uint32_t iters) {
  require(n >= 2, "grover_proxy: need at least 2 qubits");
  Circuit c;
  c.num_qubits = n;
  for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
  for (uint32_t it = 0; it < iters; it++) {
    // Oracle: mark a state with a CZ on the top pair.
    c.add(g2(G_CZ, n - 2, n - 1));
    // Diffusion: H, X, CZ, X, H.
    for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
    for (uint32_t i = 0; i < n; i++) c.add(g1(G_X, i));
    c.add(g2(G_CZ, n - 2, n - 1));
    for (uint32_t i = 0; i < n; i++) c.add(g1(G_X, i));
    for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
  }
  return c;
}

Circuit random_circuit(uint32_t n, uint32_t depth, uint64_t seed) {
  require(n >= 2, "random_circuit: need at least 2 qubits");
  Circuit c;
  c.num_qubits = n;
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> pick(0, 4);
  std::uniform_real_distribution<double> ang(0.0, 2.0 * M_PI);
  const GateKind pool[5] = {G_RX, G_RY, G_RZ, G_H, G_T};
  for (uint32_t d = 0; d < depth; d++) {
    for (uint32_t q = 0; q < n; q++) {
      GateKind k = pool[pick(rng)];
      if (k == G_RX || k == G_RY || k == G_RZ)
        c.add(g1p(k, q, ang(rng)));
      else
        c.add(g1(k, q));
    }
    // Brickwork: offset alternates, so the pairing shifts every layer.
    for (uint32_t i = d % 2; i + 1 < n; i += 2) c.add(g2(G_CX, i, i + 1));
  }
  return c;
}

Circuit hardware_efficient(uint32_t n, uint32_t depth, uint64_t seed) {
  require(n >= 2, "hardware_efficient: need at least 2 qubits");
  Circuit c;
  c.num_qubits = n;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> ang(-M_PI, M_PI);
  for (uint32_t d = 0; d < depth; d++) {
    for (uint32_t q = 0; q < n; q++) {
      c.add(g1p(G_RY, q, ang(rng)));
      c.add(g1p(G_RZ, q, ang(rng)));
    }
    for (uint32_t i = 0; i + 1 < n; i++) c.add(g2(G_CX, i, i + 1));
  }
  return c;
}

Circuit local_patch(uint32_t n, uint32_t width, uint32_t rounds, uint64_t seed) {
  require(n >= 4, "local_patch: need at least 4 qubits");
  const uint32_t w = std::min(width, n);
  Circuit c;
  c.num_qubits = n;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> ang(-M_PI, M_PI);
  for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
  for (uint32_t r = 0; r < rounds; r++) {
    // Dense activity inside the work register: several 1q layers so the run is
    // long enough for staging to pay, plus local entanglement.
    for (uint32_t rep = 0; rep < 3; rep++) {
      for (uint32_t q = 0; q < w; q++) {
        c.add(g1p(G_RY, q, ang(rng)));
        c.add(g1p(G_RZ, q, ang(rng)));
      }
    }
    for (uint32_t q = 0; q + 1 < w; q++) c.add(g2(G_CX, q, q + 1));
    // One coupling out of the register per round, rotating, so the circuit is
    // genuinely n-qubit rather than a small problem padded with idle wires.
    if (w < n) c.add(g2(G_CX, 0, w + (r % (n - w))));
  }
  return c;
}

// Fixed depths. These are part of the frozen set; changing one invalidates
// every measurement taken against it.
namespace {
Circuit b_ghz(uint32_t n) { return ghz(n); }
Circuit b_qft(uint32_t n) { return qft(n); }
Circuit b_qaoa(uint32_t n) { return qaoa_ring(n, 6); }
Circuit b_tfim(uint32_t n) { return tfim_trotter(n, 20); }
Circuit b_grover(uint32_t n) { return grover_proxy(n, 4); }
Circuit b_random(uint32_t n) { return random_circuit(n, 10); }
Circuit b_hea(uint32_t n) { return hardware_efficient(n, 3); }
Circuit b_patch(uint32_t n) { return local_patch(n, 8, 12); }
}  // namespace

const std::vector<BenchmarkSpec> &frozen_benchmarks() {
  static const std::vector<BenchmarkSpec> specs = {
      {"ghz", b_ghz},
      {"qft", b_qft},
      {"qaoa_ring_p6", b_qaoa},
      {"tfim_trotter_20", b_tfim},
      {"grover_proxy_4", b_grover},
      {"random_d10", b_random},
      {"hea_d3", b_hea},
  };
  return specs;
}

const std::vector<BenchmarkSpec> &extended_benchmarks() {
  static const std::vector<BenchmarkSpec> specs = [] {
    std::vector<BenchmarkSpec> v = frozen_benchmarks();
    v.push_back({"local_patch_w8", b_patch});
    return v;
  }();
  return specs;
}

}  // namespace qmetal
