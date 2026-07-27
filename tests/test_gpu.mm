// test_gpu.mm — every GPU kernel against the double-precision oracle.
//
// Nothing here compares the GPU to itself. Each case runs the same gate stream
// through the complex128 CPU reference and the Metal simulator and reports the
// largest amplitude difference. Deep circuits also serve as the measurement for
// open question 4: how far complex64 drifts as gate count grows.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "qmetal/circuits.h"
#include "qmetal/reference.h"
#include "qmetal/simulator.h"

using namespace qmetal;

static int failures = 0;
static int checks = 0;

static void check(bool ok, const std::string &what) {
  checks++;
  if (!ok) {
    failures++;
    printf("  FAIL  %s\n", what.c_str());
  }
}

static void check_diff(double diff, double tol, const std::string &what) {
  checks++;
  if (!(diff <= tol)) {
    failures++;
    printf("  FAIL  %s: diff %.3e > tol %.3e\n", what.c_str(), diff, tol);
  }
}

// Run one circuit through both engines; return max |a_ref - a_gpu|.
static double compare(const Circuit &c) {
  Reference r(c.num_qubits);
  r.run(c);
  Simulator s(c.num_qubits);
  s.run(c);
  auto got = s.amplitudes();
  return max_amplitude_diff(r.state(), got);
}

static const double kTol1 = 1e-5;  // single gates, complex64 floor

// ---------------------------------------------------------------------------

static void test_single_gates() {
  printf("single-qubit gates on every wire\n");
  const GateKind plain[] = {G_X, G_Y, G_Z, G_H, G_S, G_SDG, G_T, G_TDG};
  const GateKind param[] = {G_RX, G_RY, G_RZ, G_P};
  const uint32_t n = 5;

  for (GateKind k : plain) {
    for (uint32_t q = 0; q < n; q++) {
      Circuit c;
      c.num_qubits = n;
      // Spread amplitude first, so a gate cannot look correct merely by
      // acting on a mostly-empty state.
      for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
      c.add(g1p(G_T, 0, 0));  // no-op placeholder keeps arity checks honest
      c.ops.pop_back();
      c.add(g1(k, q));
      check_diff(compare(c), kTol1,
                 std::string(gate_name(k)) + " q=" + std::to_string(q));
    }
  }
  for (GateKind k : param) {
    for (uint32_t q = 0; q < n; q++) {
      Circuit c;
      c.num_qubits = n;
      for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
      c.add(g1p(k, q, 0.7853981633974483 + 0.3 * q));
      check_diff(compare(c), kTol1,
                 std::string(gate_name(k)) + " q=" + std::to_string(q));
    }
  }
}

static void test_two_qubit_gates() {
  printf("two-qubit gates on every ordered wire pair\n");
  const uint32_t n = 4;
  const GateKind kinds[] = {G_CX, G_CY, G_CZ, G_SWAP};
  for (GateKind k : kinds) {
    for (uint32_t a = 0; a < n; a++) {
      for (uint32_t b = 0; b < n; b++) {
        if (a == b) continue;
        Circuit c;
        c.num_qubits = n;
        for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
        c.add(g1p(G_RZ, 0, 0.4));  // break the symmetry of the uniform state
        c.add(g1p(G_RY, 2, 0.9));
        c.add(g2(k, a, b));
        check_diff(compare(c), kTol1,
                   std::string(gate_name(k)) + " (" + std::to_string(a) + "," +
                       std::to_string(b) + ")");
      }
    }
  }
  // CP over a range of angles, including ones where sign conventions bite.
  for (double th : {0.1, M_PI / 4, M_PI / 2, M_PI, -M_PI / 3}) {
    Circuit c;
    c.num_qubits = n;
    for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
    c.add(g2p(G_CP, 1, 3, th));
    check_diff(compare(c), kTol1, "cp theta=" + std::to_string(th));
  }
}

static void test_ccx() {
  printf("Toffoli on every wire triple\n");
  const uint32_t n = 4;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = 0; b < n; b++)
      for (uint32_t t = 0; t < n; t++) {
        if (a == b || a == t || b == t) continue;
        Circuit c;
        c.num_qubits = n;
        for (uint32_t i = 0; i < n; i++) c.add(g1(G_H, i));
        c.add(g1p(G_RZ, 0, 0.55));
        c.add(g3(G_CCX, a, b, t));
        check_diff(compare(c), kTol1,
                   "ccx (" + std::to_string(a) + "," + std::to_string(b) + "," +
                       std::to_string(t) + ")");
      }
}

static void test_benchmarks() {
  printf("frozen benchmark set, GPU vs oracle\n");
  printf("  %-18s %4s %8s %12s %12s\n", "circuit", "n", "gates", "max_diff",
         "norm_err");
  for (const auto &spec : frozen_benchmarks()) {
    for (uint32_t n : {6u, 10u, 14u}) {
      Circuit c = spec.build(n);
      Reference r(n);
      r.run(c);
      Simulator s(n);
      s.run(c);
      double diff = max_amplitude_diff(r.state(), s.amplitudes());
      double nerr = std::abs(s.norm() - 1.0);
      printf("  %-18s %4u %8zu %12.3e %12.3e\n", spec.name.c_str(), n, c.size(),
             diff, nerr);
      // Deep circuits accumulate complex64 error; the bound is generous but
      // still far tighter than anything that would hide a real bug.
      check_diff(diff, 1e-4, spec.name + " n=" + std::to_string(n));
      check_diff(nerr, 1e-4, spec.name + " n=" + std::to_string(n) + " norm");
    }
  }
}

static void test_error_growth() {
  printf("complex64 drift vs circuit depth (open question 4)\n");
  printf("  %6s %8s %12s\n", "depth", "gates", "max_diff");
  const uint32_t n = 12;
  for (uint32_t depth : {1u, 5u, 20u, 50u, 100u}) {
    Circuit c = tfim_trotter(n, depth);
    double d = compare(c);
    printf("  %6u %8zu %12.3e\n", depth, c.size(), d);
  }
}

static void test_fails_closed() {
  printf("fail-closed behaviour\n");
  bool threw = false;
  try {
    Simulator s(3);
    s.apply(g2(G_CX, 1, 1));
  } catch (const std::exception &) { threw = true; }
  check(threw, "CX on identical wires raises");

  threw = false;
  try {
    Simulator s(3);
    s.apply(g1(G_X, 7));
  } catch (const std::exception &) { threw = true; }
  check(threw, "out-of-range qubit raises");

  threw = false;
  try {
    Simulator s(kMaxQubits + 1);
  } catch (const std::exception &) { threw = true; }
  check(threw, "oversized simulator raises");
}

static void test_dispatch_count() {
  printf("dispatch accounting\n");
  Circuit c = ghz(8);
  Simulator s(8);
  uint64_t before = s.dispatch_count();  // one for the reset in the ctor
  s.run(c);
  s.synchronize();
  check(s.dispatch_count() - before == c.size(),
        "one dispatch per gate before any fusion exists");
}

static void test_expectation() {
  printf("Pauli expectation, GPU reduction vs oracle\n");
  {
    // Closed forms on GHZ: every single-qubit Z averages to zero, every ZZ
    // pair is perfectly correlated, and X on one wire alone is zero.
    const uint32_t n = 8;
    Simulator s(n);
    s.run(ghz(n));
    check_diff(std::abs(s.expectation(PauliString::single('Z', 0))), 1e-5,
               "GHZ <Z0> = 0");
    PauliString zz;
    zz.z_mask = 0b11;
    check_diff(std::abs(s.expectation(zz) - 1.0), 1e-5, "GHZ <Z0 Z1> = 1");
    check_diff(std::abs(s.expectation(PauliString::single('X', 0))), 1e-5,
               "GHZ <X0> = 0");
    PauliString allx;
    for (uint32_t q = 0; q < n; q++) allx.x_mask |= 1ull << q;
    check_diff(std::abs(s.expectation(allx) - 1.0), 1e-5, "GHZ <X...X> = 1");
    check_diff(std::abs(s.expectation(PauliString::identity()) - 1.0), 1e-5,
               "<I> = 1");
  }
  {
    // On |0...0>, <Z_q> = 1 for every wire.
    const uint32_t n = 6;
    Simulator s(n);
    for (uint32_t q = 0; q < n; q++)
      check_diff(std::abs(s.expectation(PauliString::single('Z', q)) - 1.0),
                 1e-6, "|0> <Z" + std::to_string(q) + "> = 1");
  }
  {
    // General case: random Pauli strings, including Y, against the oracle on a
    // state with no symmetry left to hide a sign error.
    const uint32_t n = 10;
    Circuit c = random_circuit(n, 4, 0xabc);
    Reference r(n);
    r.run(c);
    Simulator s(n);
    s.run(c);
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 24; trial++) {
      PauliString p;
      for (uint32_t q = 0; q < n; q++) {
        switch (rng() % 4) {
          case 1: p.x_mask |= 1ull << q; break;
          case 2: p.y_mask |= 1ull << q; break;
          case 3: p.z_mask |= 1ull << q; break;
          default: break;
        }
      }
      double want = r.expectation(p);
      double got = s.expectation(p);
      check_diff(std::abs(want - got), 1e-5,
                 "random pauli " + p.to_string(n) + " (ref " +
                     std::to_string(want) + ")");
    }
  }
  {
    // Hamiltonian sums, and norm via the same reduction path.
    const uint32_t n = 8;
    Circuit c = hardware_efficient(n, 2, 0xfeed);
    Reference r(n);
    r.run(c);
    Simulator s(n);
    s.run(c);
    Hamiltonian h;
    for (uint32_t q = 0; q + 1 < n; q++) {
      PauliString zz;
      zz.z_mask = (1ull << q) | (1ull << (q + 1));
      h.add(1.0, zz);
      h.add(0.3, PauliString::single('X', q));
    }
    check_diff(std::abs(r.expectation(h) - s.expectation(h)), 1e-4,
               "Hamiltonian sum of " + std::to_string(h.size()) + " terms");
    check_diff(std::abs(s.norm() - 1.0), 1e-5, "GPU norm reduction");
  }
}

static void test_sampling() {
  printf("sampling\n");
  {
    // GHZ has exactly two outcomes, at half each.
    const uint32_t n = 6;
    Simulator s(n);
    s.run(ghz(n));
    auto shots = s.sample(20000, 12345);
    size_t zeros = 0, ones = 0, other = 0;
    const uint64_t all = (1ull << n) - 1;
    for (uint64_t v : shots) {
      if (v == 0) zeros++;
      else if (v == all) ones++;
      else other++;
    }
    check(other == 0, "GHZ sampling produces only |0..0> and |1..1>");
    double frac = double(zeros) / double(zeros + ones);
    check(std::abs(frac - 0.5) < 0.02,
          "GHZ sampling is balanced (got " + std::to_string(frac) + ")");
  }
  {
    // Same seed, same stream. Different seed, different stream.
    Simulator s(5);
    s.run(qft(5));
    auto a = s.sample(500, 999);
    auto b = s.sample(500, 999);
    auto c = s.sample(500, 1000);
    check(a == b, "sampling is deterministic under a fixed seed");
    check(a != c, "a different seed gives a different stream");
  }
  {
    // The empirical distribution must track the exact probabilities. Compare
    // against the oracle, not against the simulator's own amplitudes.
    const uint32_t n = 6;
    Circuit circ = random_circuit(n, 3, 0x1234);
    Reference r(n);
    r.run(circ);
    auto probs = r.probabilities();
    Simulator s(n);
    s.run(circ);
    const size_t shots = 200000;
    auto draw = s.sample(shots, 2024);
    std::vector<double> hist(size_t(1) << n, 0.0);
    for (uint64_t v : draw) hist[v] += 1.0 / double(shots);
    double tv = 0.0;  // total variation distance
    for (size_t i = 0; i < hist.size(); i++) tv += std::abs(hist[i] - probs[i]);
    tv *= 0.5;
    printf("  total variation vs exact: %.4f (%zu shots)\n", tv, shots);
    // Sampling error alone is ~sqrt(2^n / shots) / 2; anything much above that
    // means the CDF walk is biased, not merely noisy.
    check(tv < 0.02, "sampled distribution matches exact probabilities");
  }
}

static void test_gradients() {
  printf("adjoint gradients vs parameter-shift on the oracle\n");

  // For U = exp(-i t P / 2) the parameter-shift rule is exact:
  //   dE/dt = [E(t + pi/2) - E(t - pi/2)] / 2
  // so the oracle here is an identity, not a finite-difference approximation.
  auto shifted_energy = [](Circuit c, size_t k, double delta,
                           const Hamiltonian &h) {
    c.ops[k].params[0] += delta;
    Reference r(c.num_qubits);
    r.run(c);
    return r.expectation(h);
  };

  struct Case { const char *name; Circuit c; };
  const uint32_t n = 6;
  std::vector<Case> cases = {
      {"hea_d2", hardware_efficient(n, 2, 0xc0ffee)},
      {"tfim_s3", tfim_trotter(n, 3)},
      {"qaoa_p2", qaoa_ring(n, 2)},
  };

  Hamiltonian h;
  for (uint32_t q = 0; q + 1 < n; q++) {
    PauliString zz;
    zz.z_mask = (1ull << q) | (1ull << (q + 1));
    h.add(1.0, zz);
  }
  for (uint32_t q = 0; q < n; q++) h.add(0.4, PauliString::single('X', q));
  // A Y term, so a sign error in the i^ny handling cannot hide.
  h.add(0.25, PauliString::single('Y', 0));

  for (auto &cs : cases) {
    Simulator s(n);
    auto [energy, grad] = s.energy_and_gradient(cs.c, h);

    Reference r(n);
    r.run(cs.c);
    check_diff(std::abs(energy - r.expectation(h)), 1e-4,
               std::string(cs.name) + " energy matches oracle");

    double worst = 0.0;
    size_t params = 0;
    for (size_t k = 0; k < cs.c.ops.size(); k++) {
      const Gate &g = cs.c.ops[k];
      if (g.num_params == 0) { 
        check(grad[k] == 0.0, std::string(cs.name) + " unparameterised gate is zero");
        continue;
      }
      params++;
      double want = 0.5 * (shifted_energy(cs.c, k, M_PI / 2, h) -
                           shifted_energy(cs.c, k, -M_PI / 2, h));
      worst = std::max(worst, std::abs(want - grad[k]));
    }
    printf("  %-10s %3zu params  max |adjoint - shift| = %.3e\n", cs.name,
           params, worst);
    check_diff(worst, 1e-4, std::string(cs.name) + " gradient matches shift");
  }

  {
    // Closed form: for H = Z0 and the circuit RY(t) on wire 0,
    // E(t) = cos(t) and dE/dt = -sin(t).
    for (double t : {0.0, 0.3, 1.1, 2.7, -0.9}) {
      Circuit c;
      c.num_qubits = 2;
      c.add(g1p(G_RY, 0, t));
      Hamiltonian hz;
      hz.add(1.0, PauliString::single('Z', 0));
      Simulator s(2);
      auto [e, gr] = s.energy_and_gradient(c, hz);
      check_diff(std::abs(e - std::cos(t)), 1e-5, "RY energy = cos(t)");
      check_diff(std::abs(gr[0] + std::sin(t)), 1e-5, "RY gradient = -sin(t)");
    }
  }

  {
    // Fail closed: a parameterised gate with no native generator raises rather
    // than silently contributing zero to the gradient.
    Circuit c;
    c.num_qubits = 2;
    c.add(g1(G_H, 0));
    c.add(g2p(G_CP, 0, 1, 0.4));
    Hamiltonian hz;
    hz.add(1.0, PauliString::single('Z', 0));
    bool threw = false;
    try {
      Simulator s(2);
      s.gradient(c, hz);
    } catch (const std::exception &) { threw = true; }
    check(threw, "non-differentiable parameterised gate raises");
  }
}

int main() {
  if (!Simulator::available()) {
    printf("no Metal device; skipping\n");
    return 0;
  }
  printf("device : %s\n", Simulator::device_name().c_str());
  printf("maxbuf : %.2f GiB\n\n",
         Simulator::max_buffer_length() / 1073741824.0);

  test_single_gates();
  test_two_qubit_gates();
  test_ccx();
  test_fails_closed();
  test_dispatch_count();
  test_expectation();
  test_sampling();
  test_gradients();
  test_benchmarks();
  test_error_growth();

  printf("\n%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
