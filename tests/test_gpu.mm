// test_gpu.mm — every GPU kernel against the double-precision oracle.
//
// Nothing here compares the GPU to itself. Each case runs the same gate stream
// through the complex128 CPU reference and the Metal simulator and reports the
// largest amplitude difference. Deep circuits also serve as the measurement for
// open question 4: how far complex64 drifts as gate count grows.

#include <cmath>
#include <cstdio>
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
  test_benchmarks();
  test_error_growth();

  printf("\n%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
