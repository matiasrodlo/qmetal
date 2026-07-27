// test_fusion.cpp — fusion passes against the oracle, with no GPU involved.
//
// Fusion is where a simulator quietly goes wrong: the fused path is faster, the
// norm still looks like 1, and the amplitudes are subtly different. So every
// pass is checked here in double precision against the same circuit executed
// gate by gate. A pass that changes the unitary fails, regardless of how much
// faster it made things.
//
// Because the pipeline has no Metal dependency, these run anywhere.

#include <cmath>
#include <cstdio>
#include <string>

#include "qmetal/circuits.h"
#include "qmetal/plan.h"
#include "qmetal/reference.h"

using namespace qmetal;

static int failures = 0;
static int checks = 0;

static void check(bool ok, const std::string &what) {
  checks++;
  if (!ok) { failures++; printf("  FAIL  %s\n", what.c_str()); }
}

static void check_diff(double d, double tol, const std::string &what) {
  checks++;
  if (!(d <= tol)) {
    failures++;
    printf("  FAIL  %s: diff %.3e > %.3e\n", what.c_str(), d, tol);
  }
}

// Every plan must reproduce the circuit exactly, in double.
static double plan_error(const Circuit &c, const FusionOptions &opts) {
  Reference a(c.num_qubits);
  a.run(c);
  Reference b(c.num_qubits);
  b.run(build_plan(c, opts));
  return max_amplitude_diff(a.state(), b.state());
}

static const double kTol = 1e-12;

static void test_lowering_is_faithful() {
  printf("unfused lowering is one-to-one\n");
  for (const auto &spec : frozen_benchmarks()) {
    for (uint32_t n : {5u, 9u}) {
      Circuit c = spec.build(n);
      Plan p = build_plan(c, FusionOptions::none());
      check(p.size() == c.size(),
            spec.name + " n=" + std::to_string(n) + " unfused size matches");
      check_diff(plan_error(c, FusionOptions::none()), kTol,
                 spec.name + " n=" + std::to_string(n) + " unfused exact");
    }
  }
}

static void test_fusion_preserves_unitary() {
  printf("fused plans reproduce the circuit\n");
  const FusionOptions all;
  const FusionOptions diag_only{false, true, true, 4, 0};
  const FusionOptions win_only{true, false, false, 4, 0};
  const FusionOptions blocked = FusionOptions::blocked(8);
  for (const auto &spec : frozen_benchmarks()) {
    for (uint32_t n : {4u, 7u, 11u}) {
      Circuit c = spec.build(n);
      std::string tag = spec.name + " n=" + std::to_string(n);
      check_diff(plan_error(c, diag_only), kTol, tag + " diagonals only");
      check_diff(plan_error(c, win_only), kTol, tag + " 1q windows only");
      check_diff(plan_error(c, all), kTol, tag + " both passes");
      check_diff(plan_error(c, blocked), kTol, tag + " with cache blocking");
    }
  }
}

static void test_algebraic_cancellation() {
  printf("windows that cancel emit nothing\n");
  {
    Circuit c;
    c.num_qubits = 3;
    for (int i = 0; i < 4; i++) { c.add(g1(G_H, 1)); }  // H^4 = I
    Plan p = build_plan(c, FusionOptions{});
    check(p.size() == 0, "H^4 collapses away entirely");
  }
  {
    // A genuine small rotation must survive. Using a relative tolerance here
    // would discard it, which is the bug this guards against.
    Circuit c;
    c.num_qubits = 2;
    c.add(g1p(G_RY, 0, 1e-6));
    c.add(g1p(G_RY, 0, 1e-6));
    Plan p = build_plan(c, FusionOptions{});
    check(p.size() == 1, "RY(1e-6) twice survives as one op");
    Reference r(2);
    r.run(p);
    Reference want(2);
    want.run(c);
    check_diff(max_amplitude_diff(want.state(), r.state()), kTol,
               "small-angle window is exact");
  }
  {
    // X X = I on one wire while another wire keeps a real gate.
    Circuit c;
    c.num_qubits = 2;
    c.add(g1(G_X, 0));
    c.add(g1(G_H, 1));
    c.add(g1(G_X, 0));
    Plan p = build_plan(c, FusionOptions{});
    check(p.size() == 1, "cancelling wire drops, other wire kept");
  }
}

static void test_diagonal_algebra() {
  printf("diagonal run collection\n");
  {
    // A phase ladder is one op regardless of length.
    Circuit c;
    c.num_qubits = 8;
    for (uint32_t k = 1; k < 8; k++)
      c.add(g2p(G_CP, k, 0, M_PI / double(1u << k)));
    Plan p = build_plan(c, FusionOptions{});
    check(p.size() == 1, "7-gate phase ladder becomes one op");
    check(p.ops[0].term_count() == 7, "with all seven terms retained");
    check_diff(plan_error(c, FusionOptions{}), kTol, "ladder is exact");
  }
  {
    // Mixed 1q and 2q diagonals fuse together; a non-diagonal gate breaks the
    // run, and the two halves stay in order.
    Circuit c;
    c.num_qubits = 4;
    c.add(g1(G_T, 0));
    c.add(g2(G_CZ, 0, 1));
    c.add(g1p(G_RZ, 2, 0.3));
    c.add(g1(G_H, 3));            // breaks the run
    c.add(g1(G_S, 1));
    c.add(g2p(G_CP, 2, 3, 0.9));
    Plan p = build_plan(c, FusionOptions{});
    check(p.size() == 3, "diag | H | diag");
    check(p.ops[0].term_count() == 3, "first run holds three terms");
    check(p.ops[2].term_count() == 2, "second run holds two");
    check_diff(plan_error(c, FusionOptions{}), kTol, "mixed diagonals exact");
  }
  {
    // RZ carries a global phase that must not be dropped. Alone it is
    // unobservable, but it stops being so the moment the state is compared
    // amplitude by amplitude, so the plan tracks it and stays exactly equal
    // rather than equal-up-to-phase.
    Circuit c;
    c.num_qubits = 2;
    c.add(g1(G_H, 0));
    c.add(g1p(G_RZ, 0, 1.234));   // diagonal run of two with the CZ below
    c.add(g2(G_CZ, 0, 1));
    Plan p = build_plan(c, FusionOptions{});
    check(p.size() == 2, "H then a two-gate diagonal run");
    check(std::abs(p.ops[1].global_phase + 0.617) < 1e-12,
          "RZ global phase retained");
    check_diff(plan_error(c, FusionOptions{}), kTol, "RZ phase exact");
  }
}

static void report_reduction() {
  printf("\ndispatch reduction (the quantity fusion exists to cut)\n");
  printf("%-18s %5s %8s %8s %8s %8s\n", "circuit", "n", "gates", "unfused",
         "fused", "ratio");
  for (const auto &spec : frozen_benchmarks()) {
    for (uint32_t n : {24u, 33u}) {
      Circuit c = spec.build(n);
      size_t un = build_plan(c, FusionOptions::none()).size();
      size_t fu = build_plan(c, FusionOptions{}).size();
      printf("%-18s %5u %8zu %8zu %8zu %7.2fx\n", spec.name.c_str(), n,
             c.size(), un, fu, fu ? double(un) / double(fu) : 0.0);
    }
  }
}

int main() {
  test_lowering_is_faithful();
  test_fusion_preserves_unitary();
  test_algebraic_cancellation();
  test_diagonal_algebra();
  printf("\n%d checks, %d failures\n", checks, failures);
  if (failures == 0) report_reduction();
  return failures == 0 ? 0 : 1;
}
