// test_reference.cpp — validate the correctness oracle against closed forms.
//
// The reference is what every GPU kernel will be judged against, so it cannot
// itself be taken on trust. Everything here is checked against an analytically
// known answer, an algebraic identity, or an independently computed value --
// never against another run of the same code.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "qmetal/circuits.h"
#include "qmetal/reference.h"

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

static void check_near(double got, double want, double tol,
                       const std::string &what) {
  checks++;
  if (!(std::abs(got - want) <= tol)) {
    failures++;
    printf("  FAIL  %s: got %.12g want %.12g (tol %g)\n", what.c_str(), got,
           want, tol);
  }
}

static const double kTol = 1e-12;

// ---------------------------------------------------------------------------

static void test_bit_convention() {
  printf("bit convention (LSB-first)\n");
  // X on qubit 0 must move amplitude to index 1; on qubit 1, to index 2.
  for (uint32_t q = 0; q < 3; q++) {
    Reference r(3);
    r.apply(g1(G_X, q));
    size_t expect = size_t(1) << q;
    check_near(std::abs(r.state()[expect]), 1.0,
               kTol, "X(q=" + std::to_string(q) + ") lands at index 1<<q");
  }
}

static void test_single_qubit_identities() {
  printf("single-qubit algebra\n");
  // H*H = I, X*X = I, S*S = Z, T^8 = I, RZ(2pi) = -I.
  {
    Reference r(1);
    r.apply(g1(G_H, 0)); r.apply(g1(G_H, 0));
    check_near(std::abs(r.state()[0]), 1.0, kTol, "H^2 = I");
  }
  {
    Reference r(1);
    r.apply(g1(G_X, 0)); r.apply(g1(G_X, 0));
    check_near(std::abs(r.state()[0]), 1.0, kTol, "X^2 = I");
  }
  {
    // On |1>: S^2 should equal Z, i.e. amplitude -1.
    Reference a(1), b(1);
    a.apply(g1(G_X, 0)); a.apply(g1(G_S, 0)); a.apply(g1(G_S, 0));
    b.apply(g1(G_X, 0)); b.apply(g1(G_Z, 0));
    check_near(max_amplitude_diff(a.state(), b.state()), 0.0, kTol,
               "S^2 = Z on |1>");
  }
  {
    Reference r(1);
    r.apply(g1(G_X, 0));
    for (int i = 0; i < 8; i++) r.apply(g1(G_T, 0));
    check_near(r.state()[1].real(), 1.0, kTol, "T^8 = I on |1>");
  }
  {
    Reference r(1);
    r.apply(g1p(G_RZ, 0, 2 * M_PI));
    check_near(r.state()[0].real(), -1.0, kTol, "RZ(2pi) = -I");
  }
  {
    // RY(pi) on |0> gives |1>.
    Reference r(1);
    r.apply(g1p(G_RY, 0, M_PI));
    check_near(std::abs(r.state()[1]), 1.0, kTol, "RY(pi)|0> = |1>");
  }
  {
    // RX(pi)|0> = -i|1>.
    Reference r(1);
    r.apply(g1p(G_RX, 0, M_PI));
    check_near(r.state()[1].imag(), -1.0, kTol, "RX(pi)|0> = -i|1>");
  }
  {
    // SDG undoes S, TDG undoes T.
    Reference r(1);
    r.apply(g1(G_X, 0));
    r.apply(g1(G_S, 0)); r.apply(g1(G_SDG, 0));
    r.apply(g1(G_T, 0)); r.apply(g1(G_TDG, 0));
    check_near(r.state()[1].real(), 1.0, kTol, "S SDG T TDG = I");
  }
}

static void test_two_qubit() {
  printf("two-qubit gates\n");
  {
    // Bell state: (|00> + |11>)/sqrt(2).
    Reference r(2);
    r.apply(g1(G_H, 0));
    r.apply(g2(G_CX, 0, 1));
    const double s = 1.0 / std::sqrt(2.0);
    check_near(r.state()[0].real(), s, kTol, "Bell |00>");
    check_near(r.state()[3].real(), s, kTol, "Bell |11>");
    check_near(std::abs(r.state()[1]), 0.0, kTol, "Bell |01> empty");
    check_near(std::abs(r.state()[2]), 0.0, kTol, "Bell |10> empty");
  }
  {
    // CX applied twice is the identity.
    Reference r(2);
    r.apply(g1(G_H, 0)); r.apply(g1(G_H, 1));
    auto before = r.state();
    r.apply(g2(G_CX, 0, 1)); r.apply(g2(G_CX, 0, 1));
    check_near(max_amplitude_diff(before, r.state()), 0.0, kTol, "CX^2 = I");
  }
  {
    // CX control and target are not symmetric.
    Reference a(2), b(2);
    a.apply(g1(G_X, 0)); a.apply(g2(G_CX, 0, 1));
    b.apply(g1(G_X, 0)); b.apply(g2(G_CX, 1, 0));
    check(max_amplitude_diff(a.state(), b.state()) > 0.5,
          "CX(0,1) differs from CX(1,0)");
    check_near(std::abs(a.state()[3]), 1.0, kTol, "CX(0,1) on |01> -> |11>");
  }
  {
    // CZ is diagonal and marks only |11>.
    Reference r(2);
    for (uint32_t q = 0; q < 2; q++) r.apply(g1(G_H, q));
    r.apply(g2(G_CZ, 0, 1));
    check_near(r.state()[3].real(), -0.5, kTol, "CZ flips |11> sign");
    check_near(r.state()[0].real(), 0.5, kTol, "CZ leaves |00>");
  }
  {
    // CP(pi) equals CZ.
    Reference a(2), b(2);
    for (uint32_t q = 0; q < 2; q++) { a.apply(g1(G_H, q)); b.apply(g1(G_H, q)); }
    a.apply(g2p(G_CP, 0, 1, M_PI));
    b.apply(g2(G_CZ, 0, 1));
    check_near(max_amplitude_diff(a.state(), b.state()), 0.0, kTol,
               "CP(pi) = CZ");
  }
  {
    // SWAP exchanges wires, and is its own inverse.
    Reference r(2);
    r.apply(g1(G_X, 0));
    r.apply(g2(G_SWAP, 0, 1));
    check_near(std::abs(r.state()[2]), 1.0, kTol, "SWAP |01> -> |10>");
    r.apply(g2(G_SWAP, 0, 1));
    check_near(std::abs(r.state()[1]), 1.0, kTol, "SWAP^2 = I");
  }
  {
    // Toffoli fires only when both controls are set.
    Reference r(3);
    r.apply(g1(G_X, 0)); r.apply(g1(G_X, 1));
    r.apply(g3(G_CCX, 0, 1, 2));
    check_near(std::abs(r.state()[7]), 1.0, kTol, "CCX |011> -> |111>");
    Reference s(3);
    s.apply(g1(G_X, 0));
    s.apply(g3(G_CCX, 0, 1, 2));
    check_near(std::abs(s.state()[1]), 1.0, kTol, "CCX inert with one control");
  }
}

static void test_ghz() {
  printf("GHZ closed form\n");
  for (uint32_t n : {2u, 3u, 8u, 12u}) {
    Reference r(n);
    r.run(ghz(n));
    const double s = 1.0 / std::sqrt(2.0);
    check_near(r.state()[0].real(), s, kTol,
               "GHZ n=" + std::to_string(n) + " |0..0>");
    check_near(r.state()[(size_t(1) << n) - 1].real(), s, kTol,
               "GHZ n=" + std::to_string(n) + " |1..1>");
    double rest = 0.0;
    for (size_t i = 1; i + 1 < r.dim(); i++) rest += std::norm(r.state()[i]);
    check_near(rest, 0.0, kTol, "GHZ n=" + std::to_string(n) + " nothing else");
  }
}

static void test_qft() {
  printf("QFT closed form\n");
  // QFT|0...0> is the uniform superposition, with zero phase everywhere. This
  // holds regardless of the missing swap layer, since uniform is symmetric.
  for (uint32_t n : {1u, 2u, 5u, 10u}) {
    Reference r(n);
    r.run(qft(n));
    const double amp = 1.0 / std::sqrt(double(size_t(1) << n));
    double worst = 0.0;
    for (size_t i = 0; i < r.dim(); i++)
      worst = std::max(worst, std::abs(r.state()[i] - cdouble(amp, 0)));
    check_near(worst, 0.0, 1e-12, "QFT|0> uniform, n=" + std::to_string(n));
  }
  // Phase-sensitive case, derived independently rather than recorded from a
  // run. The true QFT sends |x=1> to (1, i, -1, -i)/2. This circuit omits the
  // trailing swap layer, so its output is bit-reversed: out[k] = QFT[rev(k)],
  // giving (1, -1, i, -i)/2 at n=2. Both facts are checked below.
  {
    Reference r(2);
    r.apply(g1(G_X, 0));
    r.run(qft(2));
    const cdouble want[4] = {cdouble(0.5, 0), cdouble(-0.5, 0),
                             cdouble(0, 0.5), cdouble(0, -0.5)};
    double worst = 0.0;
    for (int i = 0; i < 4; i++)
      worst = std::max(worst, std::abs(r.state()[i] - want[i]));
    check_near(worst, 0.0, 1e-12, "QFT|1> phases, n=2 (bit-reversed output)");
  }
  // Same statement at n=3, reconstructed from the definition rather than
  // hard-coded: QFT|x>[rev(k)] = exp(2*pi*i*x*k/N)/sqrt(N).
  {
    const uint32_t n = 3, N = 8, x = 3;
    Reference r(n);
    r.apply(g1(G_X, 0));
    r.apply(g1(G_X, 1));  // |x=3>
    r.run(qft(n));
    auto rev = [&](uint32_t k) {
      uint32_t o = 0;
      for (uint32_t b = 0; b < n; b++) o |= ((k >> b) & 1u) << (n - 1 - b);
      return o;
    };
    double worst = 0.0;
    for (uint32_t k = 0; k < N; k++) {
      cdouble want = std::polar(1.0 / std::sqrt(double(N)),
                                2.0 * M_PI * double(x) * double(k) / double(N));
      worst = std::max(worst, std::abs(r.state()[rev(k)] - want));
    }
    check_near(worst, 0.0, 1e-12, "QFT|3> matches definition, n=3");
  }
}

static void test_unitarity() {
  printf("unitarity across the frozen set\n");
  for (const auto &spec : frozen_benchmarks()) {
    for (uint32_t n : {4u, 7u, 12u}) {
      Circuit c = spec.build(n);
      Reference r(n);
      r.run(c);
      check_near(r.norm(), 1.0, 1e-11,
                 spec.name + " n=" + std::to_string(n) + " norm");
      double psum = 0.0;
      for (double p : r.probabilities()) psum += p;
      check_near(psum, 1.0, 1e-11,
                 spec.name + " n=" + std::to_string(n) + " probabilities");
    }
  }
}

static void test_fails_closed() {
  printf("fail-closed behaviour\n");
  bool threw = false;
  try {
    Reference r(2);
    r.apply(g2(G_CX, 0, 0));  // same wire twice
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "CX on identical wires raises");

  threw = false;
  try {
    Reference r(2);
    r.apply(g1(G_X, 5));  // out of range
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "out-of-range qubit raises");
}

static void report_sizes() {
  printf("\nfrozen benchmark sizes\n");
  printf("%-18s %6s %8s %8s %8s\n", "circuit", "n", "gates", "1q", "2q");
  for (const auto &spec : frozen_benchmarks()) {
    for (uint32_t n : {24u, 28u}) {
      Circuit c = spec.build(n);
      printf("%-18s %6u %8zu %8zu %8zu\n", spec.name.c_str(), n, c.size(),
             c.count_1q(), c.count_2q());
    }
  }
}

int main() {
  test_bit_convention();
  test_single_qubit_identities();
  test_two_qubit();
  test_ghz();
  test_qft();
  test_unitarity();
  test_fails_closed();

  printf("\n%d checks, %d failures\n", checks, failures);
  if (failures == 0) report_sizes();
  return failures == 0 ? 0 : 1;
}
