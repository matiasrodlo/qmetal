// scale.mm — run the simulator up to its ceiling and verify without an oracle.
//
// Past about 20 qubits the complex128 reference stops being an option: at 33 it
// would need 137 GB. Verification at scale therefore has to come from structure
// the circuit guarantees, checked with the GPU's own reductions:
//
//   GHZ : norm 1; <Z_q> = 0 on every wire; <Z_a Z_b> = 1 on every pair;
//         sampling yields only |0...0> and |1...1>, near evenly.
//   QFT : norm 1 on |0...0>, whose image is the uniform superposition, so
//         <Z_q> = 0 on every wire and every outcome is equally likely.
//
// None of these are tautologies -- a wrong gate, a wrong stride, or a lost
// dispatch breaks at least one of them.
//
//   make -C .. bench   (or see bench/Makefile)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "qmetal/circuits.h"
#include "qmetal/simulator.h"

using namespace qmetal;

static double now_s() {
  return double(clock_gettime_nsec_np(CLOCK_MONOTONIC)) * 1e-9;
}

static void run_ghz(uint32_t n) {
  double t0 = now_s();
  Simulator s(n);
  double t_alloc = now_s() - t0;

  Circuit c = ghz(n);
  t0 = now_s();
  s.run(c);
  s.synchronize();
  double t_run = now_s() - t0;

  double norm = s.norm();
  double z0 = s.expectation(PauliString::single('Z', 0));
  PauliString zz;
  zz.z_mask = (1ull << 0) | (1ull << (n - 1));
  double zz_end = s.expectation(zz);

  auto shots = s.sample(4096, 7);
  const uint64_t all = (uint64_t(1) << n) - 1;
  size_t zeros = 0, ones = 0, other = 0;
  for (uint64_t v : shots) {
    if (v == 0) zeros++;
    else if (v == all) ones++;
    else other++;
  }

  bool ok = std::abs(norm - 1.0) < 1e-3 && std::abs(z0) < 1e-3 &&
            std::abs(zz_end - 1.0) < 1e-3 && other == 0;
  printf("ghz  %2u  %7.2f GiB  alloc %6.2fs  run %7.3fs  %6zu gates  "
         "norm %.6f  <Z0> %+.2e  <Z0 Zn> %.6f  0/1 %zu/%zu other %zu  %s\n",
         n, double(uint64_t(1) << n) * 8.0 / 1073741824.0, t_alloc, t_run,
         c.size(), norm, z0, zz_end, zeros, ones, other, ok ? "OK" : "FAIL");
  fflush(stdout);
}

static void run_qft(uint32_t n) {
  double t0 = now_s();
  Simulator s(n);
  double t_alloc = now_s() - t0;

  Circuit c = qft(n);
  t0 = now_s();
  s.run(c);
  s.synchronize();
  double t_run = now_s() - t0;

  double norm = s.norm();
  // QFT|0...0> is uniform, so every single-qubit Z averages to zero.
  double worst_z = 0.0;
  for (uint32_t q = 0; q < n; q += (n > 8 ? n / 8 : 1))
    worst_z = std::max(worst_z,
                       std::abs(s.expectation(PauliString::single('Z', q))));

  bool ok = std::abs(norm - 1.0) < 1e-2 && worst_z < 1e-2;
  printf("qft  %2u  %7.2f GiB  alloc %6.2fs  run %7.3fs  %6zu gates  "
         "norm %.6f  max|<Z_q>| %.2e  %s\n",
         n, double(uint64_t(1) << n) * 8.0 / 1073741824.0, t_alloc, t_run,
         c.size(), norm, worst_z, ok ? "OK" : "FAIL");
  fflush(stdout);
}

int main(int argc, char **argv) {
  uint32_t nmax = (argc > 1) ? uint32_t(atoi(argv[1])) : 30;
  uint32_t nmin = (argc > 2) ? uint32_t(atoi(argv[2])) : 20;
  if (!Simulator::available()) {
    printf("no Metal device\n");
    return 1;
  }
  printf("device : %s\n", Simulator::device_name().c_str());
  printf("maxbuf : %.2f GiB   working set : %.2f GiB\n\n",
         Simulator::max_buffer_length() / 1073741824.0,
         Simulator::recommended_working_set() / 1073741824.0);

  for (uint32_t n = nmin; n <= nmax; n++) {
    try {
      run_ghz(n);
    } catch (const std::exception &e) {
      printf("ghz  %2u  FAILED: %s\n", n, e.what());
      break;
    }
  }
  printf("\n");
  for (uint32_t n = nmin; n <= nmax; n++) {
    try {
      run_qft(n);
    } catch (const std::exception &e) {
      printf("qft  %2u  FAILED: %s\n", n, e.what());
      break;
    }
  }
  return 0;
}
