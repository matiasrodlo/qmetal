// ablation.mm — the composition experiment.
//
// Four conditions, one codebase, one flag moved at a time:
//
//                  no fusion        fusion
//   no blocking    baseline         F
//   blocking       B                FB
//
// The quantity of interest is FB/F -- what blocking adds on top of fusion --
// measured against B/baseline, what it adds on its own. If those are similar the
// two exploit different locality and the hierarchy composes. If FB/F is far
// smaller, they are substitutes competing for the same headroom.
//
// The prediction was registered in results/PREREGISTRATION.md before this file
// existed: FB/F ~ 1.2x, interval 1.0-1.6x, falsified by >= 2x on any structured
// circuit.
//
// All four arms run in one process, interleaved, so they share a thermal state.
// Absolutes are not comparable across sessions; the ratios are.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "qmetal/circuits.h"
#include "qmetal/plan.h"
#include "qmetal/simulator.h"

using namespace qmetal;

static double now_s() {
  return double(clock_gettime_nsec_np(CLOCK_MONOTONIC)) * 1e-9;
}

static double time_plan(Simulator &s, const Plan &p, int reps) {
  double best = 1e18;
  for (int r = 0; r < reps + 1; r++) {
    s.reset();
    s.synchronize();
    double t0 = now_s();
    s.run(p);
    s.synchronize();
    double dt = now_s() - t0;
    if (r > 0) best = std::min(best, dt);
  }
  return best;
}

int main(int argc, char **argv) {
  uint32_t n = (argc > 1) ? uint32_t(atoi(argv[1])) : 26;
  uint32_t bits = (argc > 2) ? uint32_t(atoi(argv[2])) : 10;  // S0 optimum
  int reps = (argc > 3) ? atoi(argv[3]) : 2;
  if (!Simulator::available()) { printf("no Metal device\n"); return 1; }

  printf("device      : %s\n", Simulator::device_name().c_str());
  printf("n           : %u (%.2f GiB)\n", n,
         double(uint64_t(1) << n) * 8.0 / 1073741824.0);
  printf("block       : 2^%u amplitudes (%u KiB threadgroup)\n\n", bits,
         (1u << bits) * 8u / 1024u);

  FusionOptions none = FusionOptions::none();
  FusionOptions fuse;                              // all passes, no blocking
  FusionOptions block_only = FusionOptions::none();
  block_only.block_bits = bits;
  FusionOptions both = FusionOptions::blocked(bits);

  printf("%-18s %7s %7s %7s %7s | %8s %8s %8s | %7s %7s %7s\n", "circuit",
         "p_base", "p_F", "p_B", "p_FB", "base", "F", "FB", "F/base", "B/base",
         "FB/F");

  double g_f = 0, g_b = 0, g_fb = 0;
  int count = 0;

  for (const auto &spec : extended_benchmarks()) {
    Circuit c = spec.build(n);
    Plan p_base = build_plan(c, none);
    Plan p_f = build_plan(c, fuse);
    Plan p_b = build_plan(c, block_only);
    Plan p_fb = build_plan(c, both);

    Simulator s(n);
    // Interleave the arms so drift during the run cannot favour one of them.
    double t_base = time_plan(s, p_base, reps);
    double t_f = time_plan(s, p_f, reps);
    double t_b = time_plan(s, p_b, reps);
    double t_fb = time_plan(s, p_fb, reps);
    t_base = std::min(t_base, time_plan(s, p_base, reps));
    t_f = std::min(t_f, time_plan(s, p_f, reps));
    t_b = std::min(t_b, time_plan(s, p_b, reps));
    t_fb = std::min(t_fb, time_plan(s, p_fb, reps));

    printf("%-18s %7zu %7zu %7zu %7zu | %7.3fs %7.3fs %7.3fs | %6.2fx %6.2fx "
           "%6.2fx\n",
           spec.name.c_str(), p_base.size(), p_f.size(), p_b.size(),
           p_fb.size(), t_base, t_f, t_fb, t_base / t_f, t_base / t_b,
           t_f / t_fb);
    fflush(stdout);

    g_f += std::log(t_base / t_f);
    g_b += std::log(t_base / t_b);
    g_fb += std::log(t_f / t_fb);
    count++;
  }

  printf("\ngeometric mean   fusion alone %.2fx   blocking alone %.2fx   "
         "blocking on top of fusion %.2fx\n",
         std::exp(g_f / count), std::exp(g_b / count), std::exp(g_fb / count));
  return 0;
}
