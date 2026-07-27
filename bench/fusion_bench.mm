// fusion_bench.mm — does the dispatch reduction convert into wall time?
//
// Fusion cuts Plan::size(). That only matters if this workload is as
// memory-bound as claimed, so the point of this tool is to check that the
// speedup tracks the pass reduction rather than merely existing.
//
// Both arms run in the same process, alternating, so the two share a thermal
// state. Absolute numbers here are not comparable across sessions; the ratio is.

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

static double time_plan(uint32_t n, const Plan &p, int reps) {
  double best = 1e18;
  Simulator s(n);
  for (int r = 0; r < reps + 1; r++) {
    s.reset();
    s.synchronize();
    double t0 = now_s();
    s.run(p);
    s.synchronize();
    double dt = now_s() - t0;
    if (r > 0) best = std::min(best, dt);  // discard warmup
  }
  return best;
}

int main(int argc, char **argv) {
  uint32_t n = (argc > 1) ? uint32_t(atoi(argv[1])) : 26;
  int reps = (argc > 2) ? atoi(argv[2]) : 2;
  if (!Simulator::available()) { printf("no Metal device\n"); return 1; }

  printf("device : %s\n", Simulator::device_name().c_str());
  printf("n      : %u (%.2f GiB)\n\n", n,
         double(uint64_t(1) << n) * 8.0 / 1073741824.0);

  printf("%-18s %7s %8s %8s %9s %9s %8s %8s\n", "circuit", "gates", "passes0",
         "passes1", "unfused", "fused", "speedup", "vs.pass");
  double geo = 0.0;
  int count = 0;

  for (const auto &spec : frozen_benchmarks()) {
    Circuit c = spec.build(n);
    Plan p0 = build_plan(c, FusionOptions::none());
    Plan p1 = build_plan(c, FusionOptions{});

    // Alternate the arms so a thermal drift during the run cannot favour one.
    double t0 = time_plan(n, p0, reps);
    double t1 = time_plan(n, p1, reps);
    t0 = std::min(t0, time_plan(n, p0, reps));
    t1 = std::min(t1, time_plan(n, p1, reps));

    double speedup = t0 / t1;
    double pass_ratio = double(p0.size()) / double(p1.size());
    printf("%-18s %7zu %8zu %8zu %8.3fs %8.3fs %7.2fx %7.2fx\n",
           spec.name.c_str(), c.size(), p0.size(), p1.size(), t0, t1, speedup,
           pass_ratio);
    fflush(stdout);
    geo += std::log(speedup);
    count++;
  }
  printf("\ngeometric mean speedup: %.2fx\n", std::exp(geo / count));
  return 0;
}
