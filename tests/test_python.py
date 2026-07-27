"""End-to-end check of the C ABI through the Python bindings.

Deliberately independent of the C++ suites: this exercises the boundary itself
-- handle lifetimes, array marshalling, error propagation -- against closed
forms rather than against another qmetal run.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import qmetal  # noqa: E402

fails = 0
checks = 0


def check(ok, what):
    global fails, checks
    checks += 1
    if not ok:
        fails += 1
        print(f"  FAIL  {what}")


def near(got, want, tol, what):
    check(abs(got - want) <= tol, f"{what}: got {got!r} want {want!r}")


print(f"qmetal {'.'.join(map(str, qmetal.version()))} on {qmetal.device_name()}")
check(qmetal.is_available(), "Metal device available")
check(qmetal.max_qubits() == 33, "ceiling reported as 33 qubits")

# GHZ: two poles, evenly weighted, nothing else.
n = 6
c = qmetal.Circuit(n)
c.h(0)
for i in range(n - 1):
    c.cx(i, i + 1)
check(len(c) == n, "circuit length")

sim = qmetal.Simulator(n)
sim.run(c)
near(sim.norm(), 1.0, 1e-5, "GHZ norm")
near(sim.expectation("Z" + "I" * (n - 1)), 0.0, 1e-5, "<Z0> = 0")
near(sim.expectation("ZZ" + "I" * (n - 2)), 1.0, 1e-5, "<Z0 Z1> = 1")

counts = sim.counts(4000, seed=7)
check(set(counts) == {"0" * n, "1" * n}, f"only the poles appear: {list(counts)}")
check(abs(counts["0" * n] / 4000 - 0.5) < 0.03, "poles are balanced")

amps = sim.amplitudes()
near(abs(amps[0]), 1 / math.sqrt(2), 1e-5, "amplitude |0..0>")
near(abs(amps[(1 << n) - 1]), 1 / math.sqrt(2), 1e-5, "amplitude |1..1>")

# Fusion is visible through the ABI.
# At small n a QFT stage still costs an H plus a ladder, so the ratio is near
# 2; the win grows with the ladder length. Measured at a size where it matters.
qc = qmetal.Circuit(16)
for j in range(15, -1, -1):
    qc.h(j)
    for k in range(j - 1, -1, -1):
        qc.cp(k, j, math.pi / (1 << (j - k)))
check(qc.plan_size(fuse=False) == len(qc), "unfused plan is one pass per gate")
ratio = len(qc) / qc.plan_size(fuse=True)
print(f"  QFT n=16: {len(qc)} gates -> {qc.plan_size(fuse=True)} passes "
      f"({ratio:.2f}x)")
check(ratio >= 3.0, f"QFT fuses at least 3x (got {ratio:.2f}x)")

# Gradient against the closed form dE/dt = -sin(t) for RY under H = Z.
for t in (0.0, 0.4, 1.3, -0.7):
    g = qmetal.Circuit(2)
    g.ry(0, t)
    h = qmetal.Hamiltonian().add_string(1.0, "ZI")
    e, grad = qmetal.Simulator(2).energy_and_gradient(g, h)
    near(e, math.cos(t), 1e-5, f"E(t={t}) = cos t")
    near(grad[0], -math.sin(t), 1e-5, f"dE/dt(t={t}) = -sin t")

# Errors cross the boundary as exceptions, not crashes or silent zeros.
try:
    qmetal.Circuit(2).x(5)
    check(False, "out-of-range qubit raises")
except qmetal.QMetalError:
    check(True, "out-of-range qubit raises")

try:
    qmetal.Simulator(64)
    check(False, "oversized simulator raises")
except qmetal.QMetalError:
    check(True, "oversized simulator raises")

try:
    bad = qmetal.Circuit(2)
    bad.cp(0, 1, 0.3)
    qmetal.Simulator(2).energy_and_gradient(
        bad, qmetal.Hamiltonian().add_string(1.0, "ZI"))
    check(False, "non-differentiable parameterised gate raises")
except qmetal.QMetalError:
    check(True, "non-differentiable parameterised gate raises")

print(f"\n{checks} checks, {fails} failures")
sys.exit(1 if fails else 0)
