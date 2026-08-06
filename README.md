# Quantum Metal Simulator

Quantum circuit simulator designed specifically for
Apple Silicon.

It produces exact amplitudes, ground truth for validating approximate
simulators, noise models, and hardware output, for developing algorithms up to
33 qubits.

## Numbers

M4 Max (128 GB), `complex64`, full circuits end to end (`bench/scale`):

| Qubits | State | GHZ, 33 gates | QFT, 561 gates |
|---:|---:|---:|---:|
| 30 | 8 GiB | 0.81 s | 6.5 s |
| 31 | 16 GiB | 1.65 s | 13.6 s |
| 32 | 32 GiB | 3.41 s | 28.1 s |
| 33 | 64 GiB | **7.67 s** | **58.1 s** |

Both verified at every size without an oracle — at 33 qubits a `complex128`
reference would need 137 GB. GHZ is checked by norm, `<Z_q> = 0`, `<Z_0 Z_n> = 1`
and sampling that yields only the two poles; QFT by norm and `<Z_q> = 0` on the
uniform image. None of those hold if a gate, a stride, or a dispatch is wrong.

Throughput swings **250–485 GB/s** with machine state — same binary, an hour
apart. The high figure is a cold burst at ~89% of peak; under sustained load it
settles near half that. Absolute numbers are only meaningful with their thermal
conditions attached. Ratios measured within a single run are stable and are
what this project reports.

33 qubits is the ceiling: 34 needs 128 GiB in one allocation, past Metal's
`maxBufferLength` of 80.64 GiB. It is also contingent on free system memory —
the 64 GiB allocation succeeds on an idle machine and returns OOM at 30% free,
well inside the 107 GiB working set the device advertises.

In-place execution is worth one qubit and ~14% throughput over a two-buffer
scheme. Data content is not a factor: all-zero and dense states time
identically, so the numbers are not a zero-compression artifact.

Cross-gate blocking — staging a contiguous block in threadgroup memory and
applying K gates to it before writing back — gives **3.7–4.1× at K=32**,
saturating far below the ideal K. The best block is 8 KB, not the 32 KB
maximum: occupancy beats capacity.

## Fusion

Gates are lowered into a `Plan` by a pipeline of passes, each independently
ablatable. Measured at n=26, both arms alternating in one process:

| circuit | passes | fused | speedup |
|---|---:|---:|---:|
| qft | 351 | 51 | **8.42x** |
| qaoa_ring_p6 | 650 | 55 | **6.06x** |
| grover_proxy_4 | 450 | 71 | **5.84x** |
| tfim_trotter_20 | 2020 | 160 | **5.16x** |
| hea_d3 | 231 | 96 | 3.09x |
| random_d10 | 385 | 209 | 1.88x |
| ghz | 26 | 26 | 1.00x |

Geometric mean 3.67x. Fused kernels run at 240-341 GB/s, inside the sustained
band, so they remain bandwidth-limited rather than trading memory for
arithmetic. Random circuits are deliberately hostile to fusion and GHZ has
nothing to fuse; both are in the frozen set precisely so the mean cannot be
flattered by picking friendly workloads.

Four passes. Runs of commuting diagonal gates collapse into one pass whose
phase is a sum of masked angles, so a QFT stage's whole controlled-phase ladder
becomes a single op. `CX . RZ . CX` is recognised as the diagonal ZZ coupling
it is, which is what makes Trotter and QAOA layers fusable at all. Windows of
single-qubit gates collapse per wire into one 2x2 product, tested with `rtol=0`
so a real RZ(1e-6) survives while H*H cancels away. What remains -- a layer of
gates on distinct wires, where there is nothing to multiply -- runs K at a time
with 2^K amplitudes held in registers, turning K full-state passes into one at
identical traffic.

Where speedup and pass count diverge, the gap is the finding rather than noise:
qaoa and tfim now cut passes 12x but gain 5-6x, because a layer carrying ~75
phase terms has stopped being bandwidth-limited. That is the same saturation
the cross-gate blocking probe showed, reached from the opposite direction.

Every pass is verified in double against the same circuit run gate by gate,
with no GPU involved.

## Gradients

Adjoint differentiation: every parameter in O(gates) rather than the
O(gates x params) that parameter-shift costs. One forward pass, a co-state
`|lambda> = H|psi>` accumulated one Hamiltonian term at a time, then a backward
sweep taking `Im<lambda|P|psi>` per rotation and undoing both states in step.

Verified against the parameter-shift rule computed on the double-precision
oracle -- which is an identity for `exp(-i t P/2)`, not a finite-difference
approximation. Agreement is ~1e-6 across HEA, TFIM and QAOA ansatze.

Needs a second state buffer, so gradients cap at 32 qubits rather than 33.
Only RX, RY and RZ are natively differentiable; a parameterised gate outside
that set raises rather than silently contributing zero.

## Python

A ctypes wrapper over the C ABI, so it needs no compiler, no pybind11 and no
nanobind -- just the shared library.

```python
import qmetal

c = qmetal.Circuit(4)
c.h(0)
for i in range(3):
    c.cx(i, i + 1)

sim = qmetal.Simulator(4)
sim.run(c)
print(sim.counts(1000))           # {'0000': ~500, '1111': ~500}
print(sim.expectation("ZZII"))    # 1.0
print(c.plan_size(fuse=True))     # GPU passes, not gate count
```

`sim.amplitudes()` returns a zero-copy numpy view of the unified-memory buffer.
The C ABI in `include/qmetal/qmetal.h` is plain C11 and is the surface other
languages should bind to; the C++ headers carry no stability promise.

## Conventions

- Index order LSB-first; qubit *q* has stride `1 << q`
- Amplitudes `complex64` (Apple GPUs have no fp64)
- Reductions and lookup tables accumulated in `double` on the host
- Gates applied in place

## Build

```bash
make test          # CPU oracle self-checks, then every GPU kernel against it

cd bench && make
./bw_probe 33      # bandwidth, stride sensitivity, residency
./reuse_probe 28   # cross-gate blocking, launch overhead
./zerocheck        # zero vs dense data control
```

Correctness is defined against a `complex128` CPU reference, itself validated
against closed forms rather than against another run. Amplitudes agree to
~1e-8 for shallow circuits; `complex64` drift then grows as roughly G^0.80 in
gate count (R²=0.97 over 45–4,500 gates on TFIM Trotter at n=12), reaching
1.2e-5 at 4,500 gates. It is a single regime — one power law, no knee.

An earlier version of this file reported a knee near depth 90, where the error
jumped five orders and saturated near O(0.3), and attributed it to chaotic
divergence between the fp32 and fp64 trajectories. That was wrong. It was a
bug in this simulator's parameter pool, and the tell was the norm: the state
stopped being a unit vector, which no amount of chaos in a unitary evolution
can cause. Two defects, both in `src/metal/simulator.mm`:

- The pool was recycled when the command encoder closed, not when the GPU had
  actually run the buffer. A non-blocking flush handed live parameter bytes
  back to the allocator while dispatches were still reading them.
- A pool wrap could land between the two parameter regions of a single op,
  stranding the first region above the reset while later ops were handed the
  same bytes.

Neither could fire until a single command buffer held enough dispatches to
fill the pool. Everything `test_gpu` actually *asserted* ran below that
threshold; the one case that crossed it was the drift table, which printed its
numbers without checking them — so the corruption was on screen the whole time
and read as a physics result. `test_gpu` now asserts those figures, checks the
norm alongside every amplitude comparison, extends the frozen benchmark sweep
to n=20 (the smallest width where the set fills the pool mid-buffer), and
sweeps `QMETAL_FLUSH_EVERY` to require bit-identical results regardless of how
work is grouped into command buffers.

Measurements are only comparable within one run. Let the machine idle before a
timing run, and never compare absolutes across sessions.

