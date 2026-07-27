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
~1e-8 for shallow circuits; `complex64` drift grows roughly as G^0.76 in gate
count, reaching 1.7e-5 at 4,500 gates.

Measurements are only comparable within one run. Let the machine idle before a
timing run, and never compare absolutes across sessions.

