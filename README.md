# Quantum Metal

Quantum circuit simulator designed specifically for
Apple Silicon.

It produces exact amplitudes, ground truth for validating approximate
simulators, noise models, and hardware output, for developing algorithms up to
33 qubits.

## Numbers

M4 Max (128 GB), `complex64`, one in-place single-qubit gate:

| Qubits | State | Cold | Sustained |
|---:|---:|---:|---:|
| 30 | 8 GiB | 35 ms | 63 ms |
| 32 | 32 GiB | 142 ms | — |
| 33 | 64 GiB | 290 ms | — |

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
cd bench && make
./bw_probe 33      # bandwidth, stride sensitivity, residency
./reuse_probe 28   # cross-gate blocking, launch overhead
./zerocheck        # zero vs dense data control
```

Measurements are only comparable within one run. Let the machine idle before a
timing run, and never compare absolutes across sessions.

