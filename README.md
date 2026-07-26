# corequantum

Quantum circuit simulator designed specifically for
Apple Silicon.

It produces exact amplitudes, ground truth for validating approximate
simulators, noise models, and hardware output, for developing algorithms up to
33 qubits.

## Numbers

M4 Max (128 GB), `complex64`:

| Qubits | State | One 1q gate |
|---:|---:|---:|
| 30 | 8 GiB | 35 ms |
| 32 | 32 GiB | 142 ms |
| 33 | 64 GiB | 290 ms |

~485 GB/s in place, ~89% of peak bandwidth. 33 qubits is the ceiling — 34 needs
128 GiB in one allocation, past Metal's `maxBufferLength` of 80.64 GiB.
In-place execution is worth one qubit and ~14% throughput over a two-buffer
scheme.

## Conventions

- Index order LSB-first; qubit *q* has stride `1 << q`
- Amplitudes `complex64` (Apple GPUs have no fp64)
- Reductions and lookup tables accumulated in `double` on the host
- Gates applied in place

## Build

```bash
cd bench && make && ./bw_probe 33
```

