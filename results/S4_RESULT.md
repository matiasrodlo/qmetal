# S4 result — does cache blocking compose with fusion?

Measured against `results/PREREGISTRATION.md`, committed as `aad1929` before the
blocked executor existed. Raw data: `results/ablation_n26_b10.txt`.

n = 26, block = 2^10 amplitudes (8 KiB threadgroup, the S0 optimum), all four
arms interleaved in one process.

| circuit | p_base | p_F | p_B | p_FB | F/base | B/base | **FB/F** |
|---|---:|---:|---:|---:|---:|---:|---:|
| ghz | 26 | 26 | 26 | 26 | 1.00x | 0.99x | 0.99x |
| qft | 351 | 51 | 351 | 51 | 8.43x | 1.00x | 1.00x |
| qaoa_ring_p6 | 650 | 55 | 587 | 48 | 6.10x | 1.14x | 1.03x |
| tfim_trotter_20 | 2020 | 160 | 1840 | 140 | 5.49x | 1.12x | **1.25x** |
| grover_proxy_4 | 450 | 71 | 297 | 62 | 5.68x | 1.50x | 1.13x |
| random_d10 | 385 | 209 | 295 | 202 | 1.88x | 1.36x | 0.98x |
| hea_d3 | 231 | 96 | 174 | 93 | 3.04x | 1.24x | 0.99x |

**Geometric mean: fusion alone 3.68x, blocking alone 1.18x, blocking on top of
fusion 1.05x.**

## Verdict on the prediction

**Headline claim: correct.** Blocking adds 1.05x on top of fusion. Predicted
1.2x with an interval of 1.0-1.6x, so the result lands at the bottom edge of
the stated interval. The falsifier — 2x or more on any structured circuit —
did not fire; the maximum observed is 1.25x on TFIM.

**Mechanism claim: wrong.** The prediction said blocking would help most where
fusion helps least, and that the inversion would be the evidence for
substitution. The opposite happened. Blocking adds most on TFIM (1.25x), which
fusion already accelerates 5.49x, and adds nothing on random_d10 (0.98x), which
fusion handles worst at 1.88x.

The reasoning was wrong because blocking's applicability is governed by circuit
*structure*, not by how much headroom fusion left. Blocking needs runs of
single-qubit work confined to low wires. TFIM has long RX layers that group into
exactly that; random circuits have a brickwork CX pattern that breaks every run;
GHZ has almost no single-qubit gates at all. None of that correlates with how
fusable a circuit is.

Several FB/F values sit slightly below 1.0. That is consistent with the S0
finding that staging is a net loss on short runs — the same cost appearing at
circuit scale.

## What this does not settle

Blocking alone reaches only **1.18x** here, against the **3.7-4.1x** the S0
probe measured on synthetic runs. That gap is the real story of this
experiment, and it means the measurement cannot yet distinguish two hypotheses:

1. Blocking and fusion are substitutes competing for the same locality, so
   little is left once fusion has run; or
2. Blocking barely applies to these circuits under a **static** local/global
   split, and the composition question is untested rather than answered.

The pre-registration anticipated this and called the measurement a lower bound.
The S0 probe fed the blocked kernel a best case — K gates all on wires below b,
back to back. Real circuits almost never present that, and with no qubit
reordering a run leaves the local window as soon as it touches a high wire.

Distinguishing (1) from (2) requires qubit reordering: permuting the state so
hot wires become low-index, paying one pass per reorder and amortising it over
the run. That is the next experiment, and until it is done the honest statement
is that blocking on top of fusion is worth ~1.05x **with a static split**, not
that the hierarchy fails to compose.

## Standing

- Composition is sub-multiplicative under a static split: 3.68x and 1.18x
  combine to 3.87x, not 4.34x.
- The 1.25x on TFIM is the only cell where blocking clearly pays, and it is
  where the circuit offers long low-wire runs.
- Verification unchanged: 126 CPU fusion checks including blocked plans, 225
  GPU checks, 87 oracle self-checks. Blocked plans reproduce the circuit
  exactly in double.
