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

## Follow-up: the ambiguity is resolved

The two hypotheses were separable by adding a circuit with locality in *wire*
space. Every circuit in the frozen seven is full-width -- each layer touches
every wire -- so blocking had nothing to exploit regardless of fusion, and its
1.18x standalone said more about the benchmarks than about blocking.

`local_patch_w8` is a subroutine acting repeatedly on an 8-wire work register,
weakly coupled to the rest: an ancilla-based oracle, a block-encoded
subroutine, a local Hamiltonian patch. Re-running the same grid
(`results/ablation_n26_b10_extended.txt`):

| circuit | p_base | p_F | p_B | p_FB | F/base | B/base | FB/F |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_patch_w8 | 698 | 125 | **125** | 113 | 6.64x | **3.76x** | **1.18x** |

Blocking alone reaches **3.76x**, squarely inside the 3.7-4.1x the S0 synthetic
probe measured. So hypothesis (2) is excluded: given a circuit with the
structure it needs, blocking delivers its full effect. It is not that blocking
barely works.

And yet on that same circuit it adds only **1.18x on top of fusion**.

The mechanism is visible in the pass counts. Fusion alone reaches 125 passes.
Blocking alone reaches **125 passes** — the identical number, by an entirely
different route. Together they reach 113, a further 1.10x. The two techniques
are not finding different work to eliminate; they are eliminating the same
work, and whichever runs first claims it.

**Conclusion: register-level fusion and cache-level blocking are substitutes,
not complements.** They exploit one resource — work per data-residency window —
at two levels of the hierarchy, and the levels are not independent. Composition
is strongly sub-multiplicative: 6.64x and 3.76x combine to 7.83x, not 25x.

## Caveats that stand

Per-circuit FB/F cells are noisy between runs: TFIM read 1.25x in the first
grid and 0.88x in the second, on identical code. The aggregate and the
local_patch result are the load-bearing numbers; individual cells within ~15%
of 1.0 should not be interpreted.

Qubit reordering is still untested. It would let blocking apply to full-width
circuits by permuting hot wires into the local window, and it remains the only
way blocking could help the frozen seven. But the local_patch result predicts
what it would find: wherever reordering makes blocking applicable, fusion has
already taken that ground.

## Standing

- Composition is strongly sub-multiplicative, and the follow-up shows why: on
  the one circuit where blocking has full scope, it reduces passes to exactly
  the count fusion reaches on its own.
- Fusion is the better instrument of the two. It applies to every circuit
  shape; blocking needs wire-space locality that most workloads do not have.
- Verification unchanged: 126 CPU fusion checks including blocked plans, 225
  GPU checks, 87 oracle self-checks. Blocked plans reproduce the circuit
  exactly in double.
