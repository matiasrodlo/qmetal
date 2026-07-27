# Pre-registration — does cache blocking compose with fusion?

Written before the blocked executor was implemented or measured. Committed
first so the prediction cannot be adjusted to fit the result.

## The question

Register-level fusion, cache-level blocking, and buffer-level in-place
execution are the same optimization — maximize work per data-residency window —
at three levels of the memory hierarchy. Do they compose, or do they compete
for the same headroom?

## What is already measured

**S0, blocking against an unfused baseline.** Staging a contiguous block in
threadgroup memory and applying K gates before writing back gave **3.7–4.1× at
K=32**, saturating far below the ideal K. Best block was 8 KB, not the 32 KB
maximum: occupancy beats capacity. Below K=2 blocking was a net loss (0.60×).

**S3, fusion against the same unfused baseline.** Geometric mean **3.67×**,
with structured circuits at 5.2–8.4×. Fused kernels run at 240–341 GB/s, still
inside the sustained bandwidth band.

**S3, the warning sign.** Where speedup and pass count diverge, they diverge in
one direction: QAOA and TFIM cut passes ~12× but gained only 5–6×. A diagonal
layer carrying ~75 terms has stopped being memory-bound. That is the same
saturation the S0 probe showed, reached from the opposite direction.

## Prediction

**Blocking will add far less on top of fusion than the 3.7–4.1× it showed
against an unfused baseline. Point estimate 1.2×, with 1.0–1.6× the interval I
would not be surprised by.**

The reasoning: both techniques buy the same thing, which is work per residency
window. Fusion has already claimed most of the cross-gate reuse by collapsing
runs into single passes; there is far less left for blocking to recover. The
S3 saturation says the fused kernels are already drifting away from being
bandwidth-limited, and blocking only helps a workload that is.

Specifically:

- **Circuits fusion already handles well** (qft, qaoa, tfim, grover — all
  ≥5×) will gain least, ≤1.15×. Their remaining passes are few and each is
  doing substantial per-amplitude arithmetic.
- **Circuits fusion handles badly** (random_d10 at 1.88×, ghz at 1.00×) will
  gain most, because their passes are still cheap and numerous — exactly the
  regime the S0 probe measured.
- **Against the unfused baseline**, blocking should roughly reproduce the S0
  result, 3–4×.

If that inversion holds — blocking helping most precisely where fusion helps
least — it is strong evidence the two are substitutes rather than complements,
and the composition is sub-multiplicative.

## What would falsify it

Blocking adding ≥2× on top of full fusion on any structured circuit. That would
mean the two exploit genuinely different locality and the hierarchy composes,
which is the more interesting outcome and the one this project was started to
look for.

## What is not being claimed

This first measurement uses a **static** local/global split with no qubit
reordering: only runs of operations already confined to low-index wires can be
blocked. It is therefore a **lower bound** on what blocking can deliver. A null
result here does not close the question; a positive one settles it early.
