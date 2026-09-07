# Simulation positive control: does per-clade phi recover known concordance?

**Answer: no.** At proper scale the estimator does not recover a ground truth it was
handed cleanly. This is an estimator problem, not a data problem.

## Design

`simulate_perclade.py` builds a tree of 4,096 cells, cuts it into 32 disjoint clades at
depth 6, and gives each clade a known lambda controlling how much its within-clade spatial
layout follows its own tree:

    x_i = clade_centroid + sqrt(lambda) * (tree deviation) + sqrt(1 - lambda) * noise

so the between-clade layout is untouched and only the within-clade phylogenetic signal is
graded. Ground truth is verified before anything else runs:

    corr(true lambda, measured within-clade concordance) = +0.807  (p = 2.4e-08)

| lambda | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| induced concordance | -0.002 | 0.037 | 0.098 | 0.159 | 0.208 |

Monotone, and the induced range (0-0.21) matches real eTRACER (0-0.27), so this is not an
artificially easy version of the problem. 1,066 spots, 27 of 32 clades with >= 20 spots.

## Result

| | Pearson | Spearman |
|---|---|---|
| preferred phi vs **true lambda** | **-0.098** (p = 0.59) | -0.105 (p = 0.57) |
| preferred phi vs **measured concordance** | **-0.034** (p = 0.85) | -0.097 (p = 0.60) |

mean preferred phi: lambda >= 0.5 -> 0.705, lambda < 0.5 -> 0.772 (slightly backwards).

Worse than null in detail: three of the seven lambda = 0 clades (45, 34, 36) -- the ones
with *no* within-clade spatial signal at all -- came out with preferred phi = 1.000, the
maximum within-clade weighting. Preferred phi sits high (0.4-1.0) for nearly every clade
regardless of lambda.

## This overturns an earlier positive reading

A first pass on a smaller simulation (16 clades, 191 spots, only 8 passing the spot filter)
gave preferred phi vs measured concordance = **+0.859 (p = 0.006)** and was reported as the
control passing. **That was a small-sample fluke.** With 32 clades and 5x the spots it
inverts to -0.034. The n=8 result should not be cited.

## What it means

- **Do not pool PEtracer to chase statistical power.** The real-data null (r ~ 0.38,
  p = 0.12) is not an underpowering of a working estimator -- the estimator fails on clean
  simulated data with known graded truth and adequate clade counts.
- **Claim (b) cannot go in the paper in any current form.** Neither per-clade alpha nor
  per-clade phi recovers per-clade spatial concordance.

## Why it may be failing -- both sweep designs are confounded

1. **Per-clade perturbation** (move one clade's phi, hold the rest): the `s_k` equalizer in
   `build_perclade_alpha_cost` rescales each clade's cost by a factor keyed on
   `alpha_ref_k`, so deviating from a uniform vector is itself penalised. 15 of 23 clades
   trivially preferred the baseline.
2. **Global sweep** (move all clades together, read each clade's curve): avoids that, but
   every clade's score curve now reflects the whole system's response rather than that
   clade's own preference.

So there is currently **no clean way to read a per-clade preference out of this
parameterisation**. A valid test needs the equalizer neutralised so single-clade
perturbation is meaningful -- that is a change to `build_perclade_alpha_cost`, not to the
experiment. Until then the per-clade interpretability claim is untestable, never mind unproven.

## Bugs found while building this control

| bug | severity | status |
|---|---|---|
| `sample_spatial_effects` indexes a 200x200 map assuming coords in [0,1]; >1 raises IndexError, **<0 silently wraps** via negative indexing | silent wrong values for ~half of all cells | fixed (normalise + clip + vectorise) |
| `simulate_data` does `df / df.max()`, normalising the tree distance matrix **column-wise** (asymmetric) | silent, wrong on unbalanced/non-ultrametric trees | fixed (global max) |
| `rescale_clade` scales the within-clade block including its diagonal, so lambda changes magnitude but not correlation | invalidates any graded-lambda design | worked around in the sim |
| `compute_phylogenetic_weight_matrix` was ~60% of runtime | 464-573x recoverable, numerically identical | fixed (`utils.cophenetic_distance_matrix`) |

Two of these produce wrong answers silently, which is the category that matters for a
published benchmark.

---

# Attempting the fix: `src/spotr/perclade_v2.py`

Added as a **separate module**; `spotr.spotr` is unmodified. `build_perclade_alpha_cost_v2`
takes the same arguments plus `equalize`:

- `"measured"` (default) -- scale each clade to a common **measured** median |C0| instead of
  one predicted from alpha, so `s_k` no longer depends on `alpha_k` or `phi_k`
- `"alpha"` -- reproduces the original exactly, for A/B
- `"none"` -- no per-clade rescaling

## A/B on simulated data with known lambda (n = 27 clades)

| equalizer | best == baseline | Pearson | p | phi (lambda>=0.5 / <0.5) |
|---|---|---|---|---|
| original | 15/27 | +0.012 | 0.95 | 0.656 / 0.659 |
| **v2 "measured"** | **17/27** | **+0.168** | 0.40 | 0.688 / 0.614 |

The fix is a genuine improvement on a genuine defect -- an alpha-dependent `s_k` gives a
deviating clade its own entropic temperature, which is wrong regardless -- and it moved the
correlation from +0.012 to +0.168 with the hi/lo separation going from nil to positive.

**But it did not remove the baseline artifact** (15/27 -> 17/27). The equalizer was
contributing, not causing.

## Second hypothesis, tested and eliminated

Suspected the readout: `prop = prop / prop.sum(1)` looked like it coupled every clade to
every other. Re-ran scoring on raw transported mass instead of the normalised share.

Result: **identical to three decimals on every statistic.** The reason is exact --
under balanced OT the column sums of gamma are pinned to `b_j`, so

    per-spot total mass: min 9.380861e-04, max 9.380865e-04, sd/mean 9.8e-08
    b_j = 1/n_spots    = 9.380863e-04

The row sums are *constant by construction*, so normalising divides by a constant and cannot
change a correlation. The two readouts are the same statistic.

## Conclusion: this is structural, not a bug

With the equalizer neutralised and the readout shown to be irrelevant, what remains is the
marginal constraint itself. In balanced OT every column of gamma must sum to `b_j`, so mass
given to one clade at a spot is mass taken from another. **Perturbing one clade's parameter
necessarily moves every other clade**, and the unperturbed uniform configuration wins by
default -- which is precisely the baseline spike.

So a per-clade preference cannot be read out of a balanced-OT coupling by perturbation. No
further patch to the cost function will change that.

### This is the same root cause as the composition problem

Earlier finding: balanced OT pins the output clone composition to the reference composition
exactly (corr = 1.000, verified to ~1e-7 at every eps). Same constraint, two symptoms:

| symptom | mechanism |
|---|---|
| output composition == reference composition | row sums of gamma pinned to `a_i` |
| per-clade parameters unreadable | column sums of gamma pinned to `b_j` |

**A semi-relaxed / unbalanced formulation addresses both.** Relaxing the source marginal
lets clone abundances be fitted rather than assumed, and lets a clade's total mass change
when its own parameter changes -- making single-clade perturbation local, and therefore
interpretable. That is now the single highest-value change to the method, supported by two
independent lines of evidence rather than one.

---

# Semi-relaxed OT: fixes the composition problem, does not rescue claim (b)

`src/spotr/semirelaxed.py` (new module; `spotr.spotr` untouched) relaxes the SOURCE marginal
with a KL penalty of weight `tau` and keeps the target exact:

    rho = tau / (tau + eps)
    u  <-  (a / (K v)) ** rho      # damped; rho = 1 recovers balanced exactly
    v  <-   b / (K.T u)            # exact -- a spot really is fully composed of cells

## Result 1 (eTRACER LE) -- the composition problem is solved

Mismatch imposed exactly through the source marginal, so `a`'s clone shares reproduce the
natural scRNA-seq composition (TVD 0.538 from the tissue).

| solver | TVD(out, truth) | corr(out, reference) | acc | ARI |
|---|---|---|---|---|
| balanced | **0.538** | **1.0000** | 0.138 | 0.010 |
| semi-relaxed tau=1 | 0.513 | 0.9953 | 0.154 | 0.015 |
| semi-relaxed tau=0.1 | 0.333 | 0.7689 | 0.267 | 0.078 |
| **semi-relaxed tau=0.01** | **0.078** | 0.4870 | 0.487 | 0.239 |
| semi-relaxed tau=0.001 | 0.088 | 0.4605 | **0.520** | **0.259** |

Balanced lands at TVD = 0.538 with corr = 1.0000 -- the output composition IS the reference
composition, confirming the diagnosis exactly. Semi-relaxed cuts TVD 7x and lifts ARI ~25x.
For comparison, hand-matching the composition under balanced OT gave ARI 0.348; semi-relaxed
reaches 0.259 **without access to the answer key**, which is the point.

Caveat: at tau = 0.001 composition fidelity degrades (TVD 0.078 -> 0.088) while accuracy
still rises, so tau trades composition against placement and needs principled selection --
choosing it by best ARI would be fitting to the answer key.

## Result 2 (simulation, known lambda) -- claim (b) is dead

| solver | best == baseline | Pearson | p | phi (lambda>=0.5 / <0.5) |
|---|---|---|---|---|
| balanced | 18/27 | +0.175 | 0.38 | 0.656 / 0.614 |
| semi-relaxed tau=0.1 | 17/27 | -0.084 | 0.68 | 0.625 / 0.659 |
| **semi-relaxed tau=0.01** | **10/27** | +0.074 | 0.71 | 0.547 / 0.455 |

**The predicted mechanism is confirmed**: relaxing the source marginal makes single-clade
perturbation more local, and the baseline artifact nearly halves (18/27 -> 10/27).

**But recovery is still null.** With the artifact substantially removed, preferred phi
correlates with known lambda at +0.074 (p = 0.71).

## Conclusion on claim (b)

Every identified confound has now been removed in turn:

| confound | action | result |
|---|---|---|
| alpha-dependent per-clade equalizer | fixed (`perclade_v2`) | artifact unchanged, r +0.012 -> +0.168 |
| per-spot normalisation in the readout | proved a no-op (column sums constant) | identical to 3 d.p. |
| balanced marginal forcing global coupling | relaxed (`semirelaxed`) | artifact 18/27 -> 10/27, r still +0.074 |

On simulated data with 32 clades, a verified graded ground truth (corr(lambda, induced
concordance) = +0.807) and every known confound removed, per-clade phi does not recover
per-clade spatial concordance. **The information is not in that parameter.** The claim
should be dropped from the paper rather than pursued further.

What survives, and is stronger than what was claimed: the mapping result (tree beats a
shuffled tree by 18.5 sd) and the composition result above.

---

# ROOT CAUSE: the simulated tree makes the GW term identically zero

Every per-clade conclusion drawn from simulation is void. The reason is not the estimator,
the equalizer, the readout or the marginals -- it is the tree.

`cas.sim.CompleteBinarySimulator` produces a perfectly balanced ultrametric tree, so every
leaf has an *identical* average distance to every other leaf. The GW linearisation then
reduces to an additive f(i) + g(j) structure, and `_dc_rect` double-centering annihilates
exactly that.

Measured:

| | simulation (complete binary) | eTRACER (real lineage tree) |
|---|---|---|
| tree row-means | min 0.90010, max 0.90010, **sd/mean 2e-07** | min 1.745, max 2.934, sd/mean 0.19 |
| \|L\| after double-centering | **1.0e-07** | 4.4e+02 |
| ratio \|L\|/\|F\| | **4.3e-06** | 3.3e+04 |

Nine orders of magnitude. **In simulation the GW term is numerically zero, so alpha cannot
do anything** -- which is exactly what the sweeps showed (ARI identical to 4 d.p. across
alpha = 0 ... 0.5).

This invalidates, as tests of any tree-related claim:

- "the tree does not help" (Q1) -- the term was zero
- "per-clade weighting has no headroom" (Q2 oracle) -- weighting a zero term
- every per-clade phi/alpha interpretability result run on simulated data

It does NOT affect the real-data results: eTRACER has an unbalanced tree, the term is large
(|L|/|F| = 3.3e4), and there the tree demonstrably helps (+0.037 ARI, 18.5 sd above a
shuffled control).

**It also affects the repo's existing simulation benchmark**, which uses
`CompleteBinarySimulator` throughout -- any conclusion there about the value of the tree
term inherits this degeneracy.

## Fix

Use an unbalanced tree simulator:

| simulator | row-mean sd/mean | GW term |
|---|---|---|
| `CompleteBinarySimulator` | 2.1e-16 (machine precision) | annihilated |
| `BirthDeathFitnessSimulator` | **4.3e-02** | usable |
| *eTRACER real tree* | *1.9e-01* | — |

`BirthDeathFitnessSimulator` lands within a factor of ~4 of the real tree, so it is a valid
testbed. The per-clade question -- does clade-specific weighting help, and is the fitted
weight interpretable -- remains **open and untested**, not refuted.

## Running list of simulator defects found

| # | defect | severity | status |
|---|---|---|---|
| 1 | `sample_spatial_effects` assumes coords in [0,1]; >1 raises, **<0 silently wraps** | silent wrong values | fixed |
| 2 | `df / df.max()` normalises tree distances column-wise (asymmetric) | silent, wrong on unbalanced trees | fixed |
| 3 | `rescale_clade` scales magnitude not correlation | invalidates graded-lambda designs | worked around |
| 4 | `compute_phylogenetic_weight_matrix` ~60% of runtime | 464-573x recoverable | fixed |
| 5 | `sample_combined_expression` accepts `alpha` and **never uses it** | 'tree'/'mix'/'external' identical; expression always fully tree-determined | fixed (`expression_v2`) |
| 6 | `CompleteBinarySimulator` -> balanced tree -> **GW term is zero** | every tree claim untestable | fix identified (birth-death) |

Defects 5 and 6 together made SpOTr's entire premise unreachable in simulation: expression
was always a perfect proxy for the tree, and the tree term was always zero.

---

## Coarse-to-fine testbed (2026-08-25)

### Why the previous crossed design could not work

The spec was "some clades have clear expression signal (no GW needed), others have clear
tree-space concordance but no expression signal (GW needed)". The second group is
**unidentifiable**, and this is structural, not a simulator bug.

GW is a second-order term: it compares `C_tree[i,k]` to `C_space[j,l]`, so it constrains
*relations*, not *positions*. With K spatially-compact clades it says "partition the cells
into K groups and send each to a compact region" — it does not say which clade goes to
which region. All K! assignments have essentially the same GW cost. Expression is the only
quantity measured on BOTH sides (cells and spots), so it is the only thing that can break
that symmetry. A clade with no expression signal therefore has chance-level expected
accuracy at every alpha, while alpha > 0 costs accuracy on the clades that do have
expression signal. Hence the monotonically decreasing ARI curve.

Corollary: **the tree is a regulariser, not an information source.** It can sharpen a
mapping expression already roughly determines; it cannot create localisation.

Second corollary, which matters for the project's own standard: the only other
symmetry-breaker available to GW is clade/region *size* matching, which is weak and
combinatorially fragile. A claim resting on it is initialisation-dependent by construction.
So "the claim must not depend on initialisation" and "clades with no expression signal" are
mutually incompatible requirements.

### The corrected design (`simulate_coarse_fine.py`)

Expression ambiguous rather than absent:

| | coarse markers | subclade markers | within-clade lambda | tree should be |
|---|---|---|---|---|
| expr-fine | yes | yes | 0 | unnecessary (phi_k low) |
| tree-fine | yes | no  | 1 | required (phi_k high) |

Both types identifiable at the region level, so the GW symmetry is broken for everyone.
Disjoint cell/spot populations. Verified on seed 0: expression→clade 1.000/0.999,
expression→subclade 0.800 vs 0.213 (chance 0.25), within-clade tree-vs-space r +0.617 vs
+0.010. GW term alive, |L|/|F| = 5.6e-01.

Metric had to change too. Clade-ARI is saturated by expression by construction, so a refiner
cannot show up in it. Primary metric is **subclade accuracy**; also expected transport
distance `E_j||xy_j - xy_i||`. NOT barycentre error `||E_j[xy_j] - xy_i||`, which by Jensen
rewards hedging and penalises exactly the commitment a refiner makes — it reverses the
verdict on its own.

### Results

**Q1 — does the tree help?** Yes on seed 0: subclade accuracy 0.3869 -> 0.4320 at
(alpha 0.7, phi 1.0), +11.7% relative. First simulation on this project where SpOTr's
premise reproduces at all.

**Across seeds it does not hold up** (`replicate_q1.csv`, seeds 2/4/6 died in
BirthDeathFitnessSimulator):

| seed | best (a, phi) | acc alpha=0 | best | gain | gain on tree-fine |
|---|---|---|---|---|---|
| 0 | 0.7, 1.0 | 0.3869 | 0.4320 | +0.0451 | +0.0027 |
| 1 | 0.0, 0.0 | 0.4022 | 0.4022 | +0.0000 | +0.0000 |
| 3 | 0.5, 0.5 | 0.3486 | 0.3620 | +0.0135 | +0.0294 |
| 5 | 0.0, 0.0 | 0.4373 | 0.4373 | +0.0000 | +0.0000 |
| 7 | 0.7, 1.0 | 0.4420 | 0.4669 | +0.0250 | -0.0016 |

mean gain +0.0167, zero in 2 of 5. Seeds 0 and 1 have near-identical design properties
(tree-fine r +0.617 vs +0.542, 6.1 vs 6.4 spots/subclade) yet give +0.045 and exactly 0.

**The mechanism does not check out.** Gain on the tree-fine clades — the ones the account
says the tree exists to rescue — averages **+0.0061** and is negative in one seed. Whatever
produces the overall gain, it is not "the tree resolves the clades expression cannot".

**Q2 — per-clade alpha:** no headroom. Oracle 0.1127 vs best global 0.1063 (E-dist).
alpha_k cannot express the difference: for cell i the GW sum runs over all other cells, only
~1/K of which are in i's own clade, so alpha_k is dominated by the cross-clade contribution,
which is identical for every clade. That is also why the tree helped expr-fine clades.

**Q4 — per-clade phi:** the signal is real but there is still no headroom.
Oracle 0.4300 > shuffled 0.3982 +- 0.0154 (z=+2.1) > inverted 0.3600 — clean monotone
ordering on every metric. But **uniform phi=1.0 beats the oracle** (0.4320). Reason:
`acc_sub_ef` is 0.6549 global vs 0.6590 oracle, a tie — turning the tree down where it is
uninformative buys nothing, because those are exactly the clades where expression already
resolves position. A noise GW term is roughly isotropic: it adds variance, not bias. So
over-weighting costs second-order while correct weighting pays first-order, and
uniform-maximum is very hard to beat.

**Q5/Q6 — learned phi:** does not recover the split, from any initialisation.

| loss | init 0.2 | init 0.5 | init 0.8 |
|---|---|---|---|
| clade-level (wrapper) | AUC 0.278 | 0.639 | 0.500 |
| clade-level (direct) | 0.611 | 0.611 | diverged (NaN) |
| subclade-level | 0.028 (r -0.726, p 0.008) | 0.528 | 0.472 |

Making the loss fine-grained was the obvious fix — `deconvolution_loss_ls` scores spot
composition at whatever granularity `clone_assignments` has, and clade-level is blind to
within-clade placement, which is what phi controls; `learn_alpha_gamma_perclade_alpha_data`
takes `clone_assignments` and `omega` separately so the loss can be fine while phi stays
per-clade. It did not help. Sign-unstable across inits, one fit diverged, and the only
significant result points the wrong way.

### Where this leaves the two claims

- **(a) the tree improves the mapping** — survives, but weakly: mean +0.017 subclade
  accuracy, zero in 2/5 seeds, and not attributable to the clades the mechanism names.
- **(b) per-clade weighting** — fails on all three of its parts: no performance headroom
  over a uniform maximum, not learnable from the loss, and initialisation-dependent.

### Files

`simulate_coarse_fine.py`, `q1_coarse_fine.py`, `q2_coarse_fine.py`, `q3_phi_coarse_fine.py`,
`q4_headroom.py`, `q5_learn_phi.py`, `q6_fineloss.py`, `replicate_q1.py`

---

## Variance-controlled testbed (2026-08-25, later)

### The motivation that finally worked

Expression does not respect the tree, so confusable clones get mapped into each other's
territory. Two consequences the earlier designs never contained:

**Pooling.** If each cell independently picks the right territory 55% of the time, 45% land
wrong. The GW term forces a clade to move together, so the clade makes ONE 55/45 decision
instead of n independent ones — a majority vote of 100 cells at 55% is right ~84% of the
time. That is an information gain, not refinement.

**Headroom for per-clade alpha.** Pooling is only correct for a spatially coherent clade. For
a dispersed clade it drags cells to one wrong place, so over-weighting now costs FIRST-order,
not second-order. That is what was missing in the coarse-to-fine design, where downweighting
the tree where it was useless bought nothing.

Also a correction to the earlier symmetry argument: "all K! assignments cost the same under
GW" holds only for EXCHANGEABLE clades. For a generic unbalanced tree with distinct
inter-clade distances the correct alignment is a strict optimum, and GW is rigid up to global
isometry. Expression only has to fix the global frame. The gaps are small and easily swamped
by entropic smoothing, which is why it behaves as if symmetric — but "cannot" was too strong.

### Why the first version of it did not replicate

Gains of +0.098 / +0.040 / −0.021 / +0.086 across seeds with identical NOMINAL parameters.
Design parameters were verified on seed 0 and assumed to hold. Three things were free:

* the tree — BirthDeathFitnessSimulator drew a new topology per seed (K came out 8–12, min
  clade size 18–108) and 8 of 16 seeds died of lineage extinction
* the layout — territories came from a 2D Brownian walk, so separation was a lottery
* pair separation — how far apart a confusable pair's territories land is the dominant term
  and was entirely uncontrolled

Also a harness bug: when the best global alpha was 0, the oracle search's first candidate
`(lo=0, hi=bv=0)` gave a CONSTANT per-clade vector, so shuffling and inverting it were no-ops
and produced four identical numbers. And the pair-resolution metric conditioned on the very
assignment it was scoring, so its denominator moved between settings.

### `simulate_controlled.py`

Causal order inverted: draw the layout, then fit the tree to it.

* territories from a Poisson-disk packing at unit nearest-neighbour spacing
* tree by NEIGHBOR-JOINING on the layout, not UPGMA. UPGMA gives an ultrametric, which
  saturates (everything across the root sits at one distance) and capped tree-vs-layout r at
  0.65 however clean the layout. An additive tree reaches 0.82–0.85 and lifted the row-mean
  sd/mean from 0.02 to 0.13–0.16, comparable to the real eTRACER tree (0.19)
* branch-length noise bisected to hit a TARGET tree fidelity
* expression confusability bisected until MEASURED within-pair discrimination hits target
* layout resampled until realized pair separation hits target
* exact K and clade sizes; every realized property printed and range-checked

Realized properties across seeds: tree-vs-layout r 0.820–0.854, pair separation 1.925–2.061,
within-pair discrimination 0.565–0.584, compact spread ~0.20. No lineage extinction.

Caveat: tree fidelity and pair separation are controlled at their ACHIEVABLE CEILING (an
additive tree tops out ~0.87 on planar points; a 12-point packing offers limited distances),
so variance is much reduced, not eliminated.

### Results (11 in-band seeds)

| | uncontrolled | controlled |
|---|---|---|
| oracle per-clade vs best global | 4/8, mean +0.016 | **11/11, mean +0.055, sd 0.021** |
| oracle vs no tree | 5/8, mean +0.034 | **11/11, mean +0.098** |
| global alpha vs alpha=0 | 4/8 | **9/11, mean +0.044** |
| oracle vs shuffled (z) | 7/8 | **11/11, +0.6 to +3.7** |
| inverted worst | not always | **11/11** |

**Mechanism confirmed** (`q12_mechanism.csv`):

| | best alpha |
|---|---|
| compact clades | **0.350** |
| global optimum | 0.182 |
| dispersed clades | **0.095** |

Compact accuracy climbs to alpha 0.35 (0.138 → 0.231) while dispersed peaks at 0.10 and falls
(0.322 → 0.198). The global optimum sits between them because it is a compromise; per-clade
alpha escapes it. Compact wants a higher alpha than global in 9/11 seeds.

### The remaining gap: alpha is not learnable

* `deconvolution_loss_ls` is MONOTONE increasing in alpha in 11/11 seeds — argmin pinned at 0,
  while accuracy peaks at 0.10–0.35. Agreement 1/11. This is why learned alpha fell from every
  init (0.2→0.04, 0.5→0.16, 0.8→0.39) with compact and dispersed indistinguishable
  (AUC 0.449, p=0.16).
* NOTE: corr(loss, accuracy) = −0.611 (p=0.0006) reports the OPPOSITE and is the wrong
  statistic — past the accuracy peak both fall together. Argmin agreement is the right one.
* Held-out genes do NOT fix it (`q13_heldout.csv`): the held-out loss is also monotone, argmin
  0 in 11/11. The failure is not overfitting. Any objective of the form "reconstruct spot
  expression from the coupling" is minimised by the coupling that maximises expression
  agreement, which is exactly what the linear OT term computes.
* Two-stage plug-in (`q14_plugin.py`): estimate per-clade dispersion from an alpha=0 map, set
  alpha_k from it. Fails — AUC 0.558, p=0.22. A compact clade CONFUSED by expression looks
  dispersed. Plug-in was worse than global in 9/11.
* Bimodality (confused = two tight blobs, dispersed = uniform) is worse still: AUC 0.366,
  significantly BELOW chance (p=0.011). Information is present with the opposite sign, but
  flipping it against the answer key is not an estimator.

### Where the two claims stand

* **(a) the tree improves the mapping** — supported: +0.044 with a global alpha (9/11),
  +0.098 with per-clade alpha (11/11).
* **(b) per-clade weighting** — the headroom is REAL and robust (11/11, mechanism verified),
  but there is currently NO way to realise it without the answer key. The blocker is the
  objective, not the parameterisation.

Open problem: find a per-clade estimator of spatial coherence that is not confounded by
expression confusion, or an objective that is not an expression-reconstruction loss.

### Files

`simulate_controlled.py`, `replicate_controlled.py`, `q10_loss_alignment.py`,
`q11_learn_ctrl.py`, `q12_mechanism.py`, `q13_heldout.py`, `q14_plugin.py`,
`plot_controlled.py`, plus the confusable precursors `simulate_confusable.py`,
`q7_confusable.py`, `replicate_confusable.py`.

---

## CNV anchor fixes the objective (2026-08-25, later still)

### The idea

The blocker was that the loss and the OT feature cost were the SAME modality. Any
"reconstruct spot expression" objective is minimised by the coupling that maximises
expression agreement, which is exactly what the linear OT term computes -- hence monotone in
alpha, argmin pinned at 0, held-out genes no help.

`deconvolution_loss_ls` was in fact WRITTEN for CNVs -- its docstring says "For normalized CNV
profiles" and it divides signatures by 2 "because the CNVs are the copy number of the
alleles". Feeding it expression signatures was a misuse that created the circularity.

General principle: the loss modality must be
  (a) measured on BOTH cells and spots,
  (b) a function of clone identity, so the tree predicts it,
  (c) NOT the modality driving the OT feature cost.

### Simulation

`simulate_controlled.py` gained a CNV channel: copy number as a Gaussian process over clones
with covariance exp(-kappa * D_clade), so CNV similarity decays with tree distance. Uses a
separate RNG so every other draw stays bit-identical. Design check: CNV separates confusable
pair members as well as any random pair (ratio ~1.0) while expression manages 0.57.

Also fixed a reproducibility bug introduced earlier: `tree_from_layout` seeded its RNG from
`hash(("tree", mid))`, and Python randomises str hashing per process (PYTHONHASHSEED), so the
simulator gave different output on every run. Now seeded from `int(mid * 1e9)`.

### Results (10 in-band seeds)

**Q15a -- global alpha selection.** The CNV loss escapes the monotone trap but under-selects:

| objective | picks alpha > 0 | mean abs(selected - optimal) |
|---|---|---|
| expression | 0/10 | 0.210 |
| CNV | 5/10 | 0.150 |

**Q15b/c -- per-clade alpha LEARNED against the CNV loss.** This is where it pays:

| init | mean AUC | above chance | alpha compact | alpha disp | vs best global | vs no tree |
|---|---|---|---|---|---|---|
| 0.2 | 0.964 | 10/10 | 0.152 | 0.024 | +0.0401 | +0.0861 |
| 0.5 | 0.828 | 9/10 | 0.241 | 0.140 | +0.0091 | +0.0551 |
| 0.8 | 0.475 | 4/10 | 0.553 | 0.568 | -0.0366 | +0.0095 |

Overall AUC 0.756, t=+5.36, p<0.0001.

**Q17 -- initialisation.** A plateau, not a cliff:

| init | 0.02 | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 0.65 | 0.80 |
|---|---|---|---|---|---|---|---|---|
| AUC | 0.964 | 0.961 | 0.975 | 0.964 | 0.967 | 0.828 | 0.658 | 0.475 |
| >0.5 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 8/10 | 4/10 |

init <= 0.35 gives AUC 0.96-0.98 in 40/40 fits. Failure above 0.5 is a GENUINE BAD BASIN, not
non-convergence: 1200 iterations gives the same AUC as 300 (0.567 both). Since alpha = 0 is
the no-tree null, initialising low is the principled default and the rule is stateable.

**Q16 -- anchor quality.** Robust. At 4x noise and 8 retained segments the CNV loss still
selects alpha > 0 about as often as with a clean 150-bin anchor, and selection error stays
~0.15-0.17. Practically: noisy Numbat/CalicoST calls should be fine. It also shows the
mediocre global-alpha selection (50%) is not limited by anchor quality.

### IMPORTANT CAVEAT for real data

inferCNV and (partly) Numbat infer copy number FROM EXPRESSION. If the CNV calls are derived
from the same expression matrix that builds the OT feature cost, the circularity partially
returns and this fix is weakened. Use allele-specific calling driven by SNP/BAF information
(Numbat's allele model, CalicoST) or DNA-based copy number, so the anchor is genuinely a
different measurement.

### When there is no CNV anchor

Ranked by how well they satisfy (a)-(c):

1. **In-situ lineage barcodes** (eTRACER, PEtracer). Strongest: condition (b) holds exactly
   rather than approximately, and both datasets are already in the paper plan.
2. **CNVs / SNVs / LOH** in cancer -- validated here, subject to the caveat above.
3. **Mitochondrial variants** -- clonal, and independent of nuclear expression.
4. **A functional expression split** -- cell-state genes for the cost, clone-specific genes
   for the loss. Note the RANDOM gene split failed (Q13) because both halves carried the same
   mixed signal; a functional split is untested and partly circular (it needs clone-specific
   genes identified first).
5. **No anchor at all** -- then alpha is a hyperparameter, not a learnable quantity. Set a
   global alpha by grid or prior and do not claim per-clade learning.

### Where the claims now stand

* **(a) the tree improves the mapping** -- supported: +0.044 global alpha (9/11 seeds),
  +0.098 oracle per-clade (11/11).
* **(b) per-clade weighting** -- oracle headroom 11/11 with a verified mechanism, AND now
  learnable given a clone-informative anchor: AUC 0.96-0.98 in 40/40 fits at init <= 0.35,
  giving +0.040 over the best global alpha and +0.086 over no tree. Remaining limitation:
  requires the anchor, and requires initialising below 0.5.

### Files

`q15_cnv_anchor.py`, `q16_anchor_quality.py`, `q17_init_sweep.py`, `plot_cnv.py`

---

## The anchor is better used as a FEATURE than as a loss (2026-08-25, final)

The barcode objection: if eTRACER/PEtracer read barcodes in situ AND in the dissociated data,
clone identity is a lookup and SpOTr's clone-level contribution is redundant. Correct, and it
generalises well beyond barcodes.

Note this also invalidates the eTRACER benchmark as constructed: the answer key there IS the
barcode (`spatial_lineage_LE.csv`), so the reported ARI 0.311 -> 0.348 is measured on a task
the data already answers at 1.0. eTRACER remains good evidence for the PREMISE (18/54 clones
significant, non-circular) but is compromised as a mapping benchmark. The non-redundant part
is within-clone placement, which is the coarse-to-fine regime already shown to be weak, and is
not directly measurable anyway since the dissociated cells have no ground-truth position.

### Q18: anchor quality (10 seeds, accuracy over all spots)

| anchor quality | expression | anchor only | expr+anchor | SpOTr | best |
|---|---|---|---|---|---|
| clean | 0.2234 | 0.5645 | **0.6174** | 0.3103 | expr+anchor |
| good | 0.2234 | 0.6221 | **0.6225** | 0.3132 | expr+anchor |
| poor | 0.2234 | **0.4842** | 0.3901 | 0.3063 | anchor only |
| very poor | 0.2234 | 0.1576 | 0.2469 | **0.2933** | **SpOTr** |

SpOTr beats the feature route in 0/10 seeds at clean, good and poor quality; 10/10 at very
poor. So SpOTr wins only when the anchor is so degraded it is worse than expression alone.

### Q19: partial coverage (anchor at "good" quality)

| coverage | feature route | loss route | SpOTr wins |
|---|---|---|---|
| 100% | 0.6206 | 0.3136 | 0/10 |
| 50% | 0.4696 | 0.3124 | 0/10 |
| 22% (eTRACER LE) | 0.3346 | 0.2846 | 0/10 |
| 10% | 0.2770 | 0.2839 | 7/10 |
| 3% | 0.2429 | 0.2715 | 9/10 |

Crossover ~10%. At eTRACER's actual ~22% the feature route still wins.

### Consequence

The CNV-anchor proposal (Q15-Q17) is REAL as a mechanism -- per-clade alpha is genuinely
learnable from a CNV loss, AUC 0.96-0.98 in 40/40 fits -- but it is not the right thing to do
with an anchor. An anchor good enough to make alpha learnable is good enough to map with
directly, and mapping with it directly is about twice as accurate.

**Decision rule**: measure the anchor's quality and coverage. Above the crossover (usable
quality, >~10% coverage), concatenate it into the feature cost and do not bother with
per-clade alpha. Below it, the loss route helps.

### Where this leaves the two claims

* **(a) the tree improves the mapping, with no anchor available** -- holds. +0.044 with a
  global alpha in 9/11 seeds, +0.098 with an oracle per-clade alpha in 11/11. This is the
  honest core result and the one the paper can carry.
* **(b) per-clade weighting** -- headroom is real and the mechanism is verified (compact
  clades want alpha 0.35, dispersed 0.10, global compromises at 0.18), and it is learnable
  given an anchor. But the regime where that is the best use of an anchor is narrow: anchor
  coverage below ~10%, or anchor quality below expression-only.

### Files

`q18_when_needed.py`, `q19_partial_coverage.py`

---

## alpha is not identifiable from expression + tree alone (2026-08-25)

Proposal tested: fit Brownian motion in space along the tree using the mapped positions, and
let that likelihood guide alpha (`q20_bm_likelihood.py`).

It fails, by the mirror of the expression failure. Any objective measuring "how well does the
coupling reconstruct expression" is maximised by the coupling that maximises expression
agreement -- alpha = 0. Any objective measuring "how well do the mapped positions agree with
the tree" is maximised by the coupling that maximises tree agreement -- alpha = 1. BM
likelihood is the second kind.

| objective | selected alpha |
|---|---|
| expression logL | 0.00 in 10/10 seeds |
| BM logL | 0.85 in 10/10 (grid endpoint) |
| OU logL (PSD robustness check) | 0.85 in 10/10 |
| true accuracy optimum | 0.20-0.35 |

BM logL monotone increasing in 10/10 seeds. Not an artefact of the eigenvalue clipping needed
to make H - d/2 positive definite: exp(-d/median(d)), which is PSD by construction for any
tree metric, gives the same answer.

**The joint-likelihood rescue also fails.** logL_expr + w * logL_BM was swept over ten weights
from expression-only to BM-only. It jumps straight from alpha = 0.00 to alpha = 0.85 with
nothing in between; the best weight hits the accuracy optimum in 1/10 seeds. Both terms are
-0.5*N*log(sigma_hat^2) and monotone in opposite directions, and their derivative ratio never
crosses in the interior.

**Fixing the BM rate does not help either** (analytic, not run): with sigma^2 fixed rather than
profiled, logL is proportional to -tr(r' V^-1 r), which is monotone in the same sigma_hat^2
that profiling minimises. Identical ordering, identical argmax.

### The general statement

alpha is the weight between expression and tree. It cannot be selected using only expression
and the tree -- every criterion built from those two is monotone in one direction or the
other. Selecting it requires a THIRD source of evidence, independent of both. That is exactly
what the CNV anchor supplied, and why it worked where held-out genes did not.

Empirical support: 2 expression-loss variants (raw, held-out genes) -> alpha = 0 in 11/11;
2 tree-likelihood variants (BM, OU) -> alpha = 1 in 10/10; 10 joint weights -> no interior
optimum. Not a proof, but the structural argument is general.

### Consequence for non-cancer data

For the embryo, with no clone-identifying modality on the spatial side, alpha is not
learnable. Options, in order of preference:

1. Obtain a third modality (mitochondrial variants, SNVs, in-situ barcodes). Note Q18: if it
   is good enough to learn alpha from, it is better used as a FEATURE.
2. Fixed default alpha = 0.2. Plateau [0.10, 0.35] is safe (mean gain +0.0296, never worse
   than -0.026 in any seed); alpha = 0.20 gives +0.0365, beats alpha = 0 in 10/11 seeds and
   captures 84% of what per-seed tuning would give. Report a sensitivity curve alongside.
3. Do not claim per-clade alpha in this setting.

### Files

`q20_bm_likelihood.py`

---

## Supervised transfer of an alpha RULE (2026-08-25)

Proposal: on data where the mapping is known (spatial lineage tracing, barcodes on both
sides), fit alpha supervised, then carry it to data without ground truth.

The fitted VALUES cannot transfer -- alpha_k is indexed by clade, and clades are
dataset-specific. What can transfer is a RULE: a predictor mapping clade-level features to
alpha_k, trained supervised and applied wherever those features are computable. That sidesteps
the identifiability result (Q20) rather than fighting it: alpha is never selected on the
target data at all.

Features used, all computable without ground truth: clade size; cross-validated expression
separability (scRNA-seq only); within/cross-clade tree distance ratio; and, from an alpha = 0
mapping, spatial dispersion, bimodality, coupling row entropy, effective spots occupied.
Validation is leave-one-seed-out.

### Q21: fails on the original simulator, but that is an adversarial case

LOSO AUC 0.564 (range 0.03-0.83); predicted alpha gives +0.0000 over the best global alpha,
recovering 0% of the oracle's gain.

BUT the simulator assigns `is_dispersed` at RANDOM, independent of every clade property. By
construction only mapping-derived features could predict it, and at ~22% mapping accuracy
those are swamped by mapping error. A compact-but-confused clade looks dispersed. This is the
worst case, not a fair test.

(Harness bug found and fixed here: closures stored in a dict captured loop variables by
reference, so every seed's evaluation used the LAST seed's data. The tell was the oracle
coming out WORSE than global, contradicting the established 11/11. Sanity check added.)

### Q22: how strong must the biology-dispersion link be?

A link was added explicitly with a dial -- dispersed clades get extra program-mixture breadth,
i.e. more transcriptional heterogeneity, observable from scRNA-seq alone with no spatial
information. Plastic/migratory clones being both more heterogeneous and more scattered is the
biologically plausible version of this.

| disp_signal | link AUC (heterogeneity -> dispersed) | LOSO transfer AUC |
|---|---|---|
| 0.0 | 0.616 | 0.514 |
| 0.15 | 0.636 | 0.569 |
| 0.3 | 0.687 | 0.657 |
| 0.6 | 0.840 | 0.829 |

**Transfer AUC tracks the raw link almost exactly.** The predictor recovers essentially all
the available signal and none that is not there.

### Q23: what a working rule buys

At link AUC 0.84 / transfer AUC 0.829, over 6 seeds:

| | alpha=0 | best global | PREDICTED | oracle |
|---|---|---|---|---|
| mean accuracy | 0.2082 | 0.2552 | **0.2760** | 0.3127 |

Predicted alpha beats the best global alpha in 5/6 seeds, mean +0.0208, recovering **36%** of
the oracle's gain. Compare Q21 at AUC 0.56: 0%.

### The practical protocol

1. On a ground-truth dataset (eTRACER, PEtracer -- barcodes label both cells and spots), the
   per-clade dispersion labels are known.
2. MEASURE the link: does any scRNA-derived clade feature predict dispersion? Report that AUC.
3. If it is ~0.8+, fit the rule and transfer it; expect roughly a third of the oracle gain.
   If it is ~0.6, do not -- it will contribute nothing.

Step 2 is a decision criterion computable BEFORE committing to the approach, which is what
makes this different from the objectives that turned out to be monotone.

### Caveats

* Leave-one-seed-out here is SAME-DISTRIBUTION transfer, the easiest case. Transfer across
  tumour types or tissues would be harder.
* 36% of oracle, not 100%.
* Rests on a biology-dispersion link that may or may not exist in real data -- which is
  exactly what step 2 measures.

### Files

`q21_transfer.py`, `q22_bio_link.py`, `q23_transfer_payoff.py`

---

## THE CHECK on real eTRACER data: no transferable link (2026-08-25)

Q22/Q23 said a transferable alpha rule pays only if some scRNA-seq clone feature predicts
spatial behaviour at roughly AUC 0.84 (|rho| ~ 0.6). Run on real data
(`experiments/etracer/check_bio_link_premise.py`).

**First attempt was uninformative, not negative.** Using the 2,000-bin / 1,515-cell subsample
prepared for the OT runs left only 7 clones. The best-of-6-features AUC of 0.833 sat BELOW its
own permutation null (mean 0.874, p = 0.76) and the multivariate LOO-CV AUC was 0.167. Rebuilt
from all 7,568 LE bins and up to 200 cells per clone drawn from the full 91,411-cell matrix,
giving 23 usable clones.

**Label matters.** Coherence and spread are the wrong targets -- what alpha_k needs to know is
not "is this clone clustered" but "does its tree predict where its cells sit", which
`etracer_premise.csv` already measures per clone as a Mantel r with a within-clone permutation
null. Against spread, three features were nominally significant (het rho -0.574 p 0.004,
libsize +0.529 p 0.010, n_detected +0.535 p 0.009) but they are three views of one sequencing-
depth axis, i.e. a technical confound.

**Against the correct label (per-clone tree-space concordance, n = 23):**

| feature | rho vs tree-space r | p |
|---|---|---|
| size | +0.294 | 0.173 |
| het | -0.008 | 0.971 |
| het_pair | -0.008 | 0.971 |
| expr_sep | -0.063 | 0.774 |
| libsize | +0.011 | 0.961 |
| n_detected | -0.089 | 0.687 |

Best |rho| = 0.294; permutation null for best-of-6 has mean 0.355, 95th pct 0.543, so
**adjusted p = 0.70**. Multivariate LOO-CV AUC 0.189 -- far below chance, pure overfitting.

**Power.** With n = 23 clones, |rho| above ~0.56 is detectable at 80% power. That is exactly
the regime where transfer would have paid (simulation: link AUC 0.84). So this does not merely
fail to find a link -- it excludes the useful range. A weaker link could exist but would carry
almost nothing even if real.

### Verdict on the two claims

* **(a) the tree improves the mapping** -- holds. +0.037 with a fixed alpha = 0.2 (10/11 seeds,
  84% of what oracle tuning would give), +0.044 with a tuned global alpha. The plateau
  [0.10, 0.35] is safe. Independent real-data optimum on eTRACER was 0.075, same order.
* **(b) per-clade weighting** -- headroom is real (11/11 seeds, mechanism verified: compact
  clades want alpha 0.35, dispersed 0.10, global compromises at 0.18), but there is no route
  to realising it:
    - not learnable from the data being mapped (alpha is unidentifiable from expression + tree)
    - learnable from a CNV/SNV anchor, but an anchor good enough for that maps ~2x better used
      as a FEATURE (Q18), except below ~10% coverage
    - not transferable via a supervised rule: no scRNA-seq feature predicts per-clone
      tree-space concordance on eTRACER, and the useful effect size is excluded

That is a complete answer rather than an open question, and the negative has a mechanism
behind it in each case.

---

## Claim (b) as an INTERPRETIVE readout: supported (2026-08-25)

Repositioned claim: map from expression + tree (robust to anchor quality), use the anchor only
to learn alpha, and read alpha_k as the per-clade structure-vs-expression contribution.

Supporting fact already in hand (Q18): SpOTr's accuracy is nearly FLAT across anchor quality
(0.310 / 0.313 / 0.306 / 0.293 from clean to very poor) while the feature route swings
0.617 -> 0.252. So the robustness argument is real.

### Q24A: anchor quality IS assessable without ground truth

Two label-free diagnostics, neither using spatial truth:

* `cell_pred` -- cross-validated accuracy of predicting clone identity from the CELL-side
  anchor. Clone labels come from the tree, which is given.
* `spot_fit` -- mean best correlation between a spot's anchor profile and any clone signature.

| quality | cell_pred | spot_fit | feature route | SpOTr | better |
|---|---|---|---|---|---|
| clean | 1.000 | 0.804 | 0.6090 | 0.3101 | feature |
| good | 1.000 | 0.781 | 0.6094 | 0.3136 | feature |
| poor | 0.835 | 0.661 | 0.3848 | 0.3097 | feature |
| very poor | 0.328 | 0.660 | 0.2520 | 0.3010 | **SpOTr** |

`cell_pred` vs feature-route accuracy: Spearman +0.95. **Thresholding cell_pred at 0.5 picks
the better strategy in 39/40 seed-quality combinations.**

So "we do not know how good the anchor is" is FALSE -- it is measurable, cheaply, from the
cell side alone. That is constructive rather than a refutation: it converts the dilemma into a
decision rule.

### Q24B: the interpretation survives a degraded anchor

| quality | alpha AUC | alpha compact | alpha dispersed |
|---|---|---|---|
| clean | 0.975 | 0.152 | 0.021 |
| good | 0.958 | 0.154 | 0.025 |
| poor | 0.936 | 0.149 | 0.026 |
| very poor | 0.794 | 0.186 | 0.049 |

The readout holds across the entire range, INCLUDING the regime where the anchor is useless
for mapping. Separation narrows (7.2x -> 3.8x) but never collapses.

### The resulting recommendation

1. Measure `cell_pred` on the candidate anchor. This is cheap and needs no spatial truth.
2. **Mapping**: if cell_pred > ~0.5, put the anchor in the feature cost. Otherwise map from
   expression + tree.
3. **Interpretation**: learn alpha_k from the anchor either way. It recovers per-clade
   structure-vs-expression contribution at AUC 0.94-0.98 at usable anchor quality and 0.79
   even when the anchor is too poor to map with.

This makes claim (b) a supported INTERPRETIVE claim rather than a performance claim, which is
consistent with everything else measured: oracle > shuffled > inverted (11/11), mechanism
verified (compact clades want alpha 0.35, dispersed 0.10, global compromises at 0.18), learned
alpha recovers the split from any init below 0.5.

### Anchors for non-cancer, and how to screen them

Requirements unchanged: measured on both sides, predicted by the tree (i.e. clonal), and not
the modality driving the OT cost. Cell type fails the second test -- different clones contain
the same cell types, so the tree does not predict it.

* **Mitochondrial variants** -- clonal, callable from scRNA-seq (mgatk, MAESTER); harder but
  feasible on the spatial side.
* **Allele-specific expression / X-inactivation skew** -- clonal in female samples, uses the
  same SNP/BAF machinery as Numbat and CalicoST but does NOT require copy-number changes.
  Probably the best non-cancer option.
* **Somatic SNVs** in expressed genes -- low coverage from RNA but possible.
* **Engineered barcodes** where available.

`cell_pred` screens any candidate before investment: compute it on the cell side, and it
predicts both whether the anchor is worth adding as a feature and whether it will support the
alpha readout.

---

## Anchor-quality diagnostics: how they are computed, and the circularity trap (2026-08-25)

### cell_pred

Both ingredients live entirely on the DISSOCIATED side, so no spatial data is involved:

* per-cell anchor profile -- each cell's CNV/SNV profile, inferred from its own scRNA-seq
* per-cell clone label -- from the tree, cut into clades. This is the `omega` matrix SpOTr
  already requires as input.

Train a classifier on (anchor -> clone label), cross-validate, read the accuracy.

**The trap.** This is only meaningful if the clone labels and the anchor come from DIFFERENT
measurements. If clones were called with Numbat FROM the CNVs, then "can CNV predict clone" is
1.0 by construction and the diagnostic is vacuous. Valid when the tree is from lineage
barcodes and the anchor is CNV, or the tree is from CNVs and the anchor is SNVs/mtDNA.

Note the METHOD is still fine in the circular case -- spot-side CNV is a genuinely independent
measurement of a different tissue, and the OT cost still uses expression, so loss modality !=
cost modality holds. Only the diagnostic breaks.

### Spot-side diagnostics (Q25), for when cell_pred is circular

All computed from the spot anchor and the clone signatures alone; no clone labels anywhere.
Correlate each spot's anchor profile against every clone signature, then summarise sharpness:

| diagnostic | AUC picking the right strategy | best acc | poor vs very poor p |
|---|---|---|---|
| cell_pred | 0.978 | 98% | 0.0002 |
| **softmax** (mean max softmax prob) | **0.975** | **98%** | **0.0002** |
| neg_entropy | 0.975 | 98% | 0.0002 |
| margin (top1 - top2) | 0.943 | 95% | 0.0013 |
| spot_fit (mean max correlation) | 0.849 | 82% | 0.9097 |

Mean values by regime -- softmax 0.527 / 0.477 / 0.349 / 0.280 from clean to very poor.

`spot_fit` fails because max correlation SATURATES: it is 0.661 vs 0.660 at the poor / very
poor boundary, which is exactly where the strategy flips. Sharpness statistics do not
saturate, and softmax matches cell_pred (0.975 vs 0.978) while never touching clone labels.

**Recommendation**: use the spot-side softmax sharpness as the default diagnostic. It is
non-circular regardless of how the tree was built, and loses nothing against cell_pred.

---

## The anchor must not be expression-derived (2026-08-25)

In cancer the tree normally comes from single-cell CNV inference, so the realistic pipeline is
Numbat on the scRNA-seq and CalicoST on the spatial. That is fine for the METHOD -- the OT cost
uses expression and the loss uses CNV, so loss modality != cost modality still holds, and the
tree being CNV-derived does not break it.

But inferCNV, and Numbat's expression module, infer copy number by SMOOTHING GENE EXPRESSION
along the genome. If the spot-side "CNV" is a transformation of spot expression, the loss is a
smoothed version of the expression loss, which is monotone with argmin pinned at alpha = 0.

### Q26: three anchors, matched at 20 bins

| anchor | loss picks alpha>0 | monotone in alpha | learned alpha AUC | alpha compact/dispersed |
|---|---|---|---|---|
| independent (allelic) | 7/10 | 3/10 | **0.956** | 8.0x |
| derived (expression-binned) | **0/10** | **10/10** | **0.436** | **0.8x** |
| mixed (half each) | 6/10 | 4/10 | **0.928** | 7.0x |

An inferCNV-style anchor fails completely: monotone in every seed, pinned at alpha = 0,
learned alpha below chance with the separation INVERTED. Exactly the predicted circular
failure.

A 50/50 mix retains almost all the signal (0.928 vs 0.956), which matters practically because
Numbat and CalicoST both produce joint expression+allele calls rather than pure allelic ones.

### Does the sharpness diagnostic catch it?

Here yes -- sharpness 0.397 (independent) vs 0.276 (derived), gap +0.121, and 0.276 sits at
the same level as a "very poor" independent anchor. But the MECHANISM is informativeness, not
independence: binning 240 genes into 20 bins destroys signal. A sharper expression-derived
anchor could score well and still be circular. **Independence is a pipeline design choice, not
something diagnosable from the data.**

### Recommendation for the cancer pipeline

1. Build the tree however you like, including from CNVs. That does not break the method.
2. The ANCHOR must carry allele/BAF information -- Numbat's allele model, CalicoST. Do NOT use
   a purely expression-based call (inferCNV alone) as the anchor; it fails outright.
3. A joint expression+allele call is fine; the allelic component carries the alpha signal and
   half of it is enough.
4. Then use spot-side softmax sharpness for the quality/strategy decision.

---

## Loss for an allelic anchor (Numbat + CalicoST) (2026-08-25)

`deconvolution_loss_ls` is a LINEAR mixture, spot = sum_k w_k * clone_k. Correct for total copy
number, which is additive over the cells in a spot. NOT correct for BAF, which is a ratio:

    BAF_spot = sum_k w_k * B_k / sum_k w_k * (A_k + B_k)

a ratio of weighted sums, not a weighted average of per-clone BAFs. And allele counts are
binomial, not Gaussian.

### Q27: all three losses perform identically

| depth | loss | picks alpha>0 | alpha AUC | compact/dispersed |
|---|---|---|---|---|
| 20 | ls_linear (wrong link) | 6/10 | 0.978 | 10.1x |
| 20 | ls_ratio (right link) | 6/10 | 0.981 | 10.0x |
| 20 | binomial (right link + noise) | 6/10 | 0.978 | 9.5x |
| 200 | all three | 6/10 | 0.967-0.975 | ~9.5x |

Prediction wrong: neither the link nor the noise model changes the readout. The alpha estimate
depends on the ORDERING of clades by how much the tree helps, not on calibration, so the
distortion washes out. Read depth is irrelevant too, so the binomial likelihood buys nothing.

### But that is regime-dependent

The two links coincide exactly when all clones share the same TOTAL copy number; the error
grows with the spread of total CN. Measured as a fraction of the BAF signal it must resolve:

| regime | CN coefficient of variation | link error / BAF sd |
|---|---|---|
| as simulated | 0.28 | 6.9% |
| 2x spread | 0.53 | 14.8% |
| 3x spread | 0.72 | 20.2% |
| 5x (homozygous del + amp) | 0.93 | 23.5% |

My simulation sits at 6.9%, which is why the wrong link cost nothing. Real tumours with
homozygous deletions and high-level amplifications sit at 15-24%, where it would.

**Recommendation**: use the ratio link. It is a two-line change and always correct; the linear
version is safe only in the low-CN-variation regime tested here. Skip the binomial -- it added
nothing at any depth.

```python
def allelic_loss(gamma, clone_assignments, A_k, B_k, baf_obs):
    w = jnp.einsum('ji,ik->jk', gamma.T, clone_assignments)      # [m, K]
    w = w / jnp.maximum(w.sum(1, keepdims=True), 1e-12)
    p = (w @ B_k) / jnp.maximum(w @ (A_k + B_k), 1e-9)           # ratio link
    return jnp.mean((baf_obs - p) ** 2)
```

Inputs from a real pipeline: `A_k`, `B_k` = per-clone major/minor allele-specific copy number
per segment (Numbat); `baf_obs` = per-spot phased B-allele fraction per segment (CalicoST).

## Anchors for data without copy number (embryo)

Requirements unchanged: on both sides, predicted by the tree, not the cost modality.

1. **Sparse in-situ clonal readout.** If the embryo carries a lineage-tracing construct with
   ANY spatial readout, that is the anchor -- and sparsity is an ADVANTAGE here. Q19: below
   ~10% spot coverage the loss route beats putting the anchor in the feature cost (7/10 seeds
   at 10%, 9/10 at 3%). eTRACER baseline sat at 2.8% coverage, exactly this regime. So a
   low-efficiency spatial barcode readout is close to ideal for learning alpha.
2. **mtDNA heteroplasmy.** Clonally inherited, callable from scRNA-seq (mgatk, MAESTER), and
   mito transcripts are abundant enough in spatial data to genotype. Requires a heteroplasmic
   background -- the germline bottleneck leaves a standard mouse with little diversity, so this
   works in models built for it, not by default.
3. **X-inactivation in a female F1 embryo.** Xi choice at ~E5.5 is clonally heritable, and
   allele-specific expression of X-linked genes measures it on both sides with the same BAF
   machinery. Real but COARSE: it is roughly one bit, so it partitions clades into two groups
   rather than resolving 12. Useful as a partial anchor, not a full one.
4. **Nothing available** -> alpha is a fixed default (0.2, plateau [0.10, 0.35]) and per-clade
   alpha is not claimed for that dataset.

Practical first step: check whether the specific E9.5 dataset has any clonal readout on the
spatial side. If it is a lineage-tracing dataset the answer is often yes and often sparse,
which is the best case rather than the worst.

---

## Sparse in-situ readout: coverage limits, and the eTRACER workflow (2026-08-25)

### Q28: alpha readout vs anchor COVERAGE (as opposed to quality)

| coverage | labelled spots | alpha AUC | sd | compact/dispersed |
|---|---|---|---|---|
| 100% | 246 | 0.964 | 0.037 | 10.7x |
| 33% | 81 | 0.903 | 0.107 | 9.6x |
| 22% | 54 | 0.850 | 0.164 | 6.5x |
| 10% | 25 | 0.819 | 0.120 | 6.1x |
| 5% | 12 | 0.736 | 0.077 | 5.3x |
| 2.8% | 7 | 0.606 | 0.149 | 2.3x |
| 1% | 2 | 0.528 | 0.224 | 1.8x |

Readout holds (AUC > 0.75) down to ~10% coverage; dead below ~5%.

Correction to an earlier overstatement: sparse coverage is where the loss route WINS THE
COMPARISON against the feature route, not where it performs well. At 3% both are weak
(0.272 vs 0.243 against 0.223 with no anchor).

### eTRACER really is the dual-modality situation

Verified from the raw files:

| stage | spatial bins with a clone barcode | coverage | clones in BOTH sc and spatial |
|---|---|---|---|
| baseline | 890 | ~2.5% | 62 |
| early escape | 13,626 | ~38% | 36 |
| late escape | 7,568 / 35,835 | **21.1%** | 45 |

Dissociated side: 46,171 cells with clone labels, 173 clones, plus per-clone ML trees.

Cross-referencing with Q28: EE (~38%) and LE (21%) support the alpha readout (AUC ~0.90 and
~0.85); baseline at 2.5% does not (~0.61). That is consistent with baseline being flat in the
premise test (0/7 clones significant).

### Two valid workflows, one invalid

* **VALID -- same tissue, partial coverage.** The 21% of barcoded bins train alpha; the
  mapping covers all 35,835 bins. This is the eTRACER LE case.
* **VALID -- serial sections of one specimen.** Learn alpha on a barcoded section, apply to a
  non-barcoded section. The transfer problem does NOT apply here: alpha_k is indexed by clade
  and the clades are the SAME clades, so the values carry over directly. No rule-learning
  needed.
* **INVALID -- a different specimen.** Different clones, so alpha_k has nothing to attach to.
  That needs the feature-based rule, which failed the eTRACER check (adjusted p = 0.70).

### The reframing this forces

At LE, 79% of bins have NO clone call. Barcodes cannot place anything there. That -- not
clone-level assignment on the labelled minority -- is where SpOTr earns its place on this data,
and it answers the "if you have barcodes you do not need SpOTr" objection properly.

Note also that for MAPPING at 21% coverage the feature route still wins (0.335 vs 0.285), so
the two uses are complementary rather than competing: barcodes as a feature for the best map,
barcodes as a loss for the alpha readout. Both can be run on the same data.

---

## Semi-relaxed OT more than doubles the tree's benefit on eTRACER LE (2026-08-25)

Balanced OT gives every cell equal mass, so the cell-side clone composition IMPOSES a spot-side
composition. The earlier work showed that a mismatched reference makes a real tree score worse
than no tree. Composition matching fixes it when the arithmetic allows -- but at early escape it
does not: C18 holds 59% of the labelled bins and supplies only 244 cells with trees, so
proportional matching caps the analysis at 413 cells, fewer than the per-clone floor needs.

Semi-relaxed OT removes the requirement rather than working around it: the bin marginal stays
exact while the cell marginal is KL-penalised towards the reference with weight tau, so clone
abundances are FITTED. tau -> infinity recovers balanced.

Sanity check first: the balanced limit of the semi-relaxed path reproduces the standard solver
exactly (alpha = 0 gives 0.3917 in both), so the two code paths agree.

| tau | accuracy at alpha=0 | best accuracy | best alpha | gain | ARI at best |
|---|---|---|---|---|---|
| balanced | 0.3917 | 0.4410 | 0.30 | +0.0493 | 0.2732 |
| 1.0 | 0.3910 | 0.4427 | 0.30 | +0.0517 | 0.2777 |
| 0.1 | 0.3915 | 0.4580 | 0.45 | +0.0665 | 0.3132 |
| **0.01** | 0.3772 | **0.4850** | 0.45 | **+0.1078** | **0.3398** |

Two things worth noting beyond the headline.

**The tree and the relaxation INTERACT.** At tau = 0.01 the relaxation alone is worse than
balanced (0.3772 vs 0.3917 at alpha = 0); it is relaxation PLUS tree that wins (0.4850). Freeing
the marginal is not a general improvement -- it is what lets the GW term shape the composition
instead of fighting an imposed one. That is the mechanism the earlier composition finding
predicted, now confirmed from the other direction.

**The optimal alpha rises with relaxation** (0.30 balanced -> 0.45 relaxed), and 0.45 was the
grid edge, so the optimum may be higher still. Grid extended to 0.75 for the rerun.

This also revises the deployable claim: the +0.032 to +0.049 measured under balanced OT is a
floor, not the ceiling. Under semi-relaxed OT the tree is worth +0.108 accuracy and +0.16 ARI
on this dataset.

## Early escape: the tree replicates, but EE is the weaker dataset

EE was chosen for its coverage -- 38.6% of bins barcoded against LE's 21.1%, and 27 clones with
>=10 labelled bins against LE's ~5. Coverage turned out to be the wrong thing to optimise.

**C18 holds 59% of EE's labelled bins and supplies only 244 of ~9,700 usable cells.** So the
cell and bin populations disagree structurally:

* unmatched (water-filled cap, 4,200 cells): composition TVD **0.703**. Accuracy 0.056-0.095
  throughout, near chance. Semi-relaxed did NOT rescue it -- fitted mass TVD stayed 0.63-0.79
  even at tau = 0.01, so the relaxation cannot absorb a mismatch that large.
* matched: the match is capped by C18's cell supply at **735 cells**, TVD 0.374.

| stage | cells | TVD | alpha=0 | best | gain | best tau |
|---|---|---|---|---|---|---|
| LE | 4,120 | 0.053 | 0.3917 | **0.4850** | **+0.1078** | 0.01 |
| EE | 735 | 0.374 | 0.2286 | 0.2925 | +0.0639 | 0.1 |

EE gains per tau: balanced +0.0585, tau 1.0 +0.0585, tau 0.1 +0.0639, tau 0.01 +0.0544.

**The tree helps at both stages** -- +0.049 to +0.108 at LE, +0.054 to +0.064 at EE, positive at
every tau in both. That is a two-stage replication of claim (a) on real data.

But EE is materially worse in absolute terms (0.29 vs 0.49) and semi-relaxed barely helps there
(+0.064 vs +0.059 balanced) where at LE it doubled the gain (+0.108 vs +0.049). With 735 cells
spread over 5,000 bins there is little for the relaxation to fit.

**Revision to an earlier recommendation.** I suggested EE as the next run because of its
coverage. That was the wrong criterion: what matters is whether the clones covering the mask
are represented in the dissociated data. LE's dominant clone C4 has 1,580 cells for 416 bins;
EE's C18 has 244 cells for 59% of the mask. Late escape is the usable stage.

---

# PEtracer: the premise was measured wrong, and it is the strongest dataset we have (2026-08-26)

`experiments/petracer/petracer_premise.py` reported Mantel r = +0.199, p = 0.103 and cell-pair
r = +0.092, which is why PEtracer was set aside for months. That measurement is invalid.

**The bug.** It used `t.obsm["spatial"]` as a single coordinate frame. That frame holds three
TILED SECTIONS - DBSCAN on the saved coordinates finds three disjoint components of 2890 / 2042
/ 4543 cells, matching samples M3-1 / M3-2 / M3-4 exactly. Both tests break under tiling:

* Test B correlates clade TERRITORY distance against tree distance, but a clade spanning two
  sections has its centroid in the gap between them. The signature is visible in the saved
  file: clade centroid spread 662 is BELOW mean within-clade spread 1086.
* Test C samples random cell pairs, so most pairs are cross-section and their spatial distance
  measures tile layout rather than tissue.

**Recomputed per section** (`petracer_premise_persection.py`):

| section | cells | clades | coherence | coh p | Mantel r | Mantel p | cell-pair r |
|---|---|---|---|---|---|---|---|
| M3-1 | 2890 | 18 | 0.247 | 0.005 | **+0.530** | 0.0001 | +0.330 |
| M3-2 | 2042 | 18 | 0.241 | 0.005 | **+0.500** | 0.0004 | +0.346 |
| M3-4 | 4543 | 17 | 0.359 | 0.005 | **+0.328** | 0.0015 | +0.396 |

All three significant, **signs agreeing across sections** - which is exactly where BCSA failed.
Mean Mantel +0.453; cell-pair correlation +0.33 to +0.40 against eTRACER's pooled +0.092.

**Comparison of every dataset screened, on the premise:**

| dataset | coherence | replicates across sections? | delivered? |
|---|---|---|---|
| PEtracer M3 | 0.241-0.359 | YES (3/3, same sign) | untested |
| eTRACER late escape | 0.309 | n/a (one section) | +0.108 |
| eTRACER early escape | 0.258 | n/a | +0.064 |
| eTRACER baseline | 0.205 | n/a | no (2.5% coverage) |
| BCSA2 best | 0.064 | NO (0.064 vs 0.012) | no |
| BCSA3 / BCSA6 | 0.003-0.016 | signs agree, at zero | no |

**Why PEtracer may give what eTRACER could not.** eTRACER has 45 clones of which 10 drew zero
labelled bins and 8 of 23 drew fewer than five, so per-clade alpha was never estimable
(learned alpha spanned 0.091-0.109). PEtracer has **18 clades per section, all well populated**,
on a 175-gene MERFISH panel, all within a SINGLE clone - so the clades are sub-clonal and
closely related, which is precisely the confusable-expression regime in which the tree is
supposed to matter.

**The catch, stated plainly.** PEtracer reads lineage in situ, so lineage and position are
measured in the same cells: natively a positive control, not an unmatched mapping problem. The
test therefore CONSTRUCTS one, exactly as the simulator was validated - a disjoint split, half
the traced cells becoming the dissociated side with its tree, half binned into pseudo-spots, so
no cell appears in its own spot. That is a construction and must be labelled as such, but it is
the only real dataset where both claims can be tested, since it has per-cell ground truth and
an in-situ anchor at once.

**Method note for the paper.** The premise test is the screening tool this project keeps
recommending, and here it MISFIRED - not on biology but on a coordinate frame. Compute it per
section, always, and check that the sign replicates across sections.

## PEtracer M3-4: claim (a) confirmed strongly, claim (b) blocked by loss misalignment

Disjoint split of section M3-4: 2,281 dissociated cells (with tree), 2,281 other cells binned
into 429 pseudo-spots at 61.4 um pitch, 5.13 cells/spot, 18 clades, 175-gene MERFISH panel.
Verified independently: no cell in its own spot, expression reconstructed from raw counts at
0.000e+00 error, premise survives binning (spot coherence +0.368 p=0.005, Mantel +0.302
p=0.016). Tree metric is real branch lengths from node 'time' (15,393 distinct over 16,889
edges, corr with hop depth 0.348, patristic verified against networkx Dijkstra to 2.8e-14).

Metric: SOFT accuracy -- for a cell assigned to a spot, the fraction of that spot's members
sharing its clade. spot_clade is a strict majority in only 52% of spots and its tie-breaks skew
toward low clade ids, so argmax-against-spot_clade is biased; soft accuracy uses the full
composition and no tie-break. Random-spot floor 0.0977.

### Claim (a): the strongest real-data result in the project

| tau | alpha=0 | best | at alpha | gain | relative |
|---|---|---|---|---|---|
| balanced | 0.1448 | 0.2189 | 0.10 | +0.0740 | +51% |
| 0.1 | 0.1357 | 0.2154 | 0.60 | +0.0797 | +59% |
| **0.01** | 0.1336 | **0.3183** | 0.60 | **+0.1847** | **+138%** |

Expression alone sits 48% above the random floor; tree + semi-relaxed sits 226% above it and is
2.2x the expression-only mapping. eTRACER's comparable gain was +29% relative.

### Claim (b): blocked, and not by estimability

The per-clade fit early-stopped at iteration 0 of 274 with alpha frozen at its 0.10 init
(sd 0.000), so learned / uniform-at-same-mean / shuffled are all identical at 0.2032.

The cause is visible in the sweep:

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.45 | 0.6 | 0.75 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| VAL loss | .03578 | **.03504** | .04038 | .04019 | .03851 | .03805 | .04735 | .05928 |
| test soft acc | .1357 | .2032 | .2039 | .2292 | .2589 | **.3022** | .1987 | .1639 |

**The anchor loss selects alpha = 0.1; accuracy peaks at alpha = 0.6.** This is the strongest
possible anchor -- the exact clade composition of every spot, not a noisy CNV proxy -- and it
still selects on the wrong axis.

**Why, generally.** Composition reconstruction is not placement accuracy. A coupling can
reproduce a spot's aggregate clade proportions while assigning the wrong individual cells to
it: composition is a MARGINAL statistic, placement is the JOINT. Many couplings share a
composition and the loss cannot separate them.

This supersedes the earlier "data starvation" diagnosis. Starvation was real on eTRACER (10 of
45 clones with zero labelled bins) but is absent here -- all 18 clades are populated, hundreds
of cells each -- and the fit fails anyway. Third dataset to show it, and the first where the
anchor is exact.

### Consequence for the two claims

* (a) the tree improves the mapping -- confirmed on two real datasets, +138% relative on
  PEtracer and +29% on eTRACER, with semi-relaxed OT essential in both.
* (b) per-clade alpha -- oracle headroom exists in simulation (11/11 seeds, mechanism verified),
  but alpha cannot be selected from an anchor loss at all, per-clade or global. The blocker is
  that no available objective tracks placement accuracy, which is the identifiability result in
  another guise: expression-reconstruction losses are monotone toward alpha=0, tree-agreement
  losses toward alpha=1, and composition-reconstruction losses select the wrong interior point.

---

## Per-clade alpha is a clade x REGION property, not a clade property (2026-08-26)

Prompted by the observation that the PEtracer tree-space map looks clear. It is -- and testing
that properly reopened claim (b) in a scoped form.

The three M3 sections are disjoint cell sets (cell-name overlap 0/0/0) sharing the SAME 18
clade labels, in the same order. That is the "VALID -- serial sections of one specimen" design
flagged earlier and never run. It removes BOTH blockers at once: all clades populated
(estimability), and alpha fitted against TRUE accuracy (no anchor objective).

### Result 1: the parameter has real held-out headroom (NEW, first on real data)

alpha_k fitted by coordinate ascent on a random half of a section's spots, scored on the other
half. Best global, uniform-at-same-mean and shuffled all selected the same way.

| section | held-out vs best global | vs uniform@mean | z vs shuffled |
|---|---|---|---|
| M3-1 | +0.0449 | +0.0420 | +4.6 |
| M3-2 | +0.0660 | +0.0879 | +2.5 |
| M3-4 | +0.0264 | +0.0210 | +2.7 |
| ALL  | **+0.0458** | **+0.0503** | 9/9 splits positive |

Overfitting gap (fit-half minus held-out) +0.0444, so roughly half the naive oracle gain was
overfitting -- but what survives beats uniform-at-same-mean, i.e. it is the PATTERN not the level.

### Result 2: it does not survive a change of tissue region

Per-clade coherence is measured near-perfectly (within-section split-half r = +0.945 / +0.948 /
+0.982) and yet:

| pair | coherence r | partial r (log-abundance out) | composition TVD |
|---|---|---|---|
| M3-1 vs M3-2 | **+0.956** | +0.948 | 0.113 |
| M3-1 vs M3-4 | +0.004 | +0.138 | 0.449 |
| M3-2 vs M3-4 | +0.010 | +0.192 | 0.385 |

Transfer of the fitted vector matches. Sections want different global levels (M3-1/M3-2 peak at
alpha=0.2, M3-4 at 0.6), so compare against uniform-at-same-mean:

| source -> target | vs best global | vs uniform@mean | z vs shuffled |
|---|---|---|---|
| M3-1 -> M3-2 (same block) | **+0.0407** | **+0.0655** | **+3.7** |
| M3-2 -> M3-1 (same block) | -0.0028 | +0.0059 | +0.7 |
| M3-1 -> M3-4 | -0.1136 | -0.0315 | +0.5 |
| M3-2 -> M3-4 | -0.1238 | -0.0388 | +0.1 |
| M3-4 -> M3-1 | -0.0196 | +0.0165 | +0.7 |
| M3-4 -> M3-2 | -0.0432 | -0.0108 | -0.9 |

### The statement

alpha_k is not a property of clade k. It is a property of clade k IN THAT REGION OF TISSUE. A
clade occupies a territory; whether it looks compact depends on where the section plane cuts it.

This SUPERSEDES the "anchor loss selects the wrong point" diagnosis rather than contradicting
it -- both hold. The new result explains why no objective could have succeeded across regions:
there was no stable clade-indexed target to find. The identifiability result (Q20) is unchanged.

Claim (a) is untouched: global coherence 0.24-0.36 and Mantel +0.33 to +0.53 in every section,
all p < 0.002.

### Workable design for claim (b)

Ground truth on one section, applied to an ADJACENT section of the same block. That is the
eTRACER partial-coverage workflow and the paired-section design a BCSA specimen would need.
Do NOT transfer alpha across regions -- it costs up to -0.12.

### Honest limits

* M3-1/M3-2 are near-adjacent (TVD 0.113, abundance r=+0.877) -- the weakest generalisation
  that still counts as held-out, and only 1 of the 2 directions worked. M3-2 has 204 spots to
  M3-1's 292, so its fitted vector is noisier; that is the likely cause of the asymmetry.
* The random split-half is itself a WITHIN-region test. It establishes that the parameter
  carries signal, not that it generalises. Consistent with the earlier R6 pattern (random
  split-half reliable, adjacent-territory halves not).
* n=2 same-block transfer directions is all PEtracer offers.

### Files

`experiments/petracer/perclade_replication.py`, `perclade_confound.py`,
`run_transfer_matrix.py`, `run_split_half_oracle.py`, `plot_perclade_scope.py`;
outputs `petracer_perclade_replication.csv`, `petracer_transfer_matrix.csv`,
`petracer_split_half_oracle.csv`, `petracer_M3-*_oracle_alpha.csv`, `perclade_scope.png`

### Region-indexed alpha: tested, rejected (2026-08-26)

If the variation is regional, index alpha by REGION instead of clade. Attractive because a
region-indexed alpha could in principle be estimated from SPOT-SIDE observables alone -- no
labels, no ground truth, no mapping -- which is all a single real section offers.
`run_region_alpha.py`, k-means R=4 on spot xy, cells assigned to regions by their TRUE
position (xy_sc), one OT solve per alpha reused across regions.

**(1) The optimal alpha does vary by region.** Best alpha per region:
M3-1 [0.9, 0.2, 0.2, 0.9] (section-wide 0.2); M3-2 [0.75, 0.3, 0.2, 0.45] (0.2);
M3-4 [0.9, 0.6, 0.6, 0.6] (0.6). Within-section spread 0.70 / 0.55 / 0.30.

**(2) It is NOT predictable from spot-side observables.** Two label-free candidates:
local expression coherence (kNN cosine excess) and expression-vs-distance decay.

| observable | vs best alpha (n=12) | within-section spearman |
|---|---|---|
| expr_coh | r=-0.185 (p=0.56), rho=-0.186 | -0.89 / +0.40 / +0.26, mean -0.08 |
| decay | r=-0.358 (p=0.25), rho=-0.422 | -0.89 / +0.40 / +0.77, mean +0.09 |

Signs inconsistent across sections. No usable estimator.

**(3) The headroom is small anyway.** Region-wise ORACLE vs best global: +0.0198 (M3-1),
+0.0231 (M3-2), +0.0061 (M3-4). That is the cheating upper bound and is already below the
per-clade HELD-OUT figure of +0.046. Held out it would be smaller still.

**Verdict: do not build it.** Even a perfect region-alpha estimator buys ~+0.02 oracle, and
there is no estimator. Caveat: only 2 observables tried, R=4 arbitrary, n=12 region-points --
but (3) is decisive and does not depend on the estimator search.

### The recommended configuration for a single unlabelled section

1. Allele CNVs (or any clone-informative third modality) go in the OT COST as a feature
   channel, NOT in the loss. eTRACER at 21% coverage: feature route 0.335 vs loss route 0.285.
2. Semi-relaxed OT, tau ~ 0.01, plus composition matching. Essential in both real datasets.
3. Global alpha FIXED and reported with a sensitivity curve. Not learned, not claimed as
   learned. The sections wanted 0.2 / 0.2 / 0.6, so there is no safe universal default.
4. Coherence screen BEFORE running, reported with call quality (attenuation ~ q^2).
5. No per-clade and no per-region alpha on a single unlabelled section. Per-clade alpha needs
   two adjacent sections of the same block, one with ground truth.

### Files

`experiments/petracer/run_region_alpha.py`, `petracer_common.py` (import-safe solver;
note `run_transfer_matrix.py` does its work at module level, so importing it re-runs it),
output `petracer_region_alpha.csv`

---

## alpha interacts with eps, and the headline number sits at a tuned (alpha, eps) (2026-08-27)

Found while testing whether alpha is confounded with the entropic temperature.
`experiments/petracer/run_alpha_eps_grid.py`, M3-4, soft accuracy, random floor 0.0977:

| eps \ alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.45 | 0.6 | 0.75 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| 0.005 | .133 | .190 | .174 | .148 | .132 | .148 | .123 | .104 |
| 0.0075 | .136 | .218 | .200 | .184 | .188 | .155 | .218 | .153 |
| 0.01 | .141 | **.228** | .215 | .203 | .201 | .187 | .158 | .119 |
| 0.015 | .135 | .218 | .218 | .224 | .247 | .289 | .238 | .113 |
| **0.02** | .134 | .207 | .211 | .234 | .264 | **.318** | .213 | .184 |
| 0.03 | .132 | .199 | .190 | .210 | .237 | .250 | .205 | .085 |
| 0.05 | .136 | .210 | .199 | .195 | .171 | .257 | .170 | .071 |
| 0.08 | .138 | .219 | .211 | .211 | .172 | .161 | .149 | .129 |

1. **The optimal alpha is NOT stable in eps** -- it flips between 0.1 (eps <= 0.01, and 0.08)
   and 0.6 (eps 0.015-0.05). A bimodal switch, not a smooth ridge. This is a large part of why
   alpha selection keeps failing: at fixed eps we estimate a target that moves with eps.
2. **alpha=0 is flat in eps** (range 0.0087). All eps-sensitivity comes from the tree term.
3. **The +138% headline is at a tuned (alpha, eps) = (0.6, 0.02).** A genuine local peak
   (neighbours .289 and .250) but eps=0.02 was a default, never selected. Claim (a) survives --
   the tree gain is positive at EVERY eps, +0.057 to +0.185 -- but the MAGNITUDE is tuned.
   Report the gain range, not the best cell.
4. **alpha ~ 0.1 is robust**: 0.190-0.228 over a 16x range of eps (range 0.038), worth +0.075
   over alpha=0 (+57% relative) at every eps. alpha=0.6 ranges 0.148-0.318 (range 0.170).
   A safe default with a measured robustness claim.
5. expr_loss picks the accuracy-optimal alpha in 0/8 eps slices; comp_loss in 4/8 (but by
   always saying ~0.1, which is right only when the truth is 0.1).

### eps-stability as a selector: does not work

`run_eps_stability.py`. Label-free: solve at several eps, measure how far the hard assignment
moves. Displacement picks alpha=0.1 (acc 0.2125 vs 0.1355 at alpha=0) but only barely over
alpha=0 itself (0.1374 vs 0.1425), and rank correlation with accuracy is not significant
(spearman -0.119, p=0.78; -0.321, p=0.48 excluding alpha=0). Agreement picks alpha=0, which is
trivially stable and useless. A coin flip, not a criterion.

### Solver note: the per-clade path has no across-alpha equalizer

`spotr.build_fgw_cost` ends with `s = Z/den`, holding median|C| ~ constant across alpha.
`perclade_v2.build_perclade_alpha_cost_v2` equalizes ACROSS CLADES only: `target` is the median
over clades of m_k, so for a UNIFORM alpha vector every m_k is equal, s_k == 1 exactly, and the
global magnitude of C is free. Measured drift in median|C| across the alpha grid: 1.78x (M3-1),
1.38x (M3-4). Every per-clade result in this project used that path.

Also note `exp_smooth_clipped(x, M=40) = exp(40 tanh(x/40))` saturates: at small eps the kernel
compresses toward uniform, so LOW eps gives MORE diffuse transport, not less. Row entropy is
U-shaped in eps (min at eps~0.02 on M3-4), not monotone. T_sinkhorn=50 and 400 give identical
results (marginal violation ~5e-7), so this is saturation, not non-convergence.

### Fixing the cost scale breaks the monotonicity on PEtracer

`run_scalefixed_sweep.py`, M3-4, eps=0.02, renormalising C to a fixed median magnitude at every
alpha (same cost geometry, fixed magnitude):

| | accuracy peak | expr loss monotone? | expr argmin |
|---|---|---|---|
| drifting scale (as shipped) | alpha=0.6, 0.3183 | **True** | **0.0** |
| fixed scale | alpha=0.6, 0.2782 | **False** | **0.2** |

alpha=0.2 under fixed scale gives 0.1926 vs 0.1336 at alpha=0 -- +44%, selected with no labels
and no anchor. Does not reach the 0.6 optimum, but "monotone, argmin pinned at 0" has been the
central obstacle since Q10. Entropy drift is essentially unchanged (3.19x -> 3.32x), so this is
NOT the temperature story -- it is the cost magnitude specifically.

Being checked on the controlled simulator (q29_scalefixed.py), where the 11/11 monotone result
actually lives and where the accuracy optimum is 0.10-0.35.

---

## Alpha-selection round 2 (2026-08-27): six candidates, six failures -- and the testbed is the problem

Six label-free criteria implemented and run on the controlled simulator, 10 in-band seeds,
scale held fixed, each with a MANDATORY leaf-shuffled-tree control (real patristic matrix with
leaf labels permuted: preserves the distance spectrum exactly, destroys cell identity).
Baselines: alpha=0 0.2234, fixed alpha=0.2 0.2642, oracle 0.2699.

| candidate | agree | MAE | acc@sel | vs a=0.2 | shuffled control differs |
|---|---|---|---|---|---|
| dual-perturbation-instability | 1/10 | 0.200 | 0.2234 | -0.041 | no (10/10 identical) |
| permuted-tree-excess-agreement | 0/10 | 0.395 | 0.2045 | -0.060 | no (82% of height) |
| vda-two-channel-variance | 1/10 | 0.260 | 0.2215 | -0.043 | weak, wrong direction |
| spot-routed-colocation-rank | 1/10 | 0.335 | 0.2255 | -0.039 | no |
| composition-conditional-excess | 0/10 | 0.500 | 0.2061 | -0.058 | no (10/10 identical) |
| two-sided-split-half-stability | 1/10 | 0.200 | 0.2234 | -0.041 | no (7/10 identical) |

### THE TESTBED CANNOT ANSWER THE QUESTION (the most important finding)

* argmax(acc) = 0.20 in **6/10 seeds**. Total headroom above the constant alpha=0.2 for ANY
  selector is **+0.0056**.
* The zero-information constant selector "always 0.2" scores **agreement 6/10, MAE 0.060** --
  strictly better than the best criterion (5/10, 0.065) while reading no data at all.
* So every "beats fixed alpha=0.2" claim on this simulator is arithmetically true and
  substantively empty, and every "alpha is unlearnable" result from it is UNDERPOWERED.
  PEtracer by contrast has real headroom (0.134 -> 0.318; alpha=0.1 vs 0.6 differ by 0.11).

**Consequence: do not test a seventh criterion before building a variable-alpha testbed.**
Sweep tree-vs-layout concordance (currently fixed at r=0.865) and/or within-clade expression
discriminability over ~6 settings chosen so argmax(acc) spans {0, 0.1, 0.2, 0.35, 0.5, 0.7},
>=4 seeds each. VERIFY BY ORACLE SWEEP THAT alpha* ACTUALLY MOVES before running any criterion.
Pre-register the bar as co-movement: spearman(alpha_hat, alpha*) across datasets, plus mean
accuracy at alpha_hat vs the best SINGLE constant chosen in hindsight over the whole set.

### Standard controls, from now on, in every table

1. real-tree selection, 2. shuffled-tree selection, 3. constant-selector baseline.
A mult=16 control INVERSION in this round was concealed by omitting column 1.

### Four transferable mechanisms (cite to kill future proposals cheaply)

1. **Endpoint identities are free.** V_T(0)=0, V_E(1)=0, S(0)=S(1)=0 are algebra. They
   guarantee an interior optimum EXISTS and say nothing about WHERE. permuted-tree-excess
   defeated monotonicity outright (interior argmax 10/10, monotone 0/10) -- and the shuffled
   tree still reproduced 82% of the height and picked a BETTER alpha (0.40 vs 0.55, truth 0.20).
2. **Perturbation stability has a free knob.** The selected alpha is set by the ratio of two
   injected perturbation magnitudes. p=0.5 -> 0.85 moves selection in 8/10 seeds, mean |delta|
   0.315 -- larger than the entire target band. Removing the knob (test instead of optimise)
   removes the answer with it: the repaired misspecification test returns alpha=0 in 10/10
   because V_T^obs/V_T^null is 13-74x at the FIRST nonzero alpha.
3. **Second-order coupling statistics are blind to placement, provably.** b is uniform, so
   permuting gamma's columns is the maximal gauge move; under it max|dS| = 0.000e+00 for all
   four co-location variants while accuracy collapses 0.2643 -> 0.0883 (floor 0.0833). Any
   statistic factoring through W = gamma gamma^T discards the spot index, which is exactly what
   accuracy measures.
4. **Split-half over tree EDGES measures how much weight the tree carries, never whether the
   tree is RIGHT.** Tree-vs-space discordance is a bias common to both halves.

### The sharpened statement

alpha is not identifiable from (expression, tree) alone -- not from first-order reconstruction
losses, not from second-order coupling structure, and not from any stability functional of the
solution path. Identification requires an observable measured on BOTH the cell side and the
spot side that is not already the objective being optimised. The tree is not one: it is
cell-side only and enters as a second-order relational constraint.

Ruled out to date: (i) reconstruction losses, (ii) tree-agreement likelihoods, (iii) joint
weightings, (iv) perturbation/resampling stability on either channel, (v) endpoint-cancelling
excess statistics, (vi) two-channel variance/reliability decomposition, (vii) second-order
co-location structure, (viii) eps-stability, (ix) region-indexed spot-side observables,
(x) cost-scale renormalisation.

**What does NOT follow:** none of this says a tree-informed map is worse. The tree gain is real
and large (+18% at alpha=0.2 here; +0.057 to +0.185 at EVERY eps on PEtracer). What we cannot
do is pick alpha from the data being mapped.

### Files

`cand_*.py` and `cand_*.csv` in this directory; `alpha_candidates.json` holds the six
definitions with their adversarial reviews.

---

## The variable-alpha testbed, and the settled answer on alpha selection (2026-08-27)

### The simulator cannot be fixed by turning its knobs

`build_varalpha_testbed.py`: 3x3 grid of target_r (0.55/0.70/0.85) x target_confuse
(0.45/0.58/0.72) x 4 seeds. 27 generated, 16 in band (target_confuse=0.45 is unachievable).

| | old testbed | new 3x3 |
|---|---|---|
| alpha* spans | 3 of 6 values | 4 of 6 values |
| headroom over best hindsight constant | +0.0056 | **+0.0058** |
| constant-selector agreement | 6/10 | 7/16 |

alpha* DOES move now and the headroom is unchanged. The problem was never that alpha* was
pinned -- it is that the accuracy curve is FLAT near its peak, so knowing alpha* exactly is
worth +0.006 however much alpha* varies. The knobs also do not steer it monotonically
(mean alpha* by setting: 0.350/0.200, 0.188/0.150, 0.250/0.267 -- no trend).

**The simulator systematically understates alpha-sensitivity.** Its accuracy-vs-alpha curves
span ~0.05; PEtracer spans 0.134-0.318, thirty times more. Do not use it to test alpha
selectors.

### The real testbed: 3 PEtracer sections x 8 eps = 24 configurations

`petracer_alpha_testbed.csv`, from `run_alpha_eps_grid.py` on M3-1/M3-2/M3-4.

alpha* per section across eps (ascending eps):
  M3-1: 0.90 0.20 0.20 0.20 0.20 0.20 0.20 0.20
  M3-2: 0.75 0.75 0.75 0.20 0.20 0.20 0.20 0.90
  M3-4: 0.10 0.10 0.10 0.60 0.60 0.60 0.60 0.10

  oracle 0.1722 | best hindsight constant alpha=0.2 -> 0.1547 | alpha=0 -> 0.1120
  HEADROOM +0.0175 (3x the simulator) | alpha* spans 5 of 8 values | constant agrees 11/24

### Result: no label-free criterion beats a constant, on real data with real stakes

| criterion | agree | MAE | acc@sel | vs constant | vs alpha=0 |
|---|---|---|---|---|---|
| expression loss | 0/24 | 0.377 | 0.1120 | -0.0427 | +0.0000 |
| composition loss | 6/24 | 0.298 | 0.1459 | -0.0088 | +0.0339 |
| coupling entropy | 1/24 | 0.298 | 0.1246 | -0.0301 | +0.0126 |
| entropy-matching | 7/24 | 0.215 | 0.1530 | -0.0017 | +0.0410 |
| **CONSTANT alpha=0.2** | **11/24** | **0.210** | **0.1547** | -- | **+0.0427** |

The expression loss selects alpha=0 in **24/24** configurations: the monotonicity result
reproduces on real data across 3 sections and 8 eps values.

### THE NUMBER THAT MATTERS

  alpha=0 0.1120  ->  constant alpha=0.2 0.1547  ->  per-config oracle 0.1722

**A fixed alpha=0.2 captures 71% of the gain that perfect per-dataset alpha selection would
give** ((0.1547-0.1120)/(0.1722-0.1120)). The remaining 29% is +0.0175, and nothing we can
build recovers any of it.

This is a quantified position, not a punt: alpha is a hyperparameter whose optimal value cannot
be read from the data being mapped, and whose best constant setting is worth 71% of the oracle.
State it that way, with this table.

### Status of the two claims after all of this

* (a) the tree improves the mapping -- CONFIRMED and robust. +0.0427 (+38% relative) from a
  fixed alpha=0.2 over alpha=0, averaged over 24 real configurations; positive at every eps.
* (b) per-clade alpha -- REAL but locally scoped: +0.046 held-out (9/9 splits), transfers
  within a tissue block (+0.041, z=+3.7), harmful across regions (-0.12). Requires ground truth
  on an adjacent section. NOT learnable from the target data.

---

## The PEtracer tree metric is nearly binary, and on 2 of 3 sections the phylogeny LOSES to clade labels (2026-08-27)

Found while plotting the raw data. `plot_tree_space.py`, `run_flat_tree_control.py`.

### The metric saturates

| property of the cell x cell patristic metric | M3-1 | M3-4 |
|---|---|---|
| between-clade distance, coefficient of variation | 0.040 | 0.043 |
| pairs at exactly the maximum (70.0) | 27.7% | **49.2%** |
| variance explained by the same/different-clade BINARY alone | R2 = 0.458 | R2 = 0.564 |
| between-clade variation explained by clade x clade distance | R2 = 0.181 | R2 = 0.293 |

Between-clade distances take ~3 discrete values; within-clade distances span 0-60 continuously.
Most cell pairs coalesce at or near the root, so their patristic distance is the same number.
NOTE this is NOT contradicted by "15,393 distinct branch lengths over 16,889 edges" -- diverse
branch lengths and saturated pairwise distances are entirely compatible.

Clade-level distances: range 13.45-34.98, median 23.23; average-linkage dendrogram has
cophenetic r = 0.840 and a ladder topology, the signature of a near-star matrix.

### The control: is claim (a) about the phylogeny or about clade membership?

Four metrics, identical everywhere else. real; flat (within-clade kept, ALL between-clade set
to their mean); binary (within 0, between 1, no topology); shuffled (leaf labels permuted).
tau=0.01, best accuracy over the alpha sweep:

| section | real | flat | binary | shuffled | alpha=0 |
|---|---|---|---|---|---|
| M3-1 | 0.1124 | **0.1861** | 0.1478 | 0.0977 | 0.0977 |
| M3-2 | 0.1388 | **0.1741** | 0.1663 | 0.1093 | 0.1061 |
| M3-4 | **0.3183** | 0.1875 | 0.2006 | 0.1359 | 0.1336 |
| mean | **0.1898** | 0.1826 | 0.1716 | 0.1143 | |

1. **The tree carries real cell-identity information.** Shuffled gains +0.000 to +0.003 over
   alpha=0, against real's +0.015 to +0.185. Claim (a) is not an artefact of adding any
   structured matrix with the right spectrum.
2. **But on M3-1 and M3-2 the phylogeny LOSES to clade membership.** Flat beats real by +0.074
   and +0.035. The saturated between-clade distances are actively harmful there.
3. **Only M3-4 rewards the topology**, and there decisively: 0.318 vs 0.187 flat, 0.201 binary.
   M3-4 is also the section with the highest coherence (0.36) and the only one with large
   alpha-sensitivity.

The mean (real 0.190 vs flat 0.183) hides a complete reversal between sections. Do not quote
the mean.

### Consequences

* RUN THIS CONTROL BEFORE SUBMISSION and report it. A referee will ask, and finding it
  ourselves is worth much more than having it found.
* The data requirement changes: the tree helps when its metric is GRADED, not merely deep.
  A recorder whose lineages all coalesce near the root will not deliver, however many edges it
  has. Add "between-clade patristic CV" and "fraction of pairs at the max" to the screen
  alongside coherence.
* This may explain part of the per-clade alpha story: if between-clade distances are nearly
  constant, alpha_k mostly modulates how strongly clade k's cells are pulled together, not how
  the tree's topology is used.

---

## FRAMING (2026-08-27): the tree can never be a perfect predictor of space, and that is the point

Raised by Pedro, and the data already collected supports it quantitatively. This belongs in the
paper and should constrain how every future experiment is read.

### The asymmetry, measured

Lineage predicts space; space does NOT predict lineage.

| direction | M3-1 | M3-4 |
|---|---|---|
| a cell's nearest relative IN THE TREE, spatial distance vs random | 0.319 | **0.233** |
| a cell's 5 nearest neighbours IN SPACE, tree distance vs random | 0.957 | **0.886** |

A clone's descendants occupy a neighbourhood (~179 um on M3-4, against 769 um for a random
pair). But any given neighbourhood contains many clones interleaved: only 8-16% of spots are
pure, and the median spot holds 2.0-2.7 effective clades of 18.

### Why this matters for the method

Local lineage mixing puts a CEILING on what any tree-driven mapping can achieve. A cell cannot
be placed better than the lineage composition of the neighbourhood it belongs to allows. We
should state that ceiling next to the accuracy numbers rather than presenting accuracy as if it
were unbounded -- the honest comparison is (achieved - random floor) / (ceiling - random floor),
not accuracy alone. The random floor is 0.0977 on M3-4; the ceiling is set by spot purity.

### Where expression should take over -- and why we cannot show it on PEtracer

Pedro's point: where clades interleave, the tree is uninformative and expression should carry
the mapping. That is the correct motivation for a FUSED objective rather than a tree-only one,
and it is the strongest available argument for alpha being a genuine trade-off rather than a
nuisance parameter.

BUT it cannot be demonstrated on PEtracer, because expression there is inert:
* expression vs clade: Cohen's d = +0.007, R2 = 0.0000
* expression vs tree: pearson +0.017 | expression vs space: pearson +0.028
* clade centroid separation / within-clade scatter = 0.27
On a 175-gene MERFISH panel expression contributes essentially nothing anywhere, so there is no
regime in this dataset where it "takes over". alpha=0 scores 0.134 against a floor of 0.098.

**Consequence for the paper:** the complementarity claim needs a dataset where expression is
informative. eTRACER is whole-transcriptome Stereo-seq and is the candidate. Do not make the
complementarity argument on PEtracer numbers.

**Consequence for experiments:** stop treating "the tree does not predict space perfectly" as a
defect of the data or a sign of a bug. It is the regime the method is designed for. What should
be reported is how much of the ATTAINABLE headroom is captured, and where the residual sits.

### The connection to alpha

This is also the cleanest statement of what alpha is supposed to do: raise it where lineage
predicts position, lower it where neighbourhoods are lineage-mixed. Note that this is a
SPATIALLY varying quantity, not a per-clade one -- which is consistent with the finding that
alpha_k is a clade x REGION property and does not transfer across regions. Region-indexed alpha
was tested and had small headroom (+0.020/+0.023/+0.006) with no label-free estimator, so this
remains a motivation without a working implementation.

---

## CORRECTION to the framing above (2026-08-27, later): PEtracer is the IDEAL case, not a limited one

I wrote above that "the complementarity claim needs a dataset where expression is informative"
and that it "cannot be demonstrated on PEtracer". That is WRONG and Pedro corrected it. Recording
the correct argument, because the wrong one would have weakened the paper.

### The actual use case

  dissociated side: expression + lineage, NO position
  spatial side:     expression + position, NO lineage

Expression is the ONLY quantity measured on both sides. It is therefore the only thing that can
establish a cell<->spot correspondence at all. The GW term relates the tree metric over cells to
the spatial metric over spots, but cannot by itself say WHICH cell goes to WHICH spot. So
expression is doing indispensable work, and its weakness as a LINEAGE signal is irrelevant to
its actual job.

The failure mode the method exists to fix: a map driven by expression alone is free to scatter
clonally related cells, because nothing in the expression data constrains it not to. On PEtracer
that is not hypothetical -- expression carries NO lineage signal (Cohen's d = 0.007, r = 0.017
with patristic distance) while the tree DOES track position (nearest relative 179 um vs 769 um
random). The tree supplies exactly the constraint expression cannot.

### The evidence was already in the alpha sweep and I under-read it

M3-4, tau=0.01, random floor 0.0977:

| alpha | 0.0 | 0.1 | 0.3 | 0.45 | **0.6** | 0.75 | 0.9 |
|---|---|---|---|---|---|---|---|
| soft acc | 0.134 | 0.207 | 0.234 | 0.264 | **0.318** | 0.213 | 0.184 |

**Both endpoints are worse than the interior.** Expression-only 0.134; structure-dominated 0.184;
mixture 0.318 -- more than double either. THE INTERIOR OPTIMUM IS THE COMPLEMENTARITY RESULT.
Plus the shuffled-tree control: same distance spectrum, permuted leaf labels, gains only +0.002
over alpha=0, so the gain is genuine lineage information and not arbitrary added structure.

### Why this is arguably the CLEANEST possible demonstration

Because expression carries zero lineage signal here, the division of labour is unconfounded:
expression provides correspondence, the tree provides clonal coherence, neither substitutes for
the other. On a dataset where expression partially tracks lineage the two contributions would be
entangled and the argument would be weaker, not stronger.

**Do not repeat the earlier claim that this needs whole-transcriptome data.** Paper paragraph
rewritten at `paper/caveat_paragraph.tex`.

### What survives from the earlier entry

The ceiling point stands: neighbourhoods are lineage-mixed (8-16% of spots pure, median 2.0-2.7
effective clades of 18), a cell's 5 nearest neighbours IN SPACE have tree distance 58.8 vs 66.4
random. Lineage predicts space; space does not recover lineage. Report accuracy relative to that
ceiling and the random floor, not in absolute terms.

---

## What alpha_k is really a proxy for (2026-08-27, final): clade explains ~10% of the variation

Arrived at by looking at single-cell cost maps rather than by another sweep. This resolves the
tension between two earlier results that never sat comfortably together:
  * per-clade alpha HAS real held-out headroom (+0.046, 9/9 split-half splits)
  * per-clade alpha does NOT transfer across tissue regions (-0.11 to -0.12)

### The measurement

Per-cell placement error = distance from a cell's true position to the spot it is assigned.
Available because the sc-side and spot-side cells are disjoint halves of the SAME section, so
every cell's true xy is known. Grid of 8 alphas, tau=0.01.

`run_percell_alpha.py`, `plot_percell_alpha.py`, `percell_err_*.npy`, `percell_alpha_*.csv`

**Median placement error by alpha (um):**

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.45 | 0.6 | 0.75 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| M3-1 | 596 | **535** | 563 | 559 | 588 | 635 | 606 | 629 |
| M3-4 | 612 | **446** | 493 | 472 | **446** | **446** | 508 | 675 |

Same U-shape as soft accuracy, in interpretable units, against a random scale of ~760 um.
**Use this metric in the paper** -- a reader understands microns without knowing what soft
accuracy is. Note M3-1 is much flatter (span 100 um) than M3-4 (span 229 um), consistent with
M3-1's low alpha-sensitivity everywhere else.

### The finding

Does the alpha a cell needs depend on its CLADE? Variance in each cell's best alpha explained by
clade membership, against a 999-permutation null:

| section | eta2 | permutation null | p |
|---|---|---|---|
| M3-1 | **0.0997** | 0.0118 +- 0.0038 | 0.001 |
| M3-4 | **0.0635** | 0.0074 +- 0.0026 | 0.001 |

**Clade explains the alpha a cell wants far better than chance -- and explains only 6-10% of it.
90-94% of the variation is WITHIN clade.**

Supporting detail:
* **29-33% of cells are placed best with NO tree at all** (alpha=0). Real heterogeneity, spread
  across clades, and a per-clade parameter cannot express it.
* A single cell's best alpha is NOISY: per-cell error spans a median of 665 um (M3-4) / 686 um
  (M3-1) across the grid, against median errors of 505 / 592 um. The swing exceeds the typical
  error, so no individual cell's argmin should be trusted.
* Per-clade best alpha does spread across the grid (M3-4, 15 clades with >=30 cells: 0:2, 0.1:5,
  0.45:3, 0.6:1, 0.75:3, 0.9:1) -- but the per-clade mean-error curves are nearly parallel.

### Why this reconciles everything

**Clade is a PROXY for what actually varies, not the thing itself.** That explains both standing
results at once: the proxy is correlated enough with the truth to buy +0.046 held-out within a
section, and weak enough that re-drawing the tissue re-draws the correlation, so it does not
transfer. It is also why every label-free estimator of alpha_k failed -- they were being asked
to estimate a quantity that is only a tenth clade-determined.

### Consequence for the parameterisation

The big signal is not between clades. It is that ~30% of cells want alpha=0 and ~30% want
alpha>=0.6, a split running THROUGH clades. Capturing it requires a per-CELL alpha, or one
driven by a local feature -- e.g. how well a cell's tree neighbours agree spatially under the
current coupling -- not a per-clade one. Same conclusion the clade x region result pointed at,
reached independently.

NOT YET TESTED. Do not claim it works; it is a motivated direction, and the two nearest
relatives (the alpha=0-map plug-in, AUC 0.558; the supervised feature rule, adj p=0.70 on
eTRACER) both failed. The distinction that makes it worth trying is that the feature would be
computed from the coupling AT THE CURRENT ALPHA rather than from a fixed alpha=0 map, which is
what made the plug-in circular.

### CORRECTION to the entry above (same day): measure reliability at the level of the PARAMETER

The "clade explains only 6-10% of the variation" framing is misleading and I should not have
left it standing. It measures against TOTAL variance, and the total is mostly noise.

`run_alpha_reliability.py`, `run_clade_reliability.py`. Spots split into two disjoint halves,
each mapping solved independently, giving two independent estimates.

**Per-CELL best alpha is almost pure noise:**

| section | test-retest r | exact agreement |
|---|---|---|
| M3-4 | +0.142 | 23-26% |
| M3-1 | **-0.039** | 17-20% |

So the large within-clade spread is NOT signal, and a finer-than-clade partition would be
partitioning noise. Rules out per-cell alpha as a target on this data.

**Per-CLADE optimal alpha, with a SIZE-MATCHED RANDOM GROUPING control** (essential -- averaging
~150 cells is reproducible on its own, so raw reliability is not evidence of anything):

| section | clade-level reliability | size-matched random groups | clade advantage |
|---|---|---|---|
| M3-4 | **+0.479** (rho +0.530) | +0.006 | **+0.473** |
| M3-1 | +0.109 | +0.108 | **+0.002** |

* **M3-4: clade is a genuinely meaningful partition.** Optimal alpha per clade reproduces at
  r = 0.48 across independent spot halves, far above the random control.
* **M3-1: clade is indistinguishable from a random grouping of the same sizes.** Its apparent
  eta2 = 0.108 is entirely the averaging artefact.

**Revised statement.** Of the RELIABLE variance in preferred alpha, clade captures roughly half
(eta2 0.065 against reliable fraction 0.142 on M3-4), and reproduces directly at r = 0.48. It is
a defensible grouping where the signal exists at all, and useless where it does not -- which is
the same M3-4 vs M3-1 split seen in every other analysis.

**Methodological rule, adopt everywhere:** any claim that a grouping explains variance must be
checked against a SIZE-MATCHED RANDOM grouping. Without it, M3-1 looks like it has clade
structure and does not. Add this alongside the shuffled-tree and constant-selector controls.

**Answer to "what is the right partition":** not finer than clade -- that is noise. The open
target is the ~half of reliable variance clade misses, which is a small absolute quantity on top
of an effect worth +0.046, so the expected return is limited. And on sections like M3-1 there is
no partition to find.

### RETRACTION of the clade-reliability numbers immediately above (same day)

The "M3-4: clade-level reliability +0.479 vs random control +0.006, advantage +0.473" claim is
WRONG. The random control used THREE draws (one per split). Redone with 1486 draws:

  observed clade reliability r = +0.426  (per split +0.450, +0.083, +0.745 -- very unstable)
  size-matched random null:    mean +0.143, sd 0.277, 95th pct +0.593
  z = +1.03,  **p = 0.150**   -- NOT significant

P(a single random draw <= 0.006) = 0.32, so the original control was a likely fluke.
`run_clade_reliability_check.py`.

### Granularity sweep, properly controlled (`run_granularity.py`, `run_granularity_fix.py`)

Coarsening by cutting the CLADE-level tree (the cell-level tree's top is a near-polytomy, so
fcluster jumps 18 -> 2). 3 spot-half splits x 200 size-matched random groupings per level.
Metric is exact agreement of the selected alpha between halves, since correlation over 2-4
points is meaningless.

| groups | min size | exact agree | random null | z | p |
|---|---|---|---|---|---|
| 2 | 128 | 0.667 | 0.731 +- 0.253 | -0.3 | 0.466 |
| 3 | 128 | 0.556 | 0.697 +- 0.227 | -0.6 | 0.809 |
| 4 | 128 | 0.667 | 0.709 +- 0.201 | -0.2 | 0.659 |
| 6 | 128 | 0.611 | 0.662 +- 0.173 | -0.3 | 0.664 |
| 9 | 31 | 0.481 | 0.568 +- 0.143 | -0.6 | 0.686 |
| 18 | 9 | 0.407 | 0.459 +- 0.092 | -0.6 | 0.672 |

**At EVERY granularity the clade grouping is at or BELOW a size-matched random grouping.**
Coarsening does not help. 18 was not too fine; no level is better than arbitrary.

Note the null itself rises steeply as groups get larger (0.459 -> 0.731), which is the averaging
artefact -- exactly why the control is mandatory.

### The tension this creates, and the decisive test NOT YET RUN

Per-clade alpha gives a reproducible HELD-OUT gain (+0.046, 9/9 split-half splits). How, if the
clade grouping is no better than random?

Likely resolution: the gain comes from having 18 FITTED PARAMETERS validated on held-out spots,
not from the parameters being indexed by clade. A RANDOM grouping into 18 groups, fitted the
same way, would then give a similar gain.

**RUN THIS BEFORE CLAIMING PER-CLADE ALPHA ANYWHERE.** If random 18-way groupings match the
clade result, claim (b) as currently framed does not survive: the headroom is real but it is
about model capacity, not about lineage. If clade beats random 18-way groupings on held-out
accuracy, then clade matters for the MAPPING even though its optimal alpha is not reproducible,
which is a weaker but still publishable claim.

### Methodological rule (reinforced)

Any claim that a grouping explains variance or reproduces needs a SIZE-MATCHED RANDOM grouping
control with >=100 draws. Three draws is not a control. This is the second time in one session
an undersampled control produced a false positive.

---

# ⚠ CODE AUDIT (2026-08-27): the alpha=0.6 optimum is a solver artefact

Five independent audits of the OT math, each finding verified adversarially by a separate agent
that had to reproduce it from scratch. Report:
https://claude.ai/code/artifact/47866d03-a598-4f1b-bfa7-08272110422f

## The math itself is correct

compute_Lgw matches POT to 1.1e-16. compute_Lcladegw is the exact Omega-masked half-gradient
(4.5e-17 vs brute force). The semi-relaxed iteration rho = tau/(tau+eps) is the correct Chizat
update (3.5e-9 vs an independent convex optimum). Deconvolution losses and their gradients are
right. Autodiff through the unrolled solver is right. **The defects sit between the formulas and
the solver.**

## D1 CRITICAL -- spotr.py:412-416, semirelaxed.py:57. The kernel is not exp(-C/eps)

`exp_smooth_clipped(x, M=40) = exp(M*tanh(x/M))`. This is not a stabilised exponential; it
solves a DIFFERENT, soft-clipped cost. Re-solved exactly in the log domain on M3-4:

| alpha | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| shipped code | 0.1336 | 0.2313 | **0.2994** | 0.1484 |
| exact log-domain, same cost | 0.1256 | 0.1343 | **0.0890** | 0.0166 |

Chance floor 0.0977. **The alpha=0.6 peak is manufactured by the clip.** No eps reproduces it
(best fit leaves TV 0.51). TV between shipped and exact: 0.237 / 0.753 / 0.966 / 0.955.
Tell: the calibrator guarantees max|C|/eps <= M_cap = 80 while the solver clips at M = 40 --
the guard is set at twice the firing point.

**INVALIDATES: the +138% PEtracer headline, the whole alpha x eps grid, the 24-config alpha
testbed, the flat-tree control, the two-branch results, the granularity sweep, and every
placement-error number. All of it ran through this kernel.**

## D2 CRITICAL -- spotr.py:198. perclade_dc_L is not OT-invariant

It subtracts a per-clade column mean c_{k(i)}[j], not a shift of the form f_i + g_j. Only
f_i + g_j leaves the entropic-OT argmin invariant (it is absorbed into the Sinkhorn scalings).
It exactly annihilates every clade's mean spot profile: sum_i a_k[i] L_pc[i,j] = 0 for all j,k.
Verified sd 5e-17 vs 0.29 raw; ~60% of transport mass moves when patched to _dc_rect; clade
dispersion under an alpha sweep moves 0.6% non-monotonically shipped vs 44% monotone patched.

Mechanically explains the -0.098 (p=0.59) per-clade recovery in semirelaxed.py's docstring:
**the objective literally cannot see which clade prefers which region.**

Affects build_cladefgw_cost (359), sinkhorn_cladefgw (465), make_step_fn_cladefgw (652).
**perclade_v2.build_perclade_alpha_cost_v2 uses _dc_rect and is NOT affected** -- so the two
clade paths in this codebase are different objectives, not two equalizers.

## D3 MAJOR -- perclade_v2.py:134. s_k is a per-clade temperature

Confirms the hypothesis Pedro raised. The module docstring (lines 46-48) guarantees s_k is
independent of alpha_k; it is not, because m_k = median|C0| and C0 depends on alpha_row.
Measured 4x spread in s_k at alpha >= 0.75; clip to [0.5, 2.0] binds for up to 4 of 18 clades.
With a UNIFORM alpha the per-clade path does not reduce to the single-alpha path:
corr 0.9996 / 0.9905 / 0.9282 / 0.9033 at alpha 0 / 0.3 / 0.6 / 0.9.

## Also confirmed

* J_alt=3 is NOT converged: J=3 -> 10 moves M3-4 alpha=0.6 from 0.2994 to 0.2205.
* dc() ignores its weight arguments (minor).
* compute_Lcladegw docstring writes the cross term as a matrix product where the code
  correctly uses a Hadamard product (docs only).

## Refuted -- do not re-raise

* "Semi-relaxed is pinned to the reference composition." FALSE: TVD falls 0.538 -> 0.122 as tau
  goes 1 -> 0.01. The original concern only probed near-balanced tau.
* "The eps sweeps are degenerate because scales are frozen." FALSE: every sweep reuses scales
  frozen at EPS=0.02 by design; grid_M3-4.log reproduces to 4 decimals.

## Fix order

F1 log-domain kernel (carry rho into the semi-relaxed update) -- unblocks everything
F2 _dc_rect at the three clade sites in spotr.py
F3 converge J_alt (3 -> 10+)
F4 per-clade equalizer: make s_k genuinely alpha-independent, or drop it
F5/F6 cleanup

## The open question, and the rule until it is answered

With a correct kernel at eps = 0.02, every alpha >= 0.3 is at or BELOW chance. Either the tree
carries no usable signal, or eps = 0.02 is far too cold once the tanh's implicit regularisation
is removed -- the clip was doing real work as a regulariser. These are not distinguishable yet.

**DO NOT report an alpha optimum in either direction, and do not cite any alpha-dependent number
in this file, until a correct alpha x eps sweep exists on the fixed solver.** Claim (a) as
currently stated is not supported.

---

## ✅ EXACT FGW RESCUES CLAIM (a) (2026-08-28)

Pedro's proposal: before fixing the entropic solver, check whether alpha behaves sensibly under
EXACT OT on the current FGW cost. Better than fixing-then-sweeping because it removes eps
entirely instead of adding a second parameter to sweep.

Run with POT's own reference implementation -- `ot.gromov.fused_gromov_wasserstein`, exact
conditional gradient with an EMD solve per iteration. ZERO custom solver code: no
exp_smooth_clipped, no eps, no J_alt, no per-clade equalizer, no alternation of ours.
`experiments/petracer/run_exact_fgw.py`, M3-4, random floor 0.0977.

| alpha | 0.0 | 0.20 | 0.45 | 0.60 | 0.75 | 0.90 | 1.0 |
|---|---|---|---|---|---|---|---|
| soft acc (C max-normalised) | 0.1459 | 0.1798 | 0.2206 | 0.2330 | 0.2455 | **0.2660** | 0.0299 |
| soft acc (C mean-normalised) | 0.1459 | 0.2165 | 0.2362 | 0.2492 | **0.2640** | 0.2565 | 0.0301 |
| median placement error (um) | 595 | 424 | 408 | 403 | 407 | 409 | 1085 |

**THE TREE HELPS: +0.120 absolute, +82% relative over alpha=0.** Robust to the normalisation of
C_tree vs C_space (best alpha 0.9 vs 0.75; gains +0.120 vs +0.118). Median placement error
595 -> 400 um.

### Three things this settles

1. **alpha=1.0 collapses to 0.030, BELOW the 0.0977 floor.** Pure GW has no cell<->spot
   correspondence -- structure alignment is defined only up to symmetries. This is the model
   demonstrating the framing: expression is the necessary bridge, the tree is the necessary
   constraint, neither alone works. Cleanest evidence for that claim in the project.
2. **The audit's "every alpha >= 0.3 is at or below chance" was about EPS, not about the tree.**
   That was an exact log-domain solve at eps=0.02. The auditor gave two possible explanations
   (no signal / eps too cold once the tanh's implicit regularisation is removed). Resolved:
   eps=0.02 was far too cold. The signal is real.
3. **The alpha curve SHAPE was wrong, not just its magnitude.** Broken solver peaked at 0.6;
   the true curve rises monotonically to ~0.8 then falls off a cliff at 1.0.

### Practical consequence: consider dropping entropic OT entirely

Exact FGW took 0.4-21 s per alpha at n=2281, m=429. Switching to it removes eps,
exp_smooth_clipped and the J_alt convergence question in ONE move -- a far smaller change than
repairing the entropic path. Cost: no differentiability for learning alpha, which no longer
matters since alpha is not learnable anyway (see the alpha-selection entries above).

### Caveats -- do not overstate

* BALANCED only. POT's semirelaxed_fused_gromov_wasserstein frees the SECOND marginal while our
  formulation relaxes the CELL marginal, so that variant needs the problem transposed. Not yet
  run. The +138% headline was semi-relaxed tau=0.01 and is NOT reproduced or refuted by this.
* One section (M3-4), one metric, POT's normalisation rather than our sF/sL.
* Exact OT gives a SPARSE vertex solution; argmax is well defined but the coupling is very
  different in character from an entropic one, so soft accuracy is comparable to our balanced
  baselines but not to the entropic ones.

### What to re-run on the exact solver, in order

1. alpha sweep on M3-1 and M3-2 (does the tree help there too, where the broken solver said it
   barely did?)
2. the transposed semi-relaxed comparison
3. the shuffled-tree and flat-tree controls -- these are the ones that decide whether the gain is
   lineage information or any structured matrix
4. per-clade alpha, on perclade_v2 only after D3 is fixed


================================================================================
2026-08-28  THE CORRECTED PER-CLADE COST  (src/spotr/perclade_exact.py)
================================================================================

WHY THE OLD ONE WAS NOT A COST AT ALL
-------------------------------------
build_perclade_alpha_cost_v2 iterates on   G = (1-a_i) M_ij + a_i * gwggrad(T)_ij.
For that to be dF/dT of a weighted GW term  sum_ijkl w_ik L(C1_ik,C2_jl) T_ij T_kl
we need  w_ik + w_ki = 2 a_i  for every pair (i,k).  Swap i<->k: w_ki + w_ik = 2 a_k.
Therefore a_i = a_k for all i,k.  ALPHA MUST BE UNIFORM.  With non-uniform alpha the
iteration is not minimising anything, so "the optimal alpha_k" was not even defined --
which is a cleaner explanation of the instability we measured than anything about noise.

THE FIX: weight each PAIR by the mean of its two alphas, w_ik = (a_i + a_k)/2.

    dF/dT_ab = (1-a_a) M_ab + a_a * tens(T)[a,b] + tens(T; weights a)[a,b]

The old cost PLUS a second GW term in which alpha weights the OTHER cell of the pair.
At uniform alpha=c this is (1-c)M + 2c*tens = POT's FGW gradient, exactly.

VERIFICATION (test_perclade_exact.py, n=60 x m=40 synthetic)
------------------------------------------------------------
1. gradient vs POT at uniform alpha, product coupling: max |diff| 0 to 2.8e-14 (rel ~1e-15)
2. gradient vs POT at a non-product coupling:          2.2e-15 (a=0.3), 5.3e-15 (a=0.75)
3. coupling vs ot.gromov.fused_gromov_wasserstein:     max|dT| = 0.000e+00 at a=0.1/0.3/0.75/0.9,
                                                       6.9e-17 at a=0.5. 6-10 CG iterations.
4. non-uniform alpha: F 1.026285 -> 0.257683, monotone decreasing. It is a descent method.

HONEST WRINKLE
--------------
5. The OLD parameterisation, SCORED ON THE CORRECTED OBJECTIVE, reaches 0.247377 --
   LOWER than the corrected CG's 0.257683 (gap -0.0103). FGW is non-convex; a CG with an
   exact line search finds a stationary point, not the global optimum, and the old
   iteration's different search direction happened to land in a better basin here.
   So "the parameterisation was wrong" does NOT imply "the mapping was worse".
   It implies the old alpha_k had no target to be an estimate OF. That is the point.

Real-data solve time: 15.4 s (M3-1, 1445x292), 14.0 s (M3-4, 2281x429). Exact EMD, no epsilon.

STEP 3: THE STABILITY TEST, RE-RUN ON THE CORRECTED COST
--------------------------------------------------------
run_two_branch_exact.py -- 6x6 exhaustive grid (a1,a2) in {0,.15,.30,.45,.60,.80},
branch 1 = clades 1-4, exact EMD conditional gradient, 36 solves per section
(585 s M3-4, 358 s M3-1). Split-half is over the SPOTS USED FOR SCORING; the 36
couplings are shared, so the three splits are not independent refits.

  M3-4   grid best (0.80, 0.80) = 0.2655   best diagonal a=0.80 = 0.2655   gain +0.0000
  M3-1   grid best (0.60, 0.45) = 0.1664   best diagonal a=0.30 = 0.1361   gain +0.0302

  held-out:  M3-4  2-branch 0.1706  uniform 0.1859   -0.0153
             M3-1  2-branch 0.1208  uniform 0.0988   +0.0221
  positive in 3/6 splits.

READ THE LANDSCAPE, NOT THE ARGMAX (two_branch_exact.png). Both sections put the
high-accuracy ridge ALONG the diagonal and fall off away from it. On M3-4 the global
max sits exactly on the diagonal. That is the geometric statement that the two branches
want the same alpha. M3-1's +0.03 is one cell off the diagonal on a section whose
absolute accuracy is 0.166 vs M3-4's 0.266.

For comparison, the OLD (broken kernel + non-gradient) two-branch run claimed
M3-1 +0.0483 and M3-4 +0.0056 full-grid. Fixing the cost did not rescue per-clade
alpha -- it made M3-4's advantage exactly zero.

VERDICT: with a cost that is actually a cost and a solver with no epsilon, two-branch
alpha is not supported on PEtracer. The instability we measured earlier was not (only)
an artefact of the broken parameterisation. Claim (b) does not survive at the level of
the two main branches, which was the most favourable partition available -- it was
chosen from the tree/space heatmaps as the split that visibly agrees between the two
geometries.

WHICH PARTITION? A SCREEN THAT COSTS NOTHING  (screen_partitions.py, screen_shrink.py)
--------------------------------------------------------------------------------------
exact_halves_*.npy already holds E[split, half, cell, alpha] = per-cell placement error from
UNIFORM-alpha exact solves. For any candidate grouping: pick alpha per group on spot-half A,
apply on half B. This is an ORACLE UPPER BOUND (it ignores that a mixed alpha changes the whole
coupling), so anything it kills is dead for real.

  M3-4   one global alpha, chosen on A, scored on B: 470.1 um
         2 branches   +2.0 um   random 2-way groupings  +0.4 +- 2.0   z=+0.76
         18 clades    +8.0 um   random 18-way groupings +1.3 +- 6.2   z=+1.09
         per-cell     -24.8 um  (worse: the argmin fits noise)
  M3-1   one global alpha: 576.3 um
         2 branches   -2.7 um   random  -0.6 +- 2.8   z=-0.72
         18 clades   -21.6 um   random -11.9 +- 13.2  z=-0.73
         per-cell    -64.4 um

NEITHER CLADE-BASED GROUPING BEATS A RANDOM GROUPING OF THE SAME SIZE. Third time a
random-grouping control has erased a per-clade effect in this project. The screen also agrees
with the real mixed-alpha solve on 2 branches (screen +2.0 um; real +0.0000 accuracy), which is
a calibration point in its favour.

SHRINKAGE: alpha_i(lam) = global + lam*(per-cell best - global)
  M3-4  470.1 -> 447.2 um at lam = 0.5   (-22.9 um, 4.9%)   <- interior optimum, real signal
  M3-1  monotone uphill from lam = 0     (+0.0 um)          <- no signal at all

So the cell-level effect is REAL IN M3-4 AND ABSENT IN M3-1, and it is CONTINUOUS, not blocky:
shrinkage finds it, every hard partition misses it. That is why the search for "the right
partition" has failed -- the variation is within-clade and graded.

THE REFRAME: alpha should index SPOTS, not cells.
  - the only grouping that ever showed structure was tissue region (region_alpha.log): best
    alpha per region [0.9,0.2,0.2,0.9] M3-1, [0.75,0.3,0.2,0.45] M3-2, [0.9,0.6,0.6,0.6] M3-4;
    within-section spread 0.30-0.70; region-wise oracle headroom +0.0198/+0.0231/+0.0061
    accuracy -- bigger than the two-branch gain.
  - a cell-side partition is circular BY CONSTRUCTION: what predicts the alpha a cell wants is
    where it lands, and where it lands is the unknown. Spot position is OBSERVED.
  - the corrected cost extends symmetrically: weight spot pair (j,l) by (alpha_j+alpha_l)/2.
  - CAVEAT: region_alpha.log assigned cells to regions using their TRUE positions and solved each
    region separately, so that headroom is an oracle. Spot-indexed alpha inside ONE global
    coupling is the non-oracle version and has NOT been tested.
  - it changes claim (b) from "parts of the TREE behave differently in space" to "parts of the
    TISSUE are differently predictable from lineage". Different claim. Pedro's call.

CLUSTERING CELLS IN ALPHA SPACE  (screen_alphaspace.py, screen_alphaspace2.py)
------------------------------------------------------------------------------
Pedro's question: using ground truth, DEFINE a 2-way partition that clusters in alpha space and
beats a global alpha. Cluster cells on the z-scored shape of their error-vs-alpha curve.

FIRST: kmeans labels are arbitrary, so averaging per-cluster alphas across splits mixes the
groups. Order clusters by their fitted alpha before summarising. (My first pass reported
[0.41, 0.43] -- an artefact of label switching, not a finding.)

  circular ceiling (cluster + fit + score on the SAME half):  M3-4 +89 um, M3-1 +112 um
  honest (cluster + fit on A, score on B):                    M3-4  -4.5,  M3-1 -67.7

But that conflates two things. Separating them:

(a) DOES THE PARTITION TRANSFER?  M3-4 YES. Held-out separation of the two groups' optima
    +0.392 +- 0.130, positive in 6/6 splits. Low group's held-out optimum 0.30-0.45, high
    group's 0.60-0.85, every split. M3-1 NO: +0.117 +- 0.143, 3/6.
(b) DO THE ALPHA VALUES TRANSFER?  NO. The half-A argmin is biased to the extremes -- it picks
    0.00-0.10 for the low group whose true held-out optimum is 0.30-0.45. That bias is what
    turns a real partition into -4.5 um.
(c) WITH ORACLE ALPHA ON B AND THE HONEST PARTITION: M3-4 +34.9 um (7.4%) -- but a RANDOM 2-way
    split with the same oracle freedom gets +16.4 +- 17.9, z = +1.03. Not significant.

So on M3-4 there IS a reproducible two-group structure in alpha space, and it still does not beat
a random split, because the error curve is flat near its optimum: each group loses only ~10% by
using the other's alpha, and the global alpha splits the difference.

WHY MULTIPLE ALPHAS ARE NOT SEPARABLE (the maths Pedro asked about)
-------------------------------------------------------------------
Global alpha: F = (1-a)*Lin + a*Quad. Scaling F does not move the argmin, so the solution depends
only on the RATIO a/(1-a). One degree of freedom, and the "sweep" is a 1-D family.

Per-cell alpha: F = sum_ij (1-a_i) M_ij T_ij + sum_ijkl ((a_i+a_k)/2) L_ikjl T_ij T_kl.
No common scale to factor out. Two distinct consequences:

  1. ROW SCALING. Cell i's expression cost is multiplied by (1-a_i). Under fixed marginals a
     constant per row is free, but a SCALE is not -- it changes how hard cell i competes against
     other cells for the same spot. Pure numerics, no geometry. (= the equalizer, audit item F4.)
  2. NON-SEPARABILITY. The pair (i,k) is weighted by the MEAN of the two alphas. You cannot give
     group 1 less tree without also softening the tree constraint group 1 imposes on group 2.
     Cross-pairs couple the groups no matter how the partition is drawn.

(2) is the mechanism behind the diagonal ridge in the two-branch landscape: off-diagonal does not
give each group its own problem, it gives a blended problem that serves neither. So even a
perfect partition cannot express "this group relies less on the tree" in FGW.

WHAT WOULD: a term where a_i multiplies only cell i's OWN contribution. Laplacian / graph
regularisation, sum_i a_i sum_{k in N(i)} w_ik ||T_i. - T_k.||^2, is separable per cell.
UNTESTED, and it is a different model, not a fix to this one.

================================================================================
2026-08-28  OMEGA / THETA: THE MASK MAKES PER-CLADE ALPHA WELL-POSED
================================================================================
Pedro: "isn't the masking matrix Omega taking care of that?"  He is right, and my
non-separability claim was about the UNMASKED cost, which is not the paper's cost.

main.tex eq:cladefgw_cost does NOT drop cross-clade pairs -- it SPLITS them:
    C_nm = (1-a_t(n)) C^F_nm + a_t(n)[(1-th_t(n)) L^x_nm + th_t(n) L^w_nm],  L = L^x + L^w
with Om^w_nn' = 1[t(n)=t(n')] and Om^x = 11' - Om^w. Per-cell coefficients
    u_n = a_n th_n   (within-clade pairs)      v_n = a_n (1-th_n)   (cross-clade pairs)

WHEN IS THE PAPER'S COST A GRADIENT?
  within-clade pairs: both cells share the clade, so u_n = u_n'. Valid for ANY a_t, th_t.
                      THE MASK REMOVES THE OBSTRUCTION ENTIRELY ON THIS BLOCK.
  cross-clade pairs:  swapping n<->n' demands a_t(1-th_t) = const across clades.
So the cost is a valid gradient iff a_t(1-th_t) is clade-independent. Two usable regimes:
  th_t = 1        -> cross term absent, a_t completely free, clades fully decoupled
  th_t = 1 - c/a_t -> cross weight pinned to a shared c, within weight a_t - c free
Contrast with the unmasked cost, where a_t had to be UNIFORM, full stop. So my earlier
"per-clade alpha minimises nothing" is a statement about the unmasked module only.

IMPLEMENTED in src/spotr/perclade_exact.py: tens_within (block-diagonal Om^w, cost
sum_k n_k^2 m instead of n^2 m), theta/labels args on grad_gw/cost/objective/perclade_fgw,
plus cost_paper (the tex verbatim) so the two can be diffed. VERIFIED (test_perclade_exact.py):
  1. unmasked path unchanged: uniform alpha == POT FGW gradient, 0 to 2.8e-14
  2. ONE clade + theta=1 (Om^w all-ones) == POT again: 2.2e-15, 5.3e-15
  3. theta=0.5 == exactly half the unmasked structural gradient, for 4/2/7-clade partitions
     alike (8.9e-16) -- partition-independent, as L^x + L^w = L requires
  4. paper cost vs gradient: 1.3e-15 when a_t(1-th_t)=0.2 for all t; 8.9e-16 at th_t=1;
     1.911 when a_t(1-th_t) varies  <- the obstruction, made visible
  5. per-clade alpha at theta=1: F 0.404150 -> 0.089721, monotone
  6. DECOUPLING: change clade 0's alpha 0.30 -> 0.05 and measure the other clades' rows.
       theta=1    max change 0.000e+00     <- exactly decoupled
       theta=0.5  max change 3.392e-01

(6) is the direct answer to Pedro's question. At theta=1 the clades are literally separate
problems coupled only through the shared spot marginal. The diagonal ridge in the two-branch
landscape is therefore a property of the UNMASKED run, and that experiment has to be redone
at theta=1 before per-clade alpha can be called unsupported.

STILL OPEN: theta=1 deletes every between-clade tree constraint. The flat-tree control says
that costs little on M3-1/M3-2 (flat BEAT real, +0.074/+0.035) but a lot on M3-4 (real 0.318
vs flat 0.188) -- and M3-4 is the only section with alpha-sensitivity. So theta is itself a
trade-off to sweep, not a setting to fix at 1.

THETA=1 SLICE: THE MASK IS VALID AND STILL DOES NOT DELIVER  (run_two_branch_theta1.py)
----------------------------------------------------------------------------------------
Same 6x6 (a1,a2) grid, same scoring and split-half as the unmasked run, now with theta=1 so the
two branches are exactly decoupled and per-branch alpha is a genuine gradient.
M3-4 drops 49% of cell pairs as cross-branch; M3-1 drops 28%. 1916 s / 939 s for 36 solves.

  M3-4   grid best (0.60, 0.60) = 0.2568   best diagonal a=0.60 = 0.2568   gain +0.0000
  M3-1   grid best (0.80, 0.30) = 0.1449   best diagonal a=0.30 = 0.1300   gain +0.0149

  held-out: M3-4 2-branch 0.1638 vs uniform 0.1832  -0.0194
            M3-1 2-branch 0.0995 vs uniform 0.0983  +0.0012
  POSITIVE IN 1/6 SPLITS (the unmasked run was 3/6). Masking made it WORSE, not better.

On M3-4 the global optimum is again exactly ON the diagonal. The mathematical objection is now
gone -- valid gradient, exact solver, branches decoupled -- and per-branch alpha still buys
nothing. That is a much stronger negative than the unmasked one.

THE PREDICTED FAILURE HAPPENED, AND IT IS VISIBLE (two_branch_theta1.png, right panel).
Uniform-alpha diagonal, masked vs unmasked:
  M3-4 masked  0.146 0.204 0.217 0.233 0.257 0.168   <- collapses at a=0.8
  M3-4 unmask  0.146 0.213 0.227 0.237 0.249 0.266   <- still climbing
  M3-1 masked  0.100 0.127 0.130 0.123 0.111 0.103
  M3-1 unmask  0.100 0.123 0.136 0.132 0.103 0.126
Masked tracks unmasked to a=0.6 and then falls off a cliff. This is the translation-invariance
problem: with only L^w, each block's structural term fixes its INTERNAL geometry and nothing
prefers one location for the block as a whole, so as alpha -> 1 the between-block placement is
undetermined and the mapping degrades. Predicted before the run; confirmed.

CONSEQUENCE FOR THE PAPER'S PARAMETERISATION
  theta=1 is the corner where alpha_t is fully free, and it is exactly the corner where the
  model loses the ability to place clades relative to each other. The two things we want from
  the decomposition -- free per-clade alpha, and between-clade placement -- trade off against
  each other along theta. That is a property of the formulation, not of PEtracer.
  ALSO: the well-posed model has K+1 free parameters (u_t = a_t th_t per clade, plus the shared
  cross weight c = a_t(1-th_t)), NOT the 2K that (alpha_t, theta_t) suggests.

NOT YET TESTED: intermediate theta (the c > 0 family), and finer clade levels where blocks are
smaller and within-block distances are less saturated.

INTERMEDIATE THETA: THE FIRST POSITIVE, AND IT IS ONE SECTION  (run_two_branch_theta.py)
-----------------------------------------------------------------------------------------
Grid over the well-posed family (u1, u2, c): u_t = a_t th_t free per branch, c = a_t(1-th_t)
shared. 100 solves per section, 8 workers, 1727 s (M3-4) / 376 s (M3-1). Both the two-branch
best AND the uniform best are optimised over c, so neither gets a free parameter.
CONSISTENCY CHECK: c=0 reproduces the theta=1 run exactly (M3-4 0.2568 at u=(0.6,0.6)).

1. c HELPS AS A UNIFORM HYPERPARAMETER, and this is the first thing in the whole project that
   raised the ceiling on M3-4:
       theta=1 (c=0)   0.2568
       unmasked        0.2655
       c = 0.20        0.2789    <- best mapping obtained on M3-4 so far
   So the paper's within/cross decomposition IS worth something -- as a single shared knob.

2. M3-4: per-branch u adds EXACTLY ZERO at c = 0, 0.10, 0.20 (+0.0000 each). Only c=0.30 shows
   +0.0209, and at a lower absolute (0.2649 < 0.2789). Held out -0.0153.

3. M3-1: per-branch u adds consistently, and the gain GROWS with c:
       c=0.00 +0.0047   c=0.10 +0.0062   c=0.20 +0.0220   c=0.30 +0.0272
   Held out +0.0178 / +0.0135 / +0.0295, 3/3 splits, all three choosing the SAME point
   u=(0.45, 0.30), c=0.30. Absolute best 0.1561 vs 0.1361 unmasked.
   This is the first per-clade result in the project that replicates across all splits.

   Overall 3/6 splits positive (M3-4 contributes 2 ties and 1 negative).

CAVEATS, IN ORDER OF HOW MUCH THEY COULD KILL IT
  a. NO RANDOM-PARTITION NULL YET. Three times in this project a size-matched random grouping
     has erased a per-clade effect of exactly this size. Until that control is run, M3-1's
     +0.02 is not a result. It is affordable: 25 solves per random draw at ~3.8 s on M3-1,
     so ~20 draws in about half an hour. RUN THIS BEFORE QUOTING THE NUMBER.
  b. BOTH OPTIMA SIT ON GRID EDGES. M3-1's c=0.30 is the largest c tested and its curve is
     monotone increasing; M3-4's u=0.60 is the largest u tested. The grid has to be extended
     before either optimum is believed.
  c. The three splits share the same 100 couplings and differ only in scoring spots, so 3/3
     is weaker evidence than it reads.
  d. M3-1's absolute accuracy (0.156) is still far below M3-4's (0.279), and M3-4 -- the
     section with the real tree signal -- is the one showing nothing.

THE NULL KILLS IT. FOURTH TIME.  (run_branch_null.py, branch_null.png)
-----------------------------------------------------------------------
M3-1, c = 0.30, the real branch split vs 40 RANDOM 2-way splits of the same sizes (240/1205),
each run through the identical 5x5 (u1,u2) grid. 1025 solves, 5530 s. The statistic is a
within-partition contrast (best over the grid - best on the u1=u2 diagonal), so no cross-draw
normalisation is needed. The mask depends on the partition, so the diagonal is re-solved per draw.

  FULL-GRID gain   real +0.0272   random +0.0211 +- 0.0088   z = +0.69   p = 0.268  (10/40 >= real)
  HELD-OUT gain    real +0.0126   random +0.0080 +- 0.0066   z = +0.69   p = 0.293  (11/40 >= real)
  BEST ACCURACY    real  0.1561   random  0.1584 +- 0.0087   z = -0.26
  uniform-u best   real  0.1289   random  0.1373 +- 0.0044
  "3/3 held-out splits positive": random partitions average 2.30/3.

A RANDOM SPLIT OF THE SAME SIZES IS SLIGHTLY BETTER IN ABSOLUTE ACCURACY THAN THE TREE'S OWN
TOP-LEVEL SPLIT. The gain is entirely explained by having two free parameters instead of one,
and the 3/3 replication that looked convincing is the modal outcome under the null.

CLAIM (b) IS DEAD ON PETRACER. It has now failed:
  - unmasked exact solver, two branches                (+0.0000 M3-4)
  - theta = 1, branches fully decoupled, valid gradient (+0.0000 M3-4, 1/6 splits)
  - the paper's full well-posed (u_t, c) family         (+0.0000 M3-4)
  - and on M3-1, the one section that looked positive, against a proper null (p = 0.27)
  - plus the free screens: 18 clades, coarser levels, spatial regions, alpha-space clustering,
    all at or below their random-grouping nulls.

WHAT SURVIVES, AND IT IS NOT NOTHING
  c, the SHARED cross-clade weight, is a real and useful hyperparameter. On M3-4 it gives the
  best mapping in the project: 0.2789 at c=0.20, against 0.2655 unmasked and 0.2568 at theta=1.
  So the paper's within/cross decomposition earns its place -- as ONE global knob, not as a
  per-clade one. That is a smaller claim than (b) but it is measured, replicated across the
  alpha grid, and it comes with a mechanism (it downweights the saturated between-clade
  distances without deleting them, which theta=1 does and pays for at high alpha).

RECOMMENDATION: reframe the second contribution around c, drop per-clade alpha_t, and report
the null controls as a strength of the paper rather than omitting them.

================================================================================
2026-08-28  POSITIVE CONTROL: THE METHOD WORKS, THE DATA HAS NO SUCH STRUCTURE
================================================================================
sim_varalpha_branches.py, run_sim_branch_alpha.py, run_sim_branch_null.py, sim_vs_real.png

CONSTRUCTION. Every spot gets exactly n/m = 8 cells, so the uniform marginals the solver assumes
are exactly true. Expression is a smooth random field of position plus noise, IDENTICAL rule for
both branches. The tree comes from a latent per-cell coordinate:
    branch A: latent = true position + noise  -> corr(tree, space) = +0.883
    branch B: latent = pure noise             -> corr(tree, space) = +0.001
One difference only: whether the tree is informative about position.

TWO GENERATOR BUGS FOUND AND FIXED ON THE WAY -- both would have produced a false negative:
  v1 clustered cells around subclade centres, so the true cells-per-spot distribution was wildly
     non-uniform while the solver assumes uniform marginals. That alone dragged OT BELOW plain
     expression nearest-neighbour (0.039 vs 0.124) and no alpha could recover. Fixed by exact
     per-spot occupancy.
  v2 separated the two branches by 4.0 in tree space -> cross/within mean patristic ratio 7.25x.
     C_tree.mean() then squashed within-branch distances to 0.24 while spot distances sat at
     1.01, so the GW loss compared incompatible scales and the tree could only hurt. PEtracer's
     real ratio is 1.13x. Recalibrated the offset to 0.28 -> 1.11x.
  GENERAL LESSON: the GW term is only usable when the tree's WITHIN-BLOCK distance scale is
  comparable to the spot distance scale. Worth adding as a preconditioning check -- a recorder
  with deeply separated clades silently breaks the structural term at any alpha.

RESULT (3 seeds, 100 solves each)
  seed 0: best 0.1806   uniform 0.1589   gain +0.0217
  seed 1: best 0.1667   uniform 0.1311   gain +0.0356
  seed 2: best 0.1536   uniform 0.1285   gain +0.0252     mean +0.0275, 3/3
  DIRECTION IS CORRECT IN EVERY SEED: selected u = (0.15-0.30, 0.00). The tree-informative
  branch takes positive weight, the tree-blind branch takes zero.
  Per-branch accuracy at the chosen point: branch A 0.2396/0.2257/0.1997 vs branch B
  0.1215/0.1076/0.1076. The tree lifts exactly the branch where the signal was planted.

THE SAME NULL, ON BOTH DATASETS (this is the headline)
  simulation, gain:  real +0.0217 / +0.0356 / +0.0252  vs random +0.0003 +- 0.0010 etc
                     z = +21.4, +9.3, +12.8   p = 0.048 (0/20 draws >= real, every seed)
  simulation, best:  z = +49.2, +16.6, +11.2
  PEtracer M3-1:     z = +0.69, p = 0.27;  best accuracy z = -0.26

Same cost, same solver, same statistic, same null design. Enormous in simulation, absent in
PEtracer. THE METHOD CAN EXPRESS AND RECOVER PER-CLADE ALPHA; PETRACER DOES NOT CONTAIN IT.
Note the raw gains are the same size (+0.022 sim vs +0.027 real) -- what separates them is
entirely the null, which is why the null was the right thing to demand.

CAVEAT: branch A's tree is close to a noisy copy of true position (corr 0.883). This shows the
method is CAPABLE, not that the effect size is realistic. The missing number is a sensitivity
curve over tree_noise: how much tree-space concordance is needed before the effect is
detectable? That is the number that says whether any real dataset could show this, and it
belongs in the paper next to the flat-tree control.

SENSITIVITY CURVE, AND THE NUMBER THAT EXPLAINS EVERYTHING
-----------------------------------------------------------
run_sim_noise_curve.py, noise_curve.png. Sweep branch A's tree_noise; at each level run the
(u1,u2) grid at c=0 for the true split and 5 random splits of the same sizes, 3 seeds.

  tree_noise   corr(tree,space)   real gain      random gain        draws >= real
      0.05          0.967          +0.1123    +0.0027 +- 0.0047        0/15
      0.10          0.882          +0.0275    +0.0027 +- 0.0043        0/15
      0.20          0.652          +0.0058    +0.0000 +- 0.0000        0/15
      0.35          0.381          +0.0000    +0.0000 +- 0.0000       15/15
      0.50          0.234          +0.0000    +0.0000 +- 0.0000       15/15
      0.80          0.109          +0.0000    +0.0000 +- 0.0000       15/15
(z is degenerate at 0.20 and below -- every random split gives exactly 0 gain, so the null sd
is 0. Report the draw counts, not z.)

PETRACER'S OWN WITHIN-BRANCH corr(tree distance, TRUE spatial distance), same statistic:
     M3-4  branch A +0.216   branch B +0.200
     M3-1  branch A +0.240   branch B +0.155
     M3-2  branch A +0.218   branch B +0.161

TWO INDEPENDENT REASONS CLAIM (b) CANNOT WORK ON THIS DATA:
  1. LEVEL. 0.16-0.24 sits below 0.38, where the effect is already exactly zero in simulation.
     Detectability needs ~0.65 and a convincing effect needs ~0.88.
  2. CONTRAST. The two branches have nearly the SAME concordance (0.216 vs 0.200 on M3-4).
     alpha_tau needs a SPREAD across clades, and there is none. Even a perfect estimator has
     nothing to separate.

SCREENING CRITERION FOR THE PAPER (this is the constructive output of the whole negative result):
  before attempting per-clade alpha, compute within-clade corr(tree, space) FOR EACH CLADE.
  You need (a) values around 0.6+ and (b) real spread between clades. Report both. This sits
  naturally next to the flat-tree control and the between-clade patristic CV / fraction-at-max
  statistics already in the screen.

Also visual: sim_space.png, sim_tree_vs_space.png -- the simulated tree metric is GRADED
(between-clade CV 0.450, 0.0% of pairs at the max) against PEtracer's CV 0.043 and 49% at the
max. In the spatial-distance heatmap branch A's quadrant is blocky and branch B's is speckle,
which is what a planted per-clade alpha effect looks like.

================================================================================
2026-08-29  A REALISTIC GENERATOR, AND THE CASE FOR CHANGING THE COST
================================================================================
sim_branching.py, check_signature.py, run_branching_sweep.py, branching_sweep.png

WHY THE OLD GENERATOR COULD NEVER SETTLE THE QUESTION
  tree = position + Gaussian noise makes GLOBAL (metric) and LOCAL (neighbourhood) concordance
  move together, so it cannot distinguish a cost that reads one from a cost that reads the
  other. Real lineage decouples them.

  statistic                       PEtracer M3-4 / M3-1 / M3-2   old sim (tn=0.10)
  global corr(tree, space)             0.208 / 0.197 / 0.189       0.883  <- way off
  local  1 - d(tree-NN)/d(random)      0.511 / 0.452 / 0.444       0.703

THE NEW GENERATOR: a spatial branching process. One founder; each generation every cell divides
and each daughter displaces by N(0, sigma) from its parent; patristic distance is the summed
branch length back to the MRCA (lognormal branch lengths). After g generations spatial
separation grows like sigma*sqrt(2g) while tree distance grows like 2g -- CONCAVE, which is
what flattens the global correlation while keeping neighbours close. Plus RECORDER SATURATION:
a finite barcode cannot resolve deep relationships, so all pairs past a depth get one maximum
distance (sat_q sets the fraction). Spots are equal-size k-d partitions so uniform marginals
are exactly true.

SIGNATURE MATCH (no fitting -- these fall out of the mechanism):
                          global    local   btwnCV    @max
  PEtracer M3-4            0.208    0.511    0.043   49.2%
  PEtracer M3-1            0.197    0.452    0.040   27.7%
  branching sat_q=0.5      0.201    0.472    0.078   50.3%
  branching sat_q=0.3      0.205    0.472    0.098   30.3%
  (between-clade CV still ~2x high; everything else lands.)

STEP 2: DOES SQUARE-LOSS FGW EXTRACT ANYTHING FROM IT?
A FOURTH GENERATOR ARTEFACT CAUGHT BEFORE REPORTING: the first run said "alpha=0 always wins,
saturated or not" -- but alpha=0 was at 83x chance. Expression had already solved the task, so
nothing could have helped. Re-ran across expression-noise levels:

  expr noise  tree          a=0     best   at a     gain   x chance   relative
     3.0      SATURATED   0.3197   0.3210  0.10   +0.0013     41       1.004
     3.0      unsat       0.3197   0.3197  0.00   +0.0000     41       1.000
     6.0      SATURATED   0.1556   0.1576  0.10   +0.0020     20       1.013
     6.0      unsat       0.1556   0.1556  0.00   +0.0000     20       1.000
    10.0      SATURATED   0.0745   0.0846  0.20   +0.0101     10       1.136
    10.0      unsat       0.0745   0.0755  0.10   +0.0010     10       1.013
    16.0      SATURATED   0.0319   0.0456  0.20   +0.0137      4       1.430
    16.0      unsat       0.0319   0.0462  0.20   +0.0143      4       1.449

FINDINGS
  1. The tree buys almost nothing until expression is nearly useless (4-10x chance), and even
     then only ~1.14-1.45x. PEtracer's own tree delivers 1.82x (0.1459 -> 0.2655). CAVEAT: that
     is soft (clade-composition) accuracy against exact-spot accuracy here, so the two numbers
     are not directly comparable -- computing the soft metric on the simulator is the fix.
  2. SATURATION IS NOT THE CULPRIT. Saturated and unsaturated behave the same, and saturated is
     slightly BETTER at expr noise 10. So the obstacle is the concave tree-space relation that
     any branching process produces, not the recorder's resolution limit. That kills the
     "just get a better recorder" reading and points squarely at the cost.
  3. THE OPTIMAL ALPHA IS ALWAYS 0.10-0.20 HERE, never the 0.6-0.9 PEtracer's sweep selects.
     Worth understanding -- it may mean PEtracer's high alpha is doing something other than
     using tree geometry (e.g. acting as a clade-identity prior, consistent with the flat and
     binary trees beating the real one on 2 of 3 sections).

CONCLUSION: square-loss GW is the wrong reader for lineage data. It demands metric equality
between tree and space; a branching process only ever supplies neighbourhood structure. This
is the justification for step 3, the Laplacian/graph term, which reads exactly the part that
survives.

CALIBRATED GENERATOR, AND A METRIC BIAS WORTH FIXING IN THE PAPER
------------------------------------------------------------------
The visual comparison (Pedro asked for it) caught what the summary statistics hid: clades were
almost fully intermixed. CLADE COHERENCE (mean within-clade distance / mean overall) was 0.797
against PEtracer's 0.591/0.664/0.641. Fixing it exposed a tension -- raising coherence also
raised both concordances, because one mechanism drove all three. PEtracer has coherent clades
AND weak fine-scale concordance, i.e. a tree that resolves clade membership but not sub-clade
structure. Needed a SECOND mechanism: local_mix, cells shuffling within their clone's territory
after the lineage is set (motility/packing). Final defaults sigma_decay=0.85, local_mix=0.40:

                     coherence   global    local    @max
  PEtracer            0.59-0.66  0.19-0.21 0.44-0.51 28-49%
  branching (final)      0.659     0.256    0.471    50.3%
  (global still ~22% high; between-clade CV 0.078 vs 0.043.)

RE-RUN OF THE ALPHA SWEEP ON THE CALIBRATED GENERATOR (the earlier table was stale):
  expr noise  tree         a=0     best   at a   relative
     3.0      SATURATED  0.3525   0.3701  0.10    1.050
     6.0      SATURATED  0.1673   0.1748  0.20    1.045
    10.0      SATURATED  0.0785   0.0947  0.20    1.206
    16.0      SATURATED  0.0368   0.0550  0.30    1.495
Saturated ~ unsaturated throughout, so saturation is still not the obstacle.

THE METRIC BIAS (metric_bias.png). The simulator's optimal alpha was 0.1-0.3 while PEtracer's
sweep picks 0.75-0.8. Scoring the simulator THREE ways resolves it -- soft clade-composition
accuracy systematically selects a HIGHER alpha than either spatial metric, on data where the
truth is known:
  expr noise   exact-spot   placement   SOFT(clade)
     3.0          0.10        0.10         0.30
     6.0          0.20        0.10         0.60
    10.0          0.20        0.20         0.60
    16.0          0.30        0.20         0.60
Soft accuracy rewards landing among clade-mates, and the tree term encodes clade identity
directly, so it inflates with alpha whether or not placement improves.

ON PETRACER THE BIAS IS REAL BUT MILD -- placement error bottoms at alpha 0.60 (402.8 um) while
soft accuracy peaks at 0.75 (0.2640). NOT the 0.20-vs-0.60 gap of the simulator. Also correcting
myself: I said earlier the placement optimum was 0.4-0.5; that was the coarse grid. The fine
sweep says 0.60.

CONSEQUENCE FOR THE PAPER. Quote claim (a) at alpha = 0.60, where both metrics agree:
    soft accuracy 0.1459 -> 0.2492   (+71%, not the +82% quoted at alpha=0.75)
    median placement error 595.4 -> 402.8 um   (-32%)
Claim (a) stands and is still strong; the headline number and the recommended alpha both move
slightly, and reporting BOTH metrics is the defensible presentation.

================================================================================
2026-08-29  RETRACTION, A PROPER GENERATOR, AND THE REALISTIC PER-CLADE TEST
================================================================================
sim_eden.py, run_eden_sweep.py, run_eden_perclade.py, eden_perclade.png

RETRACTED: "square-loss GW is the wrong reader for lineage data; that justifies the Laplacian."
That was based on a generator that understated the data's structure (no territoriality, and no
tree-reconstruction-error mechanism). WRONG. With a properly calibrated generator, FGW extracts
about as much from simulated lineage data as it does from PEtracer:
                             a=0     best   at a  relative   (SOFT accuracy, like for like)
  Eden expr=6              0.2397   0.3949   0.90    1.647
  Eden expr=10             0.1820   0.3886   0.75    2.136
  Eden expr=16             0.1358   0.3810   0.75    2.805
  PEtracer M3-4            0.1459   0.2640   0.75    1.810
Same optimal alpha, same order of gain, and the same metric split (placement picks 0.45-0.60,
soft picks 0.75). There is NO evidence the cost is failing to extract available signal.

THE GENERATOR NEEDS THREE MECHANISMS, each necessary, none sufficient:
                        coher  global  local    terr
  PEtracer           0.59-0.66 .19-.21 .44-.51  +.20/+.31
  free diffusion         0.659   0.256  0.471   +0.135   <- no territoriality
  Eden growth alone      0.360   0.624  0.851   +0.612   <- global way too high
  Eden + local mixing    0.526   0.415  0.531   +0.236   <- global still too high
  Eden + mixing + TREE RECONSTRUCTION ERROR
                         0.526   0.190  0.484   +0.236   <- matches
The third is the interesting one: a recorder resolves clade MEMBERSHIP reliably but its
branch-length estimates are noisy, so the inferred metric is nearly CATEGORICAL. That single
mechanism drops global concordance 0.42 -> 0.19 while leaving local concordance and
territoriality untouched -- and it predicts the flat/binary trees beating the real tree on 2 of
3 sections, and soft accuracy being the metric that likes high alpha.
(3D growth + slab section was also tried: fixes territoriality but leaves global at 0.33-0.48
and pushes coherence to 0.79. Not the explanation. Remaining mismatches: coherence 11% low,
between-clade CV 0.172 vs 0.043.)

THE REALISTIC PER-CLADE TEST. Two BALANCED, tree-coherent groups of clades differing only in
MOTILITY -- the paper's own mechanism. (The tree's own root split is unbalanced, 305 vs 719, and
its geometry alone shifts the local statistic by -0.33 at equal motility; balanced clade unions
remove that confound.) 3 seeds per level, 15 random partitions of the same sizes each.

  planted contrast   real gain    random gain        random >= real
      -0.026 (none)   +0.0016   +0.0018 +- 0.0040        3/15   <- negative control PASSES
      +0.102          +0.0104   +0.0031 +- 0.0028        1/15
      +0.225          +0.0029   +0.0040 +- 0.0047        7/15   <- NON-MONOTONE
      +0.380          +0.0192   +0.0020 +- 0.0027        0/15   <- clean, 3/3 seeds
Direction is correct wherever the effect appears: the low-motility group always takes the
higher u. PEtracer's own contrasts are 0.011 (M3-4), 0.038 (M3-1), 0.100 (M3-2).

READ IT HONESTLY. The effect is recoverable at the LARGEST planted contrast (+0.380) and
unreliable below it -- +0.225 is indistinguishable from the null in two independent runs, so
there is no clean threshold to quote and the +0.102 point (1/15) should not be leaned on. What
is solid: the negative control passes, so the machinery does not manufacture effects; and every
PEtracer contrast sits at or below the weakest level tested, where the gain would be ~+0.01
absolute -- under the noise of our held-out evaluations.

CONCLUSION FOR THE PAPER. Claim (b) needs a between-clade motility difference far larger than
anything PEtracer exhibits, and even then buys ~+0.02 absolute accuracy. That is a quantitative,
mechanistic reason for the negative -- much stronger than an unexplained null -- and it comes
with a screening criterion: measure per-clade local tree-space concordance and require real
spread before attempting alpha_tau.

================================================================================
2026-08-30  WHERE THE TREE'S SIGNAL ACTUALLY LIVES -- AND THE DECOMPOSITION ERROR
================================================================================
plot_tree_signal_classes.py, per_clade_benefit.png, who_the_tree_helps.png

Pedro's correction, accepted: alpha_tau being relative to expression is not a defect. It is a
trade-off weight, so of course it depends on both terms. The question is which tree signal to
decompose and how.

FIRST, WHO THE TREE HELPS (per CELL). The tree term enforces clonal coherence, so it helps cells
sitting inside their clone and misplaces cells that have strayed from it. Binning M3-4 cells by
distance from their own clade centre (in units of that clade's spread):
    core   (closest 25%)   tree helps 58%   median +155 um
    inner                             52%           +70 um
    outer                             51%           +28 um
    strays (furthest 25%)             36%            +0 um
    Spearman rho = -0.185, p = 5e-19, n = 2281.
The misled example (cell 2040) is not a tree error: all 25 of its nearest relatives are correct
and in its own clade. It has simply left its clone -- clade 2 spans 195 um and this cell sits
1802 um from the centre.

SECOND, DO CLADES DIFFER? YES, and my "it is all within-clade" was too strong.
    eta^2 = 0.0641 vs permutation null 0.0063 +- 0.0024, p = 0.0005
    tree removes 439 um from clade 18 and ADDS 307 um to clade 14
BUT NOT BECAUSE OF COHESION:
    clade spatial cohesion   vs benefit: rho = +0.168, p = 0.550
    stray fraction           vs benefit: rho = -0.004, p = 0.990
    tightness IN THE TREE    vs benefit: rho = -0.007, p = 0.980
    how badly expression placed it:      rho = +0.529, p = 0.043  (uncorrected, 15 clades,
                                          does not survive correcting for 5 hypotheses)
Clade 14 is the MOST compact clade in the section and the tree hurts it most. Between-clade SD
of benefit 191 um against within-clade SD 481 um -- real signal, 2.5x under the noise, which is
why every fitted alpha_tau failed to transfer.

THIRD, AND THIS IS THE ACTIONABLE ONE. Bin cell pairs by patristic distance and measure their
spatial separation relative to a random pair (1.00 = no information):
                                  fraction of pairs      spatial dist / random
                                  M3-4  M3-1  M3-2       M3-4   M3-1   M3-2
  same clade, resolved            10.1%  7.2%  6.8%      0.591  0.664  0.641   ATTRACT strongly
  different clade, still resolved 40.7% 65.1% 63.0%      0.753  0.947  0.910   attract weakly
  at the saturation maximum       49.2% 27.7% 30.2%      1.288  1.213  1.270   REPEL
Same-clade pairs are 100% resolved, but only 10-20% of resolved pairs are same-clade.
Within the resolved range the GRADED distance carries almost nothing (0.62 / 0.72 / 0.68 across
sub-bins on M3-4, non-monotone). The signal is essentially three-valued, not metric.

THE DECOMPOSITION ERROR. The paper's Omega^w / Omega^x split is same-clade vs different-clade.
That puts the middle class (different clade but resolved -- 41-65% of all pairs, ATTRACTIVE at
0.75-0.95) in the same bucket as the saturated class (REPULSIVE at 1.21-1.29). Omega^x mixes
pairs whose spatial meanings are opposite, and one shared weight cannot serve both. This is a
concrete, measured defect, and it explains why the FLAT tree beat the real one on M3-1 and M3-2:
flattening removes the conflation.

PROPOSED: split by the RESOLUTION BOUNDARY, not by clade membership.
    Omega_near = same clade & resolved      (strong attraction)
    Omega_mid  = different clade, resolved  (weak attraction)
    Omega_far  = at the saturation cap      (repulsion)
Three weights, SHARED across clades -- no per-clade overfitting, and each is interpretable as
how much to trust lineage at that resolution level. It is a re-masking of machinery that already
exists (spotr.perclade_exact takes arbitrary block masks), not a new model, and it predicts its
own outcome: w_far should come out near zero or negative on the sections where flat beat real.
NOT YET TESTED.

RESOLUTION SPLIT: TESTED, AND IT IS ONE SECTION  (multimask.py, run_multimask*.py)
----------------------------------------------------------------------------------
Split the structural term by TREE RESOLUTION rather than clade membership:
    w_near  same clade, resolved        (measured 0.59-0.66x random spatial distance: attract)
    w_mid   different clade, resolved   (0.75-0.95x: attract weakly)
    w_far   at the saturation ceiling   (1.21-1.29x: REPEL)
Scalar weights shared across clades, so the cost is always a valid gradient. Uniform weights
reproduce POT's FGW exactly, so ordinary FGW is a point inside the grid.

FAST PATH. The `resolved` mask is EXACTLY block-constant at the clade level -- 324/324 clade-pair
blocks pure, 0.0000% of cells misassigned, all three sections. Resolution is a property of clade
PAIRS, not of cells. That collapses the heavy term to a single n^2 m product: 277 s -> 39 s.

HELD-OUT (weights chosen on half the spots, scored on the other half, median placement error):
  M3-4  split 383.8 um vs uniform 405.6 um   +21.8 um, 3/3 splits, same weights (2.0,1.0,0.5) each
  M3-1  split 450.0 um vs uniform 471.5 um   +21.6 um, 3/3 splits
  M3-2  split 467.9 um vs uniform 467.7 um   -0.3 um,  1/3
Weight ordering is w_near > w_mid >= w_far everywhere, matching the measured attract/repel.

BUT THE NULL (true clade-pair resolution matrix replaced by random symmetric matrices of matched
density -- same class sizes, same parameter count, only WHICH pairs are called resolved changes):
  M3-4  real +17.6   random +5.5 +- 2.6   0/8 reach it   -> +12.1 um attributable to the structure
  M3-1  real +16.8   random +17.4 +- 6.0  3/8 reach it   -> -0.6 um   ALL of it is extra capacity
  M3-2  real  +6.2   random  +6.2 +- 0.0  8/8            -> degenerate, nothing tested

DEGENERACY, worth knowing about. When the optimum has w_mid == w_far the two classes merge and
the resolution matrix drops out entirely -- the run tests nothing. That happened 9/9 times on
M3-2 (so M3-2 genuinely prefers merging them) and 5/9 on M3-4's null draws. A first null on a
coarse grid {0,0.5,2.0} was degenerate on two of three sections and reported null SD exactly
0.000, which is what exposed it. Any future grid must contain values that let w_mid != w_far.

VERDICT SO FAR: the resolution split is real on M3-4 ONLY. +12.1 um attributable out of +17.6
total, against a tree contribution of 197 um overall, so ~6% of what the tree buys. 0/8 null
draws reach it, but 8 draws only bounds p at 0.11 -- NOT yet significant, needs ~30 draws
(~2 h on M3-4). M3-1's apparent gain is entirely the extra parameters. M3-2 has nothing.
Consistent with M3-4 being the only section where the real tree beat a flattened one.

VERDICT ON THE RESOLUTION SPLIT (agreed with Pedro, 30 Aug)
-----------------------------------------------------------
KEEP as a diagnostic, DROP as a model component.

What is durable and belongs in the paper:
  - SATURATION. 49% / 28% / 30% of cell pairs (M3-4 / M3-1 / M3-2) sit at exactly the ceiling
    of 70.0. Resolved pairs span 0.7-65.1 continuously, then a gap, then a spike. For those
    pairs the tree says "not close" and nothing more -- a censoring marker, not a distance,
    which square-loss GW nonetheless treats as a number.
  - THREE MESSAGES, and they are per-cell, not just aggregate. Median over all 2281 M3-4 cells
    of (distance to that class / distance to a random cell): near 0.50, mid 0.73, far 1.46.
    91.6% of cells put their saturated partners further than their same-clade partners;
    82.8% have near < mid; 69.3% show the full ordering.
  - This explains the flat-tree control (flat beat real on M3-1 and M3-2): flattening discards
    the false precision in the 70.0s.

Why the three-weight MODEL does not earn its parameters:
  - M3-4 median error 398.2 -> 380.6 um, but 55.4% of assignments are unchanged, 23.6% improve
    (median +306 um) and 20.9% worsen (median -289 um). The net is wins barely outnumbering
    losses, and improved/worsened cells are spatially intermixed with no pattern.
  - Null: only +12.1 of the +17.6 um is attributable to the true resolution structure on M3-4;
    on M3-1 it is -0.6 um (all capacity); on M3-2 the optimum degenerates to w_mid == w_far.
  - +12.1 um against the 197 um the tree contributes overall = ~6%, on one section of three,
    and 0/8 null draws only bounds p at 0.11.

THE REAL LEVER, if this is pursued further: the split has to change WHICH cells move, not just
how many. Churning 44% of the section to net 17 um means the reweighting is not selective --
it is not identifying the cells the tree should be trusted for. Per-cell trust (e.g. whether a
cell's own relatives agree about where they are) is the quantity that would be selective;
a global reweighting of pair classes is not.

================================================================================
2026-08-30  SELECTIVE REWEIGHTING AND THE ITERATIVE SCHEME -- BOTH CLOSED
================================================================================
run_percell_trust.py, run_loss_probe.py

(1) CEILING TEST FOR PER-CELL SELECTIVE REWEIGHTING. alpha_i computed (not estimated) from a
trust statistic, two global parameters (base, scale), solved with the verified pair-mean cost.
M3-4, median placement error:
    best uniform alpha                     397.1 um
    ORACLE trust (distance from own clone) 445.9 um   -48.8   WORSE
    observable (tree-neighbourhood tight)  468.8 um   -71.8   WORSE
    SHUFFLED oracle values                 382.1 um   +15.0
THE ORACLE DIRECTION IS WORSE THAN RANDOM. Even perfect knowledge of which cells have strayed
from their clone does not help when used to down-weight the tree for them.
WHY: non-separability, again. Under the pair-mean weighting w_ik = (a_i+a_k)/2, lowering a_i
does not only free cell i from the tree -- it also weakens the tree constraint cell i IMPOSES
on its relatives. Telling a stray "do not trust the tree" simultaneously stops it anchoring the
clone-mates it should still anchor. The reweighting cannot act on one side of a pair.
(The shuffled arm's +15 um is most likely basin luck: FGW is non-convex and we know 44% of cells
move for a 17 um change. Do not read it as random heterogeneity helping.)

(2) CAN ANY SELF-SUPERVISED LOSS GUIDE ALPHA? This is the precondition for EVERY iterative
scheme, including the paper's eq:data_objective. Sweep alpha; compare the truth against losses
computable without positions:
  alpha   err     deconv   heldout  coherence
   0.00  595.4    2.7531   2.4095   0.8860
   0.30  417.4    2.8549   2.4109   0.8920
   0.60  402.8    2.8684   2.4211   0.9065     <- truth's optimum
   0.90  440.6    3.1294   2.4792   0.9239
  deconvolution NLL   rho -0.548 vs error   picks alpha = 0     (truth 0.6)
  held-out genes      rho -0.429             picks alpha = 0.2
  spot coherence      rho -0.619             picks alpha = 0
ALL THREE POINT THE WRONG WAY. Placement error FALLS with alpha while every available loss
RISES, so each would choose less tree than the truth wants, and two would choose none at all.

CONSEQUENCE FOR THE PAPER: eq:data_objective, as written, would drive alpha toward 0 on this
data. The iteration is not the problem -- the objective is. An alternating scheme built on these
losses converges confidently to the wrong answer, and would look like it was working.

WHY the losses fail: they are all goodness-of-fit on EXPRESSION. Adding tree structure moves
cells away from their best expression match by construction, so any expression-based loss
penalises exactly the thing that improves spatial placement. A loss that could guide alpha has
to be orthogonal to expression -- which is the reason the CNV/subclone anchor was proposed in
the first place. That anchor is worth revisiting on this evidence; the three losses tested here
are not viable and should not be used.

2026-08-30  SELECTIVE REWEIGHTING: FOUR VARIANTS, ALL NEGATIVE
---------------------------------------------------------------
run_percell_trust.py, run_rogue_trust.py, run_iterative_rogue.py, src/spotr/trustfgw.py

Pedro's proposal: rogue cells are probably MISASSIGNED, so the objective should push toward
clade coherence, with rogueness recomputed from the current map rather than taken on input.
The premise checks out -- cells sitting far from their clade ARE the poorly-supported clade
calls (tree margin 13.69 for core cells vs 8.87 for strays; rho = -0.234, p = 8.4e-30, and the
margin is a pure TREE quantity, no spatial input). But acting on it does not work.

M3-4, best uniform = 397.1 um median placement error:
  pair-mean weight, oracle strayness            445.9   -48.8
  pair-mean weight, observable                  468.8   -71.8
  pair-mean weight, SHUFFLED                    382.1   +15.0  (basin luck, not a result)
  product weight, observable margin             437.6   -40.5
  product weight, oracle strayness              416.0   -18.9
  product weight, SHUFFLED                      396.6    +0.5
  ITERATIVE self-consistent rogueness (lam 1.0, scale 0.25):
      err 397.1 -> 432.7 -> 457.6 -> ... -> 438.2 over 8 iterations
      mean trust 1.00 -> 0.90, fraction at zero 0.0% throughout

The product weight w_ik = a_i a_k was built specifically to fix the pair-mean form's inability
to remove a cell (a_i = 0 there still leaves the pair weighted by a_k/2). Verified: a_i = 0
zeroes the row exactly and stops the cell constraining others. It did not help. So the
parameterisation was never the obstacle.

The iterative scheme is WELL BEHAVED -- trust converges to ~0.84-0.90, nothing collapses to
zero, no positive-feedback runaway. It simply converges to a worse map, degrading on the first
iteration and never recovering. No null control was run because there is no positive effect to
control for.

WHY, most likely: expression alone places cells at 595 um against 398 um for tree+expression.
Cutting a cell loose from the tree does not give it "no constraint", it gives it the WEAK
constraint. Even a misdirected tree pull keeps a cell in a plausible region. Compounding this,
the OT coupling is a joint object under marginal constraints, so freeing some cells changes
where all the others may go -- per-cell interventions do not decompose.

CONCLUSION FOR THE PAPER. The tree helps as a GLOBAL STRUCTURAL CONSTRAINT, not as a per-cell
or per-clade prior. Every attempt to modulate who gets to use it has failed: per-clade alpha,
coarser tree levels, spatial regions, alpha-space clustering, planted motility groups, per-cell
trust (4 variants). The lever that remains is the REPRESENTATION of the tree -- what its
distances mean -- which is where the saturation finding lives.

================================================================================
2026-08-31  THE EXPRESSION COST WAS UNFAIR. FIXED. CLAIM (a) GETS STRONGER.
================================================================================
prep_fair_expression.py, run_fair_sweep.py.  Pedro asked whether the expression cost was fair.
It was not, and I had taken the input data as given for the entire project -- the solver, the
mask, the parameterisation and the null controls were all audited; the data build was not.

THE BUG. prep_petracer.py builds a pseudo-spot as
    spot = log1p(normalise(SUM of its member cells' raw counts))
    cell = log1p(normalise(that cell's raw counts))
log1p is concave, so log1p(mean) >= mean(log1p): the spot profile is inflated relative to the
average of its own members in the exact space where the cost is computed. Summing raw counts
also weights members by library size. Neither is biology.
DIAGNOSTIC: a spot-side cell finds its OWN spot as nearest neighbour only 40.7% / 48.1% / 48.2%
of the time (M3-4 / M3-1 / M3-2).

THE FIX (Pedro's, "average of normalised"): normalise and log1p each spot-side cell FIRST, then
average within the spot, so the spot is the centroid of its members in the cost space.
    own-spot recovery 40.7% -> 80.7%,  48.1% -> 86.7%,  48.2% -> 86.4%
    expression-only placement improves by 69 / 79 / 87 um
Everything else (disjoint 50/50 split, lattice, tree) is byte-identical, so the difference is
attributable to the aggregation alone.

EFFECT ON THE HEADLINE, and it is the opposite of what I predicted:
                  current                       FAIR
  M3-4   595.4 -> 402.8  (-32%, a=0.60)   549.7 -> 356.8  (-35%, a=0.75)
  M3-1   605.5 -> 477.2  (-21%, a=0.20)   566.9 -> 410.7  (-28%, a=0.45)
  M3-2   627.2 -> 464.9  (-26%, a=0.45)   600.9 -> 419.7  (-30%, a=0.45)
I expected a stronger expression term to need LESS tree. Wrong: the optimal alpha shifts UP and
the tree's relative contribution GROWS. Reading: the two costs were not merely weak but mutually
INCONSISTENT -- the tree was partly fighting a distorted expression landscape. With the spot as a
true centroid the tree can be trusted further.

Claim (a) is now 28-35% on all three sections instead of 21-32%, and M3-1 stops looking like a
weak outlier (it gains the most, +66 um at its optimum).

EVERYTHING DOWNSTREAM WAS COMPUTED ON THE UNFAIR COST and its absolute numbers are superseded:
the per-clade sweeps, the resolution split, the coherence work, the simulator calibration
targets. The NEGATIVES are unlikely to flip -- they were null-controlled, and a uniformly better
expression cost does not create per-clade structure where there was none -- but that is an
assumption until rechecked, and the per-clade benefit's correlation with "how badly expression
placed that clade" (rho = +0.53) is the one result most at risk, because the Jensen bias was
worst exactly where a spot's member cells were most heterogeneous.

LESSON: audit the data build before the model. This was the single largest improvement in the
project and it came from questioning an input nobody had looked at.

================================================================================
2026-08-31  THE PETRACER PAPER (Koblan et al., Science 390, eadx3800, 2025)
================================================================================
Read for context. Several things change how we should read our own results.

OUR DATA IS M3 T1. "we generated MERFISH data from three sections of an additional tumor
(M3 T1) ... assaying 104,219 total cells" with the EXPANDED 175-GENE LIBRARY (matches our 175
genes). Fig 6D: "Phylogeny for all malignant cells with >60% integration BC detection efficiency
across M3 T1 S1 to S3 ... Branches are colored by clade assignment (Clades 1 to 18)". So our 18
clades ARE the paper's clades, and M3-1/M3-2/M3-4 are its sections.

BRANCH LENGTHS ARE IN DAYS. Methods: "We use inferred branch lengths for this calculation, so
path lengths represent the estimated number of days since the LCA divided."
=> C_tree(i,j) = 2 x (days since the LCA of i and j).

THE TREE IS EXACTLY ULTRAMETRIC. Verified on our matrix: 0.0% of 20,000 random triples violate
the ultrametric condition, median relative gap 0.0000. All leaves sit at the harvest time.
=> the 70.0 ceiling is 2 x 35 days: pairs whose LCA is the FOUNDING CELL. 0.0% of them are
same-clade, exactly as a subtree definition of clades requires.

CORRECTION TO OUR EARLIER FRAMING. I described the 70.0 as "a censoring marker, not a distance,
which square-loss GW nonetheless treats as a number". That is wrong as stated: it IS a distance
(divergence at the root). What is genuinely limited is RESOLUTION near the root -- the recorder
was tuned to ~70% edit completion over 4-6 weeks, and the paper's own simulations say "60 to 80%
edit site saturation maximizes reconstruction accuracy" -- so early divergences that cannot be
resolved collapse onto the root. The pile-up at 70.0 is therefore part real topology (a star-like
early phylogeny; the paper reports exponential growth for all clades, i.e. clades established
early) and part unresolvable early structure. Our empirical three-class result is unaffected; its
INTERPRETATION improves:
    near = recent LCA          -> 0.59x random spatial distance
    mid  = intermediate LCA    -> 0.75x
    far  = LCA at the founding cell -> 1.29x
i.e. SPATIAL PROXIMITY DECAYS WITH DIVERGENCE TIME, and cells that diverged at the tumour's
founding sit further apart than random. That is a clean biological statement, not a data defect.

WHY THE FLAT TREE BEAT THE REAL ONE, REINTERPRETED. If tree distance is divergence TIME, there is
no reason for it to be linearly comparable to spatial distance -- diffusion gives separation ~
sqrt(time), and growth under confinement saturates. Square-loss GW asks for a numerical match
between the two, so the problem is the LINK FUNCTION between divergence time and expected spatial
separation, not noise in the tree. This is directly actionable: the link is measurable from the
data (mean spatial distance as a function of tree distance) and a monotone transform of C_tree
would make the square loss appropriate. It is also exactly the "adjust the tree distances" arm of
Pedro's iterative proposal, now with a principled target.

CLADES WERE HAND-PICKED. "High-confidence clades were manually selected and numbered according to
their location in the phylogeny." They are NOT a uniform-depth cut, so clade size and clade depth
vary by construction -- relevant to our finding that clade SIZE is the only significant predictor
of per-clade tree benefit (rho = +0.56), which may be partly a selection artefact of how clades
were drawn.

TREE DISTANCE BEATS RAW BARCODE DISTANCE, per the paper: "Cells with more-similar sets of LMs
(lower LM distances) also tended to be spatially close, although this trend was much weaker than
that observed for phylogenetic distance". So the reconstruction adds real value over the raw
character matrix -- worth citing when we justify using the tree at all.

MULTI-SCALE SPATIAL CONFINEMENT, and it supports the decomposition idea. "LMs defining later
branches in the tree, representing cells with more recent common ancestors, are confined to
progressively smaller spatial regions." Early marks span large regions, late marks small ones.
A depth-dependent weighting of the structural term is therefore biologically motivated by the
paper's own observation, not just by our fitting.

MOTILITY IS REAL AND ACKNOWLEDGED. "although spatial intermixing of clades indicates that 4T1
cells remain motile in vivo" -- supports our stray-cell finding and Pedro's reading that some
rogue cells are genuine outliers rather than misassignments.

KNOWN CLADE-SPACE STRUCTURE IN M3 T1: the lung-adjacent cancer-cell module "was enriched for
clades 13 to 16". So there IS documented clade-space association in exactly our tumour -- a
useful external check on any mapping we produce.

================================================================================
2026-08-31 (overnight)  FACT 0, THE CONCORDANCE DECOMPOSITION, AND A METHOD
experiments/night/*, src/spotr/genfgw.py, tests/test_genfgw.py
================================================================================

FACT 0 -- WHAT A SQUARE LOSS ACTUALLY READS FROM THE TREE
----------------------------------------------------------
Under fixed marginals the square-loss GW term expands as

    sum_ijkl (C1_ik - C2_jl)^2 T_ij T_kl
      = sum_ik C1_ik^2 p_i p_k    (constant -- row marginal)
      - 2 <C1, T C2 T'>
      + sum_jl C2_jl^2 q_j q_l    (constant -- column marginal)

so   argmin_T GW(T) = argmax_T <C1, T C2 T'>.   Two consequences, both verified numerically
(00_affine_invariance.py; the three-term expansion matches ot.gromov.gwloss to 1e-10, the
affine identity to 1e-15):

  1. AFFINE INVARIANCE.  C1 -> a*C1 + b changes the objective by a multiplicative a (absorbed by
     alpha) and an additive constant b*q'C2q.  Rescaling, mean-normalising, max-normalising or
     shifting the tree metric CANNOT change the mapping.  This retro-explains two earlier
     results: "robust to max- vs mean-normalisation (best alpha 0.9 vs 0.75)" was affine
     invariance showing up as a pure alpha shift, and the AFFINE arm of run_calibrated_tree.py
     landing within 0.7 um of raw on M3-4 was not a null result, it was the theorem.
  2. THE TREE ENTERS THROUGH ONE SCALAR: the covariance between patristic distance and the
     spatial distance between the images.  Everything else the tree knows is invisible.

CORRECTION TO AN EARLIER NUMBER.  This file records "global corr(tree, space) 0.208 / 0.197 /
0.189".  That statistic was computed WITHIN each of the two top-level branches and averaged
(check_signature.py passes `grp`).  The plain overall correlation is +0.410 (M3-4), +0.266
(M3-1), +0.323 (M3-2).  Both are meaningful -- the gap between them says most of the linear
concordance comes from the top-level split -- but they must not be quoted interchangeably.

THE CONCORDANCE DECOMPOSITION
------------------------------
If GW can only read a correlation, the question is how large that correlation can be made.  Two
transforms are free, phi on the tree side and psi on the spatial side, and the pair maximising
corr(phi(t), psi(s)) is the classical maximal correlation, computed by one SVD of the normalised
joint histogram (correspondence analysis):

    p(t,s) / (p(t) p(s))  =  1 + sum_r rho_r phi_r(t) psi_r(s)

  section   plain corr   best phi alone   maximal corr (rho_1)   rho_2    rho_3
  M3-4        +0.410         +0.521            0.530             0.230    0.047
  M3-1        +0.266         +0.298            0.306             0.146    0.032
  M3-2        +0.323         +0.374            0.379             0.171    0.042

So a square loss on the raw distances leaves 15-29% of the available concordance unread, and
almost all of the recoverable part is on the TREE side.

WHAT THE TRANSFORMS LOOK LIKE (relationship.png, components.png)
  phi_1 is nearly a STEP at the saturation ceiling: strongly negative for every resolved pair,
        positive at 70.0.  Read: the tree's dominant spatial message is "did these lineages
        diverge at the tumour's founding, or later".  Cells diverging at the founding sit 1.2-1.3x
        FURTHER apart than two random cells.
  phi_2 is a short-range term, very negative at small t and flat above ~50: recent relatives are
        neighbours.  This is the local-vs-global concordance split, recovered automatically.
  psi_1 is monotone but COMPRESSIVE past ~1.5 mm -- beyond that, spatial distance carries no
        further lineage information.

BINNING MATTERS.  Quantile bins collapse (the tree metric is heavily tied) and leave the
informative short-divergence range in one enormous bin.  genfgw.tree_bins uses equal width in
DIVERGENCE TIME, with the ceiling as its own bin and sparse bins merged rightwards.

THE MODEL (src/spotr/genfgw.py)
--------------------------------
    F(T) = (1-alpha) <M,T> + alpha sum_r rho_r sum_ijkl (phi_r(C1)_ik - psi_r(C2)_jl)^2 T_ij T_kl

Ordinary FGW is the rank-1 identity case.  Cost per gradient: R products of size n^2 m -- one
ordinary GW gradient times R.  Exact conditional gradient, EMD steps, closed-form quadratic line
search, no epsilon.

VERIFIED (tests/test_genfgw.py): with R=1 and phi, psi the standardisations of C1, C2, the
coupling equals ot.gromov.fused_gromov_wasserstein's to max|dT| = 0.0 - 6.9e-17 at five alphas,
and on real M3-4 data to 3.6e-16 (356.8 um both).  Monotone descent with a rank-2 kernel.

RESULT 1 (oracle, transforms fitted on the section being mapped, 02_learned_loss.py)
  section  expression   FGW    rank1   rank2   rank3   rank4
  M3-4        549.7     355.9  364.9   331.3   323.2   336.2
  M3-1        566.9     410.7  423.4   405.0   403.3   407.6
  M3-2        600.9     416.1  411.3   403.3   397.3   414.5
Rank 3 best in 3/3, rank 4 worse in 3/3.  Rank 1 alone is NOT better than FGW: the gain needs the
second component, i.e. it needs the local term as well as the saturation step.  phi-only and
psi-only ablations also fail -- both geometries have to be re-read.

RESULT 2 (the honest protocol, 09_split_estimation.py).  Fit phi, psi on a DISJOINT HALF of the
cells; map the other half.  This is the realistic PEtracer protocol: only part of a section
recovers high-confidence barcodes, so use those cells to learn the relationship and map the rest.
  section   expression   shuffled-pairing control   FGW     learned rank 2
  M3-4        549.7              544.6              364.4       342.1
  M3-1        566.9              556.8              437.8       430.5
  M3-2        600.9              572.5              442.4       419.3
LEARNED BEATS FGW IN 11 OF 12 SPLITS (2 seeds x 2 halves x 3 sections).

CONTROLS (10_controls.py, 12_family.py)
  - shuffled pairing: refit the transforms on the same half after permuting the (t,s) pairing.
    Same estimation, same rank, same knots, same alpha grid.  Result 545/557/573 um -- no better
    than expression alone.  The gain is the relationship, not the extra capacity.
  - power-transform family: phi = t^gamma, psi = s^delta over a 6x6 grid containing FGW at
    (1,1).  The FITTED transform beats 100% / 97% / 100% of the 36 members.  It is the right
    shape, not any shape -- and no fixed power works across sections (M3-4 prefers s^4, M3-1 and
    M3-2 prefer s^0.25), so it has to be estimated.
  - robustness: over the whole weight grid the learned loss has a lower MEDIAN error too
    (418->396, 517->503, 511->466).  FGW blows up past its optimum (M3-1 reaches 666 um);
    the learned loss stays in a 400-460 band.  This matters because w cannot be tuned on
    unlabelled data.

RESULT 3 -- CLADES ARE NOT THE RIGHT DECOMPOSITION, AND NOW WE KNOW WHY (11_clade_vs_time.py)
  (a) Patristic distance predicts same-clade membership with AUC 0.999 / 0.997 / 0.996, because
      the clades are subtrees of an ultrametric tree.  A FREE TRANSFORM OF DIVERGENCE TIME
      THEREFORE ALREADY CONTAINS EVERY WITHIN/BETWEEN-CLADE WEIGHTING.  The paper's Omega^w /
      Omega^x split is a coarse special case of phi, and cannot add to it.
  (b) The within-clade tree-space correlation replicates between HALVES of one section
      (pearson +0.917, spearman +0.897, n=26) but NOT between SECTIONS (+0.345 and +0.380
      against M3-4; only M3-1 vs M3-2 agree at +0.692).  So it is a property of HOW THE SECTION
      CUTS THE CLADE, not of the clade.  That is the mechanism behind every failed per-clade
      alpha in this project, and it is a positive statement: decompose by divergence time, which
      is a property of the tree, not by clade, which is a property of the slice.

RESULT 4 -- THE UNMATCHED APPLICATION (08_cross_section.py, 13_application.py)
Map one section's cells (expression + tree, positions withheld entirely) onto ANOTHER section's
spots.  No cell is shared and no position in the target is used to build the cost.  Scored on
clade territory: per clade, the predicted density over the target's spots against the density of
the target's OWN cells of that clade, weighted by clade size.
  pair            expression only   FGW     learned   clade labels permuted
  M3-1 -> M3-4         0.011        0.065    0.079         +0.002 +- 0.016
  M3-2 -> M3-1         0.081        0.183    0.210         -0.001 +- 0.017
  M3-4 -> M3-2         0.057        0.096    0.103         +0.002 +- 0.019
  within-section ceiling: 0.265 (M3-4), 0.214 (M3-1), 0.193 (M3-2)
The tree recovers ~59% of the within-section ceiling; expression alone recovers ~22%.  On
M3-2 -> M3-1 the cross-section map (0.210) essentially matches M3-1's own within-section
ceiling (0.214).

RESULT 5 -- THE FIRST LABEL-FREE WEIGHT SELECTOR THAT WORKS (14_select_w.py)
Every earlier criterion (deconvolution NLL, held-out genes, spot coherence) drove alpha to 0
because they were all goodness-of-fit on EXPRESSION, and the tree necessarily moves cells away
from their best expression match.  The reference gives a criterion that is not about expression:
a mapping induces its own joint distribution of (tree distance, distance between the assigned
spots); pick the weight whose induced joint is closest (KL) to the reference joint.
  FGW:      picks a weight within +0.0 / +3.7 / +4.7 um of the oracle.
  learned:  +18.9 / +44.7 / +14.1 um -- the criterion is biased here because the learned loss
            optimises something close to the criterion itself, so it pushes w upward.
So the selector is unbiased for FGW and self-serving for the learned loss; a debiased version
restricted to the canonical components the loss does NOT use is the obvious fix.

LIMITATION, STATED PLAINLY.  The transforms must be estimated in the tissue being mapped.
Fitted on a different SECTION they transfer no better than a wash (03_transfer.py: M3-4 365.4 vs
FGW 355.9; M3-1 411.2 vs 410.7; M3-2 402.3 vs 422.4), and the fitted phi moves 2.03x further
between sections than between halves of one section (10_controls.py).  Sections of one tumour
differ enough -- 49% / 28% / 30% of pairs at the ceiling -- that the relationship is section
specific.  The design implication is concrete: barcode a fraction of the section you intend to
map into.

MARGIN DIAGNOSTIC, AND WHY THE GAIN IS WHERE IT IS (04_margin.py, 05_local_margin.py)
A structural loss is a set of weights on the (tree distance, spatial distance) histogram, so its
usefulness is its ability to score the true assignment below corrupted ones -- measurable with no
OT solve at all.  Against a GLOBAL shuffle every loss behaves identically (square loss is even a
hair steeper).  Against LOCAL corruption -- moving each cell to a random spot within a radius,
which is the regime a 360 um error lives in -- the learned loss rises 40-65% faster:
  slope per 100 um over the first 700 um   square loss   learned rank 2
  M3-4                                        0.088          0.145
  M3-1                                        0.097          0.152
  M3-2                                        0.104          0.147
That is the mechanism: the square loss can tell a map from noise as well as anything, and is
markedly worse at fine placement.

ADDENDUM (same night): THE FAIREST COMPARISON IS "NO TUNING AT ALL"
--------------------------------------------------------------------
alpha cannot be tuned on unlabelled data, so best-vs-best flatters both arms equally but answers
the wrong question.  Summarising the whole weight grid under the split protocol (4 splits per
section, mean over splits of the per-split statistic):

  section          best        median       worst
  M3-4  FGW         364          437          513
        learned     342          381          453
  M3-1  FGW         438          512          612
        learned     431          485          559
  M3-2  FGW         442          509          584
        learned     419          464          550

  learned better on the BEST over the grid:    11/12 splits
  learned better on the MEDIAN over the grid:  12/12 splits

The median-over-grid gain (-56 / -27 / -45 um) is roughly twice the best-vs-best gain, because
FGW degrades sharply once the tree is given more weight than its narrow optimum and the learned
loss does not.  Quote the median-over-grid comparison as the headline; it is both the larger
effect and the honest one.

Expression-only measured on the SAME half-section subsets: 554.6 / 567.2 / 590.5 um (the
full-section values 549.7 / 566.9 / 600.9 are not comparable to half-section arms).

THE AUTOMATIC PIPELINE, HONESTLY
  KL selector on FGW:            364.4 / 441.5 / 447.1   (oracle 364.4 / 437.8 / 442.4)
  KL selector on learned:        361.0 / 475.2 / 433.4   (oracle 342.1 / 430.5 / 419.3)
  learned at FGW's selected w:   367.9 / 466.7 / 473.3   <- does NOT work; the learned loss wants
                                 2-5x more tree weight, so FGW's weight cannot be transplanted
  debiased selector (only the canonical components the loss does not use): helps on M3-4
  (+7.5 vs +18.9) and hurts elsewhere.  Plain joint KL is the better of the two.
So: the selector is essentially exact for FGW and unreliable for the learned loss.  End to end
and fully automatic the two arms are a wash; the learned loss's advantage is established on the
same footing on which alpha is always reported (a swept weight), plus the median-over-grid
statistic above, which needs no tuning at all.

ADDENDUM 2 (same night): 30 SPLITS, AND AN INDEPENDENT TESTBED
----------------------------------------------------------------
Extended the split protocol to 5 seeds x 2 halves x 3 sections = 30 splits (09_split_estimation.py
plus .more).  Expression-only on the same subsets: 555.7 / 570.9 / 593.3 um.

  section          best over grid   median over grid   worst over grid
  M3-4  FGW             363.2            448.5              517.4
        learned         347.0            385.5              443.5
  M3-1  FGW             441.7            511.5              617.8
        learned         427.9            481.4              593.2
  M3-2  FGW             432.8            498.2              587.5
        learned         417.6            464.6              549.9

  learned better on the BEST over the grid:    26/30 splits
  learned better on the MEDIAN over the grid:  29/30 splits

SIMULATION (15_simulation.py, sim_eden.py at dim=3, n=1200, m=128, sigma_expr=6.0).  The Eden
generator was calibrated to PEtracer's signature before any of tonight's work, so this is an
independent process, not a restatement.  Its concordance is weaker than the real data's
(corr 0.171, maximal 0.190, 50% of pairs at the ceiling), so the tree buys less overall.

  seed   expression   FGW best / median   learned best / median   shuffled   learned FROM ANOTHER
                                                                              TUMOUR best / median
   0        4.7         4.1 / 5.0            3.9 / 4.4              4.7          3.9 / 4.3
   1        4.6         4.1 / 5.9            4.1 / 4.4              4.5          4.1 / 4.4
   2        4.7         4.2 / 5.0            4.0 / 4.3              4.5          4.1 / 4.4

Same pattern as the real data: a small gain on the best-over-grid, a clear gain (-12%) on the
median-over-grid, and the shuffled control sitting at the expression-only level.

THE IMPORTANT LINE IS THE LAST COLUMN.  Transforms fitted on a DIFFERENT simulated tumour work as
well as transforms fitted on the same one.  So transfer is not intrinsically impossible -- when
the generating process is identical, the relationship transfers perfectly.  The failure of
cross-section transfer on PEtracer is therefore a statement about how much the three sections
differ (49% / 28% / 30% of pairs at the ceiling; fitted phi moving 2.03x further between sections
than between halves), not about the method.  That changes the recommendation from "this cannot be
transferred" to "match or pool references by composition, and expect transfer to work between
comparable sections".

FILES (experiments/night/)
  00_affine_invariance.py     Fact 0, numerically
  01_relationship.py          correspondence analysis of (tree, space); relationship.csv
  02_learned_loss.py          oracle sweep, all ranks + ablations; learned_loss.csv
  03_transfer.py              cross-SECTION transfer + pooled atlas; transfer.csv
  04_margin.py / 05_local_margin.py   loss discrimination without solving OT
  06_perclade_relationship.py per-clade concordance and its size-matched null
  07_final_maps.py            couplings for the figures
  08_cross_section.py         cross-section application sweep
  09_split_estimation.py(.more)  THE MAIN PROTOCOL
  10_controls.py              shuffled-pairing control + transform-stability diagnosis
  11_clade_vs_time.py         AUC(clade | tree distance), clade-statistic reproducibility
  12_family.py                power-transform family control
  13_application.py           application with null and ceiling; application.pkl
  14_select_w.py              label-free weight selection
  15_simulation.py            independent Eden testbed incl. cross-tumour transfer
  plot_*.py -> relationship.png components.png result.png maps.png application.png margin_fig.png

ADDENDUM 3 (same night): CROSS-SECTION TRANSFER DOES WORK -- ON THE UNTUNED STATISTIC
--------------------------------------------------------------------------------------
16_copula_transfer.py.  The earlier "transfer is a wash" verdict was judged on the BEST over the
weight grid, which needs an oracle to pick the weight in the first place.  On the median over the
grid -- the statistic that applies when there is nothing to tune on -- the picture inverts.

  target  source     median over grid            best over grid
                   FGW   plain   raked        FGW   plain   raked
  M3-4    M3-1    434.4  420.6   399.7       355.9  370.6   374.1
  M3-4    M3-2    434.4  396.6   395.9       355.9  375.2   368.3
  M3-1    M3-2    496.8  492.1   509.3       410.7  458.9   458.5
  M3-1    M3-4    496.8  470.8   466.4       410.7  438.6   453.3
  M3-2    M3-1    490.3  462.0   462.1       422.4  414.7   380.0
  M3-2    M3-4    490.3  472.5   475.3       422.4  422.4   406.8

  transfer beats FGW: 6/6 ordered pairs on the median, 2/6 on the best.
  mean over targets: FGW 473.9, plain transfer 452.4, raked 451.4 um.

READING.  The transferred transform is a slightly wrong version of the right shape.  It still
fixes the square loss's main defect -- over-trusting the 49% of pairs with no resolvable
divergence time, and collapsing once the tree is given more weight than its narrow optimum -- but
it loses the fine tuning a locally fitted transform has.  Both statistics belong in the paper.

RAKED = transfer only the DEPENDENCE and re-fit the marginals.  Both marginals are observable on
the target with no knowledge of where any query cell belongs: the query cells' own tree gives the
tree-distance marginal, the spot coordinates give the spatial one.  Iterative proportional fitting
moves the source's joint onto them, preserving its odds ratios -- "the copula transfers, the
marginals do not".  Worth about a wash on average (451.4 vs 452.4) but it produces the single best
transfer in the table (M3-2 from M3-1, 422.4 -> 380.0).  Something better than IPF is probably
available; this is the obvious place to look next.

SUPERSEDES the "the transform must be estimated in the tissue being mapped" statement in the
first entry of tonight's block.  The accurate statement is: the shape transfers between sections,
the tuning does not.

ADDENDUM 4: SHRINKAGE DOES NOT RESCUE THE TUNED TRANSFER GAIN (17_shrinkage.py)
--------------------------------------------------------------------------------
If a transferred transform is a slightly wrong version of the right shape, mix it with the square
loss and look for an interior optimum:
    Struct_lam = (1-lam) GW(z(C_tree), z(C_space)) + lam * sum_r rho_r GW(phi_r, psi_r)
lam=0 is FGW exactly, lam=1 is full transfer.  Six ordered pairs, lam in {0, .25, .5, .75, 1}:

  no consistent interior optimum.  On the MEDIAN over the weight grid lam=1 is best or near-best
  in 5/6 pairs (e.g. M3-1->M3-4: 434.4 at lam=0 -> 394.7 at lam=1).  On the BEST over the grid
  lam=0 wins whenever the target is M3-4 or M3-1.  The curve is close to monotone in both
  regimes, in opposite directions -- so this is not a bias-variance trade-off with a sweet spot,
  it is the same two-regime picture as Addendum 3.  Do not report shrinkage as a fix.

ALPHA = 1 ON THE FAIR DATA (the framing, verified on the corrected expression cost)
  M3-4  pure GW, expression term removed:  1087.9 um
        random assignment:                  754.8 um
        expression only:                    533.1 um (nearest neighbour) / 549.7 (OT)
Pure structure alignment is WORSE THAN RANDOM, because a Gromov-Wasserstein match between two
metric spaces is defined only up to their symmetries.  This is the cleanest single number for
the framing: expression is the necessary bridge and the tree is the necessary constraint, and
neither works alone.  (The earlier soft-accuracy version of this claim, 0.030 against a 0.098
floor, was computed on the pre-fix expression cost; the placement-error version above supersedes
it and says the same thing.)

DATA NOTE: koblan25 "tumor" contains four sections M3-1..M3-4, but M3-3 has ZERO cells with a
clade label (2890 / 2042 / 0 / 4562).  Three traced sections is all the dataset has.

ADDENDUM 5: SENSITIVITY TO THE ONE HYPERPARAMETER (18_binsens.py)
-------------------------------------------------------------------
The estimator has essentially one knob: how many bins the lookup tables use.  Split protocol,
seed 0, both halves, tree bins swept (spatial bins scale with it):

  section          FGW      9 bins   13      17*     25      33
  M3-4   best     366.9     364.3   357.8   351.4   341.5   338.0
         median   423.9     434.4   390.9   392.1   401.4   393.3
  M3-1   best     431.2     457.4   456.5   436.1   432.1   426.1
         median   510.3     497.9   483.9   480.4   479.4   477.5
  M3-2   best     434.5     429.7   434.9   407.7   413.4   412.0
         median   490.5     471.9   469.3   463.0   460.7   467.3
  (* the setting used everywhere else)

Beats FGW on the median at 13 bins and above in all three sections, and on the best at 17 and
above.  Nine bins is too coarse -- it loses on M3-1 and M3-2 -- which makes sense: with 49% of
pairs at the ceiling, a coarse grid cannot separate the short-divergence range at all.  Nothing
degrades up to 33 bins, so 17 is conservative rather than tuned; there is no overfitting cliff in
the tested range.  Report the sweep, not a single setting.

ADDENDUM 6: THE WEIGHT SELECTOR, GOT WRONG TWICE BEFORE GETTING IT RIGHT
-------------------------------------------------------------------------
19_selector_check.py / 19b_selector_check.py.  Two successive artefacts in the same claim:

  1. GRID EDGE.  14_select_w.py swept w over CENTRE[sec] * MULT, i.e. a grid centred on each
     section's known FGW optimum.  On M3-4 the KL criterion is MONOTONE over that grid, so the
     reported "+0.0 um gap" was just its lower edge.  Widening to geomspace(0.002, 1.5, 16)
     exposes a real interior minimum -- at w = 0.0117 against an oracle of 0.044.
  2. SAMPLE SIZE.  With the wide grid and 2 splits the gaps read +0.0 / +0.0 / +27.7 um.  With 8
     splits per section per arm they are:
        M3-4  FGW  mean +38.7 (median +34.0, max +71.0), exact 0/8
        M3-4  lrn  mean +65.8 (median +62.8, max +86.7), exact 0/8
        M3-1  FGW  mean +11.7 (median  +3.7, max +42.0), exact 4/8
        M3-1  lrn  mean +38.5 (median +19.6, max +136 ), exact 3/8
        M3-2  FGW  mean +14.3 (median +10.4, max +45.8), exact 1/8
        M3-2  lrn  mean  +2.8 (median  +0.0, max +12.9), exact 5/8
     The choice was interior in 48/48 splits, so the criterion is never degenerate.

THE RIGHT WAY TO QUOTE IT: fraction of the tree's benefit retained, (expr - selected) /
(expr - oracle):
     M3-4  FGW 80%   learned 69%
     M3-1  FGW 91%   learned 71%
     M3-2  FGW 91%   learned 98%
against 0% for every previously-tried criterion (deconvolution NLL, held-out genes, spot
coherence), each of which chose alpha = 0.  That is a real positive; it is not tight enough to
decide learned-vs-FGW, which is why the headline stays on the swept-weight and median-over-grid
statistics.

RULE REINFORCED (this is the fourth time in the project): before quoting a selection result,
(a) check the criterion's optimum is interior to the grid, and (b) get more than two splits.

ADDENDUM 7: THE SPOT MARGINAL IS THE OTHER LEVER, AND IT STACKS (20_marginals.py, 20b)
----------------------------------------------------------------------------------------
Every result above assumes each spot receives the same number of the cells being mapped.  That is
an assumption about the QUERY SET, not about the tissue, and it is false: real occupancy is 2-15
cells per spot, CV about 0.5.  Setting q_j proportional to the spot's own cell count is
non-circular -- it is a property of the spatial data (nuclei counts), and the query cells' halves
are disjoint from the spot-side halves that define it.

30 splits per cell (5 seeds x 2 halves x 3 sections), median placement error in um:

  section   loss      marginal    best over grid   median over grid
  M3-4      FGW       uniform         363.2             435.7
  M3-4      FGW       observed        338.7             396.2
  M3-4      learned   uniform         347.0             380.7
  M3-4      learned   observed        307.2             335.0
  M3-1      FGW       uniform         441.7             503.5
  M3-1      FGW       observed        431.4             487.7
  M3-1      learned   uniform         427.9             473.8
  M3-1      learned   observed        405.1             448.0
  M3-2      FGW       uniform         432.8             485.7
  M3-2      FGW       observed        424.2             484.7
  M3-2      learned   uniform         417.6             467.0
  M3-2      learned   observed        412.9             459.0

PAIRED vs FGW + uniform:  FGW+observed 26/30 (best) 24/30 (median);  learned+uniform 26/30, 28/30;
learned+observed 28/30 (best), 30/30 (median).  THE TWO LEVERS STACK and they are independent --
one changes what the tree means, the other what the marginals assert.

EXPRESSION-ONLY OT for reference:  uniform 555.9 / 571.6 / 593.8;  observed 529.6 / 561.2 / 595.5.
THE RECOMMENDED RECIPE (learned loss + observed marginal) against expression-only with uniform
marginals:  45% / 29% / 30% reduction, against FGW's 35% / 23% / 27%.

SEMI-RELAXED (no spot marginal at all; solve_semirelaxed added to genfgw.py -- the linear step is
a row-wise argmin instead of an EMD, so it is cheaper than the balanced solve) reaches a good BEST
(335.0 / 427.0 / 420.6 for FGW; 321.1 / 403.4 / 391.2 learned) but its MEDIAN over the weight grid
blows up (learned on M3-4: 928.7 um) because with the spot marginal gone the mapping can collapse
onto a few spots at high tree weight.  Good ceiling, bad robustness -- do not recommend it without
a density penalty.

ADDENDUM 8: THE APPLICATION, RERUN UNDER THE RECOMMENDED CONFIGURATION
-----------------------------------------------------------------------
13_application.py with the observed-density spot marginal (was uniform).  Clade-territory
agreement between one section's cells placed into another section and that section's own cells:

  pair            null (labels permuted)   expression   FGW    learned   within-section ceiling
  M3-1 -> M3-4      +0.015 +- 0.017          0.038      0.095   0.075          0.326
  M3-2 -> M3-1      +0.008 +- 0.019          0.101      0.186   0.215          0.221
  M3-4 -> M3-2      +0.022 +- 0.016          0.056      0.088   0.096          0.246
  mean               0.015                   0.065      0.123   0.129          0.264

IMPORTANT: with an observed-density marginal the NULL IS NO LONGER ~0 (0.015 vs 0.001 under a
uniform marginal), because putting more cells where there are more cells already reproduces part
of any clade's density.  Subtract it before quoting fractions: the tree recovers 43-46% of the
ceiling, expression alone 20%.  (Under uniform marginals the same computation gave 59% and 22%,
with a null of 0.001 -- the two are not comparable and the null-corrected version is the honest
one.)

FGW and the learned loss are close on this metric (0.123 vs 0.129).  That is expected: clade
territory is coarse structure, which both losses already capture; the learned loss's advantage is
in fine placement, which this metric does not measure.

================================================================================
2026-09-01  A SECOND DATASET, AND A PRE-FLIGHT CHECK  (21_etracer_prep.py, 22_etracer_map.py)
================================================================================
eTRACER asks less of the reference than PEtracer does.  No cell has a position; what the spatial
assay gives is a CLONE LABEL PER BIN.  The relationship is still estimable: for a pair of spots
with clone compositions (comp_j, comp_l), draw a clone from each, then draw a patristic distance
from the empirical distribution for that clone pair, which the dissociated cells supply.  That
turns spot pairs into (tree distance, spatial distance) draws and everything downstream is
unchanged.  Bins aggregated into 512 k-d spots (purity 0.64/0.67), reference = a random half of
the spots, scoring on the other half by per-clone density agreement, 3 seeds.

  dataset   permuted   expression only   FGW      learned
  LE         -0.011        +0.453       +0.574    +0.575
  EE         -0.007        +0.300       +0.470    +0.472

CLAIM (a) REPLICATES ON A SECOND DATASET, and strongly: +27% (LE) and +57% (EE) over expression
alone, against a permutation null of zero.  With a reference that only knows which clone is where.

THE LEARNED LOSS ADDS NOTHING HERE -- and that is a prediction, not a disappointment.  The
canonical spectrum of the sampled joint says why:

                     rho_1    rho_2    rho_3     plain corr
  PEtracer M3-4      0.530    0.230    0.047       +0.410
  PEtracer M3-2      0.379    0.171    0.042       +0.323
  PEtracer M3-1      0.306    0.146    0.032       +0.266
  eTRACER LE         0.269    0.007    0.003       +0.261
  eTRACER EE         0.184    0.024    0.005       +0.182

With a clone-level reference rho_1 EQUALS the plain correlation (0.269 vs 0.261; 0.184 vs 0.182),
so there is nothing for a nonlinear transform to gain, and rho_2 is essentially zero, so there is
no second component to add.  The learned loss must reduce to FGW, and it does, to within 0.002.
The short-range structure that component 2 encodes lives WITHIN clones, and a clone-level
reference averages it away.

CONSEQUENCE, AND IT IS THE CONSTRUCTIVE ONE: rho_2 of the reference is a PRE-FLIGHT CHECK.  It
costs one 2-D histogram and one SVD, needs no mapping, and says whether a learned structural loss
can help at all.  rho_2 <= 0.03 -> use plain FGW.  rho_2 >= 0.15 -> the learned loss is worth it
(3-4% better placement on PEtracer).  The simulation sits in between (rho_2 = 0.056, 3.2% gain),
so the threshold is not sharp and should be quoted as a screen, not a law.

This also sharpens the recommendation about references: a clone-resolved in-situ readout is enough
to USE the tree, but you need a CELL-RESOLVED matched reference to learn how to read it better.

CONTROLLED TEST OF THE RESOLUTION CLAIM (23_reference_resolution.py), AND A CORRECTION
---------------------------------------------------------------------------------------
I explained eTRACER's rho_2 = 0.007 as "clone-level averaging destroys the within-clone
structure".  Tested directly on PEtracer by degrading its reference to exactly eTRACER's
information -- real spatial distances between reference cells, but each pair's tree distance
replaced by a DRAW from its clade pair's empirical distribution (both marginals preserved, the
pair-level link destroyed):

                     cell reference          clone reference
  M3-4   corr +0.393  rho [0.494 0.209]   corr +0.356  rho [0.484 0.145]
  M3-1   corr +0.276  rho [0.310 0.147]   corr +0.253  rho [0.296 0.115]
  M3-2   corr +0.348  rho [0.415 0.173]   corr +0.331  rho [0.410 0.144]

Clade aggregation costs about 25-30% of rho_2 -- REAL BUT NOT A COLLAPSE.  So it does NOT by
itself explain eTRACER's 0.007/0.024.  CORRECTED STATEMENT: eTRACER's reference has essentially no
second component, and only part of that is the clone-level aggregation; the rest is that its tree
carries less short-range spatial information than PEtracer's.  The two cannot be separated on
eTRACER, which has no cell positions at all.

THE MAPPING SIDE OF THE SAME CONTROL IS CLEANER (observed marginals, 4 splits per cell):
  section   FGW              learned, CELL ref     learned, CLONE ref
            best / median    best / median         best / median
  M3-4      342.3 / 399.0    313.8 / 351.9         344.6 / 390.2
  M3-1      435.1 / 488.9    436.9 / 458.1         437.0 / 464.9
  M3-2      429.3 / 487.8    413.1 / 463.0         430.6 / 462.8

Degrading the reference to clone resolution ERASES the tuned gain (M3-4 313.8 -> 344.6, i.e. back
to FGW's 342.3; M3-2 413.1 -> 430.6 against FGW's 429.3) while leaving most of the
robustness gain on the median.  That is the controlled version of the eTRACER observation and it
is the statement to make: A CELL-RESOLVED MATCHED REFERENCE IS WHAT BUYS THE TUNED IMPROVEMENT;
a clone-resolved one still buys robustness to the weight.

FORMAL TESTS ON THE 30 SPLITS
------------------------------
Paired over splits (each split contributes one number per arm):

  comparison                                     statistic   won    mean gain   sign      Wilcoxon
  learned vs FGW, uniform marginals              best       26/30    +15.0 um   3.0e-05   2.0e-05
  learned vs FGW, uniform marginals              median     29/30    +42.2 um   2.9e-08   1.9e-09
  learned+observed vs FGW+uniform                best       28/30    +37.5 um   4.3e-07   3.1e-08
  learned+observed vs FGW+uniform                median     30/30    +61.0 um   9.3e-10   9.3e-10

CAVEAT to state in the paper: splits within a section reuse the same cells, so the p-values are
optimistic.  The stronger evidence is that the effect is consistent across all three sections
independently, plus the simulation and the destroyed-relationship control.

================================================================================
2026-09-01  BCSA: THE PIPELINE FITS, BUT THE PRE-FLIGHT CHECK SAYS NO
================================================================================
BCSA is structurally the eTRACER setting, and the inputs are already in the right shape:
  bcsa3_{D1,B1}_inputs.npz -- 444 scRNA-seq cells with a CNV clone tree (C_tree 444x444, 28
  distinct distances, within-clone 0.134 vs between 0.279), 2298-2643 Visium spots with
  coordinates and `spot_soft` (2643 x 12), which IS the clone-composition matrix the eTRACER
  construction takes.  No cell has a position, so only the clone-resolved route is available.

INDEX TRAP: `clones` = [1..12] and `cell_clone` / `spot_clone_numbat` use those ids, but
`spot_soft` has 12 columns indexed 0..11.  Column j corresponds to clones[j].  Getting this wrong
gives the same near-zero answer for the wrong reason -- I did it first time.

PRE-FLIGHT CHECK (correctly indexed, 1.5M sampled spot pairs):
    BCSA3 D1   corr(tree, space) = +0.002   rho = [0.023 0.012 0.007]
    BCSA3 B1   corr(tree, space) = -0.004   rho = [0.011 0.008 0.007]
  against  eTRACER LE/EE  rho_1 = 0.269 / 0.184     PEtracer  rho_1 = 0.31-0.53

WHY, and it is not about the method: same-clone spots sit at 0.956x (D1) and 0.975x (B1) the
distance of different-clone spots.  PEtracer clades sit at 0.59-0.66x.  The Numbat clone calls
have essentially NO spatial territories, and no structural term can use a tree whose groups do not
occupy space.

THIS REPRODUCES WHAT THE REPO ALREADY RECORDED, by a different route.
experiments/bcsa3/dataset_requirements.csv:
    dataset        clones/usable   coherence   r_pairs   r_mantel   p_mantel
    BCSA2/TumC2       4 / 2          0.239      0.115     0.376      0.44
    BCSA3/B1         12 / 8          0.051      0.025     0.278      0.11
    BCSA3/D1         12 / 8          0.096      0.015    -0.054      0.77
    BCSA3/C1          8 / 6          0.082      0.039     0.505      0.24
    BCSA6/TumA1      19 / 10         0.029      0.031    -0.010      0.94
  No Mantel correlation is significant on any sample.  (The tiny r_pairs p-values are an artefact
  of millions of spot pairs; read the effect sizes.)

WHAT TO DO WITH BCSA
  1. The clone calls are the bottleneck, not the mapping.  The check is on the CALLS -- a real
     clonal geography could still be there behind noisy Numbat posteriors.  Deeper allele
     coverage / allele-only posteriors is the lever (the allele_premise_* scripts already go this
     way).
  2. If any sample is worth a run it is BCSA2/TumC2 (coherence 0.239, r_mantel 0.376) or BCSA3/C1
     (r_mantel 0.505) -- both underpowered on usable clones (2 and 6).
  3. Meanwhile the OBSERVED-DENSITY SPOT MARGINAL needs no tree at all and is worth ~20 um on
     PEtracer.  It is the one piece of tonight's work that applies to BCSA today.
  4. Report rho_1 / rho_2 of the reference as a standard line in every dataset table.  It costs
     one histogram and one SVD and it is the same quantity across PEtracer, eTRACER and BCSA.

2026-09-01  BCSA3 RUN ON THE PROJECT'S OWN BENCHMARK (24_bcsa_run.py)
----------------------------------------------------------------------
Ran the full pipeline on BCSA3 D1/B1 and scored with experiments/bcsa3/evaluate.py -- the same
spot-deconvolution metric as the cached DestVI / scDOT baselines.  Transform fitted on a random
half of the spots' clone compositions, every arm scored on the other half, 3 reference draws.

TWO BUGS OF MINE, FOUND AND FIXED, WHICH CHANGED NOTHING
  1. One D1 spot has a library size of 8.7e7 against a median of 5.3e3.  It sets M.max() and
     crushes the expression cost relative to the structural term.  Dropped.
  2. Balanced OT FORCES the predicted composition over spots to equal the composition of the
     reference cells.  Those differ by TVD 0.55 on D1 and 0.68 on B1 (clone 2 is 38.5% of the
     cells and 3.2% of the spots; clone 3 has 19 cells and ZERO spots), so a balanced map is wrong
     by construction before the cost is consulted.  Added solve_semirelaxed(..., side="spots"):
     spot marginal fixed, cell mixture free.  That is the right deconvolution formulation and it
     is now in the library.
Neither fix moved the result.

RESULT (accuracy / ARI / mean per-clone Pearson, held-out spots)
  BCSA3 D1  majority clone 0.285
            expression only  balanced 0.105 / 0.027 / 0.032   free-mixture 0.093 / 0.020 / 0.033
            FGW              balanced 0.090 / 0.010 / 0.021   free-mixture 0.086 / 0.007 / 0.023
            learned          balanced 0.105 / 0.023 / 0.034   free-mixture 0.086 / 0.015 / 0.035
  BCSA3 B1  majority clone 0.279
            expression only  balanced 0.075 / 0.027 / 0.022   free-mixture 0.039 / 0.025 / 0.026
            FGW              balanced 0.072 / 0.016 / 0.023   free-mixture 0.040 / 0.029 / 0.023
            learned          balanced 0.073 / 0.025 / 0.025   free-mixture 0.038 / 0.025 / 0.021

THE DECISIVE CONTROL: THE PUBLISHED BASELINES ARE ALSO AT CHANCE (all spots, their own footing)
  BCSA3 D1  majority 0.286   DestVI 0.061 / -0.004 / -0.006   scDOT 0.024 / 0.017 / 0.006
  BCSA3 B1  majority 0.280   DestVI 0.242 /  0.032 /  0.012   scDOT 0.262 / 0.008 / 0.017
(Baseline scoring verified: cached CSVs reindex correctly by spot barcode and clone name, rowsums
1.0, DestVI's alphabetical column order handled.)

NO METHOD BEATS "PREDICT THE MOST COMMON CLONE EVERYWHERE" on this benchmark -- not ours, not
DestVI, not scDOT.  The task as posed on BCSA3 is not solvable from these inputs, and that is a
statement about the Numbat clone calls, not about any of the methods.  Three independent lines
agree: the tree-space premise fails (rho_1 = 0.02, same-clone spots at 0.96x the distance of
different-clone spots), the reference is not a representative sample of the tissue (TVD 0.55),
and every method sits at chance.  The repo already recorded the first two in
experiments/bcsa3/dataset_requirements.csv.

DO NOT run more BCSA mapping variants until the clone calls improve.  The lever is upstream:
allele-only / deeper-coverage posteriors (the allele_premise_* scripts), or a sample with real
clonal geography.

2026-09-01  SLIDE DECK (7 slides) -- experiments/night/slide_*.png, deck artifact
  https://claude.ai/code/artifact/5aa50175-089d-4429-8b7a-cee6a263b515
  1 premise (slide_premise.png)  2-3 methodology (slide_diagnosis.png, components.png)
  4 placement (maps.png)  5 unmatched (application.png)  6 per-clade negative (slide_clade.png)
  7 BCSA negative (slide_negative.png)
  Territoriality ratio, same-group / different-group mean pair distance, verified:
  PEtracer 0.56x, eTRACER 0.60x, BCSA3 D1 0.96x.

================================================================================
2026-09-01  THE CLONE-SPREAD COST: AN INTERPRETABLE REPLACEMENT FOR GW
src/spotr/clonespread.py, tests/test_clonespread.py, experiments/night/32-38_*.py
================================================================================
Pedro's redirect: do not fit an uninterpretable transform per tumour.  Use PEtracer as ground
truth to DERIVE a cost, with parameters that carry units and transfer, and that also yields
motility from a single slice.

WHAT THE DATA SAYS (33_look_at_law.py, law.png)
  The conditional law of separation given divergence age changes SHAPE with age: a tight spike
  near zero for recent divergence, and at the ceiling simply the distance distribution between two
  random points in the tissue.  So the precise failure of GW is not "far in the tree should be far
  in space is violated" -- the MEAN relation is monotone -- it is that for 28-49% of pairs the tree
  carries no positional information and GW still hands them a numeric distance target.
  THE HARD CEILING: the tree explains 14.1% / 4.5% / 5.4% of the variance in pairwise spatial
  distance (M3-4 / M3-1 / M3-2).  Nothing built on the tree can exceed that.
  Free diffusion (E[s^2] linear in age) fits poorly, R2 0.26-0.37; sigma(age) is CONVEX in age and
  in sqrt(age), i.e. steeper than Brownian -- growth, not diffusion.

THE MODEL
  Two cells whose LCA lived a days ago started in the same place.  The tree predicts a SCALE:
      r(a) = spread of a clone whose founding cell lived a days ago      [microns]
      -log P(d | a) = d^2/(2 r(a)^2) - log d + 2 log r(a)
  Under FIXED MARGINALS the last two terms are constants on the polytope (2 log r depends only on
  the cells, -log d only on the spots), so the whole NLL reduces to
      S(T) = 0.5 sum_ijkl W_ik d_jl^2 T_ij T_kl,        W_ik = 1 / r(a_ik)^2
  an inverse-variance-weighted attraction.  Expanding d^2 leaves only n^2 and nm work -- NO n^2 m
  tensor.  This is the Laplacian/graph-regularised form flagged as UNTESTED in the 2026-08-28
  entry, now DERIVED rather than proposed.

VERIFIED (tests/test_clonespread.py): analytic gradient vs finite differences 1.8e-10; the
objective is exactly the quadratic the line search assumes (3.4e-16, after fixing a spurious
factor of 1/2 in a2); monotone descent; fit_spread recovers a planted r(a) to 1 um; alpha=0
reproduces ot.emd exactly.

HEAD TO HEAD (36_headtohead.py, 30 splits, observed spot marginals, one common wide w grid)
  section   expression   FGW     clone spread   learned rank 2
  M3-4        529.6      342.9      342.3           314.5
  M3-1        561.2      439.6      407.4           416.2
  M3-2        595.5      433.4      412.0           407.1
  mean        562.1      405.3      387.2           379.3

  PAIRED:  clone spread beats FGW           22/30 splits, +18.1 um, sign p = 8.1e-03
           learned rank 2 beats FGW         24/30 splits, +26.0 um, sign p = 7.2e-04
           learned rank 2 vs clone spread   17/30 splits,  +7.9 um, sign p = 0.29  -- A TIE
  So the interpretable one-function cost is statistically indistinguishable from the fitted
  two-lookup-table version, and both beat GW.  INTERPRETABILITY IS FREE HERE.
  It is also 2-2.5x faster per solve (M3-4: 1.4 s vs 2.9 s for FGW, 3.4 s for the learned loss).

MOTILITY FROM A SINGLE SLICE (37_motility.py, motility.csv)
  m_tau = RMS observed within-clade separation / RMS separation r(a) predicts.  Dimensionless,
  1.0 = exactly as dispersed as the lineage says.
  1. REAL VARIATION: sd 0.233 / 0.240 / 0.181 against a size-matched random-grouping null of
     0.038 / 0.046 / 0.050 -- 6.1x / 5.2x / 3.6x the noise.
  2. ESTIMABLE: reproduces between halves of one section, pearson +0.844 (n=135 clade-halves).
  3. ACROSS SECTIONS, MIXED: M3-1 vs M3-2 +0.684, and it IMPROVES with clade size
     (+0.666 at n>=25, +0.701 at n>=40, +0.748 at n>=60, +0.817 at n>=80) which is what a real
     quantity measured with noise should do.  M3-4 vs M3-1 is +0.064 overall, +0.513 once clades
     with <40 cells in either section are dropped, then unstable at higher thresholds (n<=5).
     M3-4 is again the odd section out.
  ROBUST ACROSS ALL THREE: clade 8 is among the least dispersed everywhere (0.44/0.75/0.59);
  clade 14 is among the most (1.48/1.16/1.36).

HONEST LIMITS
  - The dimensionless collapse of s/r(a) across tumours (34_collapse.py) is real but the test
    discriminates weakly: KS 0.010-0.016 between six real and simulated datasets, against a
    split-half noise floor of 0.008 and 0.028-0.030 for a Rayleigh or two-uniform-points law with
    the same second moment.  Only the half-normal (0.182) is clearly excluded.  So "the residual
    is close to Rayleigh" is a defensible modelling choice, not a measured universal.
  - Motility across sections is established for M3-1/M3-2 and not for M3-4.

2026-09-01  DEPTH STRATIFICATION: PROMISING, CONFOUNDED, NOT YET A RESULT (44_boundary.py)
--------------------------------------------------------------------------------------------
obs carries much more than we were using: tumor_boundary_dist (40-696 um), fitness (0.7-6.2),
hotspot_module, cell_subtype (all traced cells are Malignant / within_tumor), centroid_z (+-12 um,
too thin for 3D).

Splitting cell pairs into thirds by depth in the tumour, corr(divergence age, separation):
              edge    middle   core        (tree-shuffle-within-stratum null ~0.00 everywhere)
  mean       0.409    0.363   0.285        monotone edge > middle > core in 3/3 sections
  fitness     2.81     2.48    2.26        the edge is also the fitter tissue

CONTROLS.  Mean divergence age is IDENTICAL across strata (32.4-32.8 d), so this is not an age
composition artefact.  BUT the geometry is very different -- the outer third is an annulus spanning
the full tumour diameter (extent 2169 um, mean separation 1034 um) while the core is a compact disc
(1540 um, 529 um) -- and the ceiling fraction differs (0.40 edge vs 0.25 core).  Restricting to
RESOLVED pairs only the ordering survives but weakens and loses its cleanliness:
              edge 0.277   middle 0.236   core 0.193   (M3-4 has middle == core)

VERDICT: the annulus-vs-disc confound is not removed by the shuffle null, which controls
significance but not the cross-stratum magnitude comparison.  A geometry-matched design is needed
-- equal-size, equal-shape local neighbourhoods sampled at different depths, rather than concentric
strata.  DO NOT quote the edge/core numbers until that is run.

2026-09-01  THE DEPTH EFFECT IS GEOMETRY, AND THE MOTILITY INDEX IS MOSTLY CLADE SPREAD
  45_geometry_matched.py
------------------------------------------------------------------------------------------
GEOMETRY-MATCHED DESIGN: discs of fixed radius 260 um centred on each cell, subsampled to EXACTLY
25 cells, so shape and count are matched by construction.  4744 neighbourhoods; only 0.1% dropped
for having too few cells, and the drop rate is flat with depth, so no selection effect.

RESULT 1 -- the concentric-strata finding was entirely the annulus-vs-disc confound.
  corr(divergence age, separation):  edge 0.159   middle 0.171   core 0.162    -- flat
  per section the ordering is inconsistent (M3-2 actually increases with depth), and the
  continuous spearman is +0.006 / -0.030 / +0.151.  The edge > middle > core result from
  44_boundary.py DOES NOT SURVIVE.  Do not use it.

RESULT 2 -- the motility index looked like it had a depth trend, and that is density.
  motility index: edge 0.260  middle 0.295  core 0.303, monotone in 3/3, spearman +0.38/+0.61/+0.56.
  BUT core neighbourhoods have 11% larger REALISED spread (rms 209 -> 256 um) because the core is
  less dense, and m scales directly with realised spread.  Stratifying by realised spread the
  trend vanishes:
      rms bin   edge  middle  core
      rms1     0.238  0.254  0.256
      rms3     0.292  0.293  0.290
      rms5     0.319  0.323  0.320
  and the within-bin spearmans are inconsistent in sign.  NO DEPTH EFFECT.

RESULT 3 -- AND THIS IS THE ONE THAT MATTERS FOR THE TOOL.  The per-clade motility index is
largely just raw clade dispersion, which needs no tree at all:
      corr(m, raw clade spread)   +0.790 / +0.845 / +0.757   (pooled +0.813, spearman +0.784)
      rank corr with raw spread   +0.750 / +0.680 / +0.773
      corr(m, predicted spread)   -0.452 / -0.387 / -0.206
      the PREDICTION varies across clades by only cv 0.13-0.15 (397-780 um)
  So ~65% of the motility index's variance is "how spread out is this clade".  The lineage
  correction is real and does reorder clades, but it is a secondary term, not the main signal.
  Combined with the earlier finding that the index does not reproduce between M3-4 and the other
  sections, THE MOTILITY READOUT IS NOT STRONG ENOUGH TO HEADLINE A TOOL.

CONSEQUENCE FOR FRAMING: the durable assets are the COST (interpretable, derived, beats GW 22/30,
ties the fitted alternative, 2-2.5x faster) and the SCREEN (territoriality ratio 0.56 / 0.60 /
0.96).  r(a) is a well-estimated descriptive curve and the object the cost is built from, but it
is not by itself a finding.  Position the paper on the method, not on a biological readout.

2026-09-01  A THIRD SYSTEM: C. ELEGANS (packer19), AND A BETTER SCREEN  (46_celegans.py)
-----------------------------------------------------------------------------------------
pycea ships two more datasets besides koblan25.  yang22 (KP-Tracer mouse lung adenocarcinoma,
1109 cells, tree + fitness + plasticity) has NO spatial coordinates -- not usable.  packer19 IS
usable: 988 C. elegans embryo cells, full transcriptome (20222 genes), 3D positions, and the
INVARIANT lineage, so the tree is exact and divergence times are known in minutes with no recorder
noise.  CAVEAT: the positions come from the embryonic atlas keyed on cell identity, not from
imaging the sequenced cells, and cells are sampled across 74-400 min.

RESULT: the premise largely fails there, for a different reason than BCSA.
  all pairs                     corr(age, sep) +0.113   r(a) range 1.27x
  pairs at the same timepoint   corr           +0.230   r(a) range 1.19x
  PEtracer                      corr +0.27 to +0.41     r(a) range 3.4x to 5.3x
Time-mixing was depressing the correlation (0.113 -> 0.230 once restricted to cells observed at the
same developmental time), so the correlation reaches PEtracer's low end.  But the DYNAMIC RANGE OF
r(a) STAYS FLAT AT ~1.2x whatever the restriction.  Cells that diverged 7 minutes ago and 257
minutes ago are almost equally far apart (14.1 vs 17.9 units in a 50x30x27 embryo).

WHY: the embryo is tiny, tightly packed, and full of programmed migration.  There is no room for
clones to occupy distinguishable territories, and lineage does not stay put by design.  BCSA fails
because the groups have no territory; C. elegans fails because the whole embryo is one territory.

CONSEQUENCE -- THE SCREEN SHOULD BE THE DYNAMIC RANGE OF r(a), NOT THE CORRELATION.
  r(a) max/min:   PEtracer 3.4-5.3x   |   C. elegans 1.2x
  correlation:    PEtracer 0.27-0.41  |   C. elegans 0.23  <- does not separate them
The dynamic range is exactly what the cost consumes (W = 1/r(a)^2 is flat if r(a) is flat), so it
is the right quantity, and it separates the two cases cleanly where the correlation does not.
SCOPE STATEMENT: the method needs a tissue large enough, and growth local enough, for clones to
occupy distinguishable territory.

2026-09-01  THE DATA LANDSCAPE: IT IS NOT JUST PETRACER ANY MORE
-----------------------------------------------------------------
Literature survey (web, 1 Sep 2026).  Datasets with ALL THREE of recorded lineage + position
MEASURED in situ + gene expression, in the same cells:

  dataset          venue                 tissue                    expression      data
  PEtracer         Science 2025          mouse 4T1 lung mets       MERFISH 124/175 GSE290975,
                   eadx3800              ~368,722 cells,           genes           figshare
                                         10 TUMOURS / 2 mice                       28473866
  Spatio-DARLIN    Nat Methods 2026      mouse small intestine     whole txome     GSE296896 +
                   23:1588               + brain, ~7000 clones     (BMKMANU S3000) GSE296897
                                         per intestinal section
  SPACE-seq        Cell Stem Cell 2026   mouse liver cancer,       whole txome     GSE327939
  (Camargo/Chen)   10.1016/j.stem        cortex, developing liver  (Slide-seq)
                   .2026.05.017
  SpaceBar         Nat Methods 2025      human melanoma in vitro,  seqFISH 120     github only,
  (Raj lab)        23:328                mouse xenograft, 67k      genes           raw "on request"
                                         cells in a tumour section
  intMEMOIR        Science 2021          Drosophila brain          in situ readout
                   eabb3099
  Spatial iTracer  Nat Methods 2022      human cerebral organoid   Visium
                   19:90
  baseMEMOIR       bioRxiv 2024, STILL   mESC colonies, 197 cells  smFISH 12 genes CaltechDATA
                   A PREPRINT                                                      pmpby-gpj05

CRITICAL AND IMMEDIATELY ACTIONABLE: PEtracer HAS TEN TUMOURS.  pycea.datasets.koblan25() ships
ONLY "mouse 3 tumor 1" (our M3), which is why we have been stuck on one tumour.  The rest are in
GSE290975 (69 samples, verified live) and figshare 28473866.  prep_petracer.py already does the
per-section build for M3 and should adapt.  THIS IS THE HIGHEST-VALUE NEXT STEP -- it turns
"one tumour" into a comparative r(a) study without needing a new assay.

IMPORTANT DISTINCTION for our method: SpaceBar uses STATIC barcodes (combinatorial clone labels,
no evolving recorder), so it gives clone membership but NO TREE and no divergence times.  It
supports the clone-level route (as eTRACER does) but cannot support r(a) directly.  The
tree-bearing ones are PEtracer, Spatio-DARLIN, SPACE-seq, intMEMOIR, iTracer, baseMEMOIR.

NEAR MISSES worth knowing: the Weissman mouse-embryogenesis PEtracer follow-up (bioRxiv 2026,
>1.5M cells) has NO MERFISH -- its spatial claims come from inferring A-P position from Hox
expression.  Nowakowski's lineage-resolved human cortex atlas (Nature 2025) is dissociated.
BARseq3 reads non-heritable projection barcodes.  Hypercascade (Elowitz 2026) has excellent deep
recording but NGS readout, no spatial -- positioned as the chemistry for a future spatial version.

CAVEATS: Spatio-DARLIN and SPACE-seq full texts are paywalled, so cell counts and platform details
are from GEO/README rather than the papers.  The survey exhausted its search budget, so the list
may be incomplete for the most recent weeks.

CORRECTION AND EXPANSION of the entry above, after two further independent surveys
------------------------------------------------------------------------------------
THE DISTINCTION THAT MATTERS FOR US: r(a) needs DIVERGENCE TIMES, i.e. an EVOLVING recorder that
yields a TREE.  Static lentiviral barcodes give clone membership only and support the clone-level
route (as eTRACER does) but not r(a).  Splitting the field on that line:

TREE-BEARING + measured spatial + expression (what r(a) needs)
  PEtracer        Science 2025      mouse 4T1 lung mets, ~368,722 MERFISH cells, TEN TUMOURS,
                                    124/175-gene panel.  figshare 28473866; GEO GSE290975.
                                    pycea ships ONLY M3 T1.
  Spatio-DARLIN   Nat Methods 2026  mouse intestine + brain, DARLIN evolving scars, whole
                                    transcriptome, BMKMANU S3000 array (3-99 um bins),
                                    ~25-50% barcode recovery, ~7000 intestinal clones.
                                    GSE296896 + GSE296897.
  SPACE-seq       Cell Stem Cell    mouse iCCA, developing cortex, embryonic liver.  DARLIN +
  (Camargo/Chen)  2026 33:1398      Slide-seqV2 10 um beads, whole transcriptome.
                                    GSE327939 (super) / GSE327687 + GSE326431.
  intMEMOIR       Science 2021      Drosophila brain -- only ~529 usable cells.
  Spatial iTracer Nat Methods 2022  human cerebral organoid, Visium, 2038 spots.
  baseMEMOIR      bioRxiv, STILL a  mESC colonies, 197 cells.  Too small.
                  preprint

CLONE-LABEL ONLY (no tree; clone-level route only)
  SpaceBar (Nat Methods 2026, human melanoma xenograft, seqFISH, ~38k barcoded cells, RAW DATA ON
  REQUEST), Steller (bioRxiv 2026, mouse forebrain, LARRY, >2000 cells), SPLINTR+Stereo-seq mouse
  spleen (Cell Rep Methods 2024, GSE246611 + Zenodo 10685805, deposited expressly for methods
  development), Space-TREX (Nat Neurosci 2022, only ~1079 spots with cloneIDs).

NOTE ON OUR OWN eTRACER DATA: on the SPATIAL arm eTRACER reads the STATIC barcode; the evolving
CRISPR scars were read only in the dissociated scRNA-seq arm.  That explains exactly why our
etracer_inputs_*.npz has a clone label per bin and no cell-level tree, and why the clone-resolved
route was the only one available there.  Not a preprocessing gap on our side.

WHAT DOES NOT QUALIFY: packer19 (dissociated, positions atlas-assigned -- confirms our own read),
the Weissman mouse-embryogenesis follow-up (no MERFISH; A-P position inferred from Hox), Slide-tags
and all mtDNA/CNV work (lineage INFERRED), scGESTALT / LINNAEUS / ScarTrace (no spatial arm),
Harkins 2025 (barcodes NOT captured in the beads; projected by RCTD -- the exact failure mode to
avoid).  NO Visium or Xenium paper with a RECORDED barcode in tissue exists yet; X-CODE (bioRxiv
2026) is the leading edge and its Xenium arm is stamped cells, not tissue.

DEMAND SIGNAL: LineageMap (bioRxiv 2026) states outright that public spatially resolved lineage
tracing datasets are not widely available and validates mostly on simulation, falling back on
baseMEMOIR's 197 cells.  Raphael's MOLLUSC did the same.  The methods are arriving ahead of the
data and the 2025-26 wave has just closed the gap.

CAVEATS: the three surveys disagreed on some author spellings and on whether PEtracer's GEO
accession could be confirmed (one verified GSE290975 live with 69 samples, another could not reach
the data-availability statement).  Spatio-DARLIN and SPACE-seq full texts are paywalled, so cell
counts come from GEO summaries.  All three surveys exhausted their web-search budgets, so the most
recent weeks may be under-covered.

FINAL CROSS-VERIFIED VERSION (four independent surveys, 1 Sep 2026)
---------------------------------------------------------------------
Additions and confirmations beyond the entry above:

  Spatio-DARLIN also BioProject PRJNA1261899; clone positions reproducible across adjacent 20 um
  sections (a useful property for us -- it means r(a) could be checked between serial sections).
  SPACE-seq also BioProject PRJNA1451866.
  intMEMOIR data: CaltechDATA 10.22002/D1.1444.  Spatial iTracer: ArrayExpress E-MTAB-10972.
  SPLINTR + Stereo-seq mouse spleen (Cell Rep Methods 2024): GSE246611 + Zenodo 10685805,
  deposited EXPRESSLY for methods development on clonally resolved spatial data.  Static barcodes,
  10 clones, limited sampling -- but the most frictionless public download on the list.

DEFINITIVELY RULED OUT (checked by two or more surveys):
  Zombie -- the chick CNS experiment has NO expression readout; the mouse olfactory bulb one has
    only 2 genes (Tbx21, Th) and static barcodes with no clonal reconstruction.  Framed by its own
    authors as a compatibility demonstration.  No follow-up application 2021-2026; the Askary lab
    went to INSCRIBE (signalling, not lineage) and SMORE.
  scGESTALT / ScarTrace / LINNAEUS / CARLIN / DARLIN / Polylox / PolyloxExpress / MEMOIR-original
    -- all dissociated or in vitro colonies.  No spatial variant has ever been published for any
    of them.  MEMOIR-original is cultured mESC colonies with ONE gene (Esrrb), not tissue.
  packer19 -- dissociated, positions atlas-assigned.  Confirms our own read.
  All CNV / mtDNA / SNV work (CalicoST, HTAN/WUSTL 131 sections, Erickson prostate, STmut, SUMMIT,
    SpaceTracer) -- lineage INFERRED and spot-level.  Only InSituCNV (bioRxiv 2025, Xenium
    Prime/CosMx WTx) reaches true single-cell inferred clones, and it ships no clone calls.

A SCOPE PREDICTION WORTH MAKING BEFORE RUNNING ANYTHING.  Our C. elegans result says the method
needs clones to occupy distinguishable territory.  Intestinal crypts ARE clonal units, so
Spatio-DARLIN intestine should give a large r(a) dynamic range.  Developing cortex involves radial
migration, so SPACE-seq cortex and Spatio-DARLIN brain may look like the worm (flat r(a)).  If
that pattern holds it validates the screen on independent data and turns the scope limit into a
result.

2026-09-01  THE REFRAME THAT WIDENS THE VALIDATION SET, AND A PROVENANCE CORRECTION
-------------------------------------------------------------------------------------
r(a) IS ESTIMATED FROM DIVERGENCE TIMES AND POSITIONS.  EXPRESSION IS NOT INVOLVED -- it is needed
only for the mapping application.  Every survey so far screened on all THREE components and so
missed a set of datasets that can validate the measurement half, several with GROUND-TRUTH TREES
from live imaging (zero reconstruction error):

  TypeTracker    Betjes et al., Sci Adv 2023 (van Zon / Tans).  MOUSE INTESTINAL ORGANOIDS.
                 9 organoids, 3D confocal at 12 min for 60 h, ALL cells tracked, full trees to
                 ~7 divisions, 869 endpoint sister pairs, 3D positions throughout.  Endpoint
                 identity by antibody panel.  NO ACCESSION ("data in the paper") -- would need an
                 email.  Directly tests the crypt prediction.
  intMEMOIR      mESC arm: 1,453 cells / 105 colonies, time-lapse ground-truth trees, same
  (mESC arm)     colonies then fixed and read by smFISH, positions measured in-colony.
                 CaltechDATA 10.22002/D1.1444.
  EPiC           C. elegans, Waterston/Murray.  260 experiment series, x/y/z per named cell PER
                 TIMEPOINT from automated tracking, ~130-180 genes by promoter reporter (one per
                 embryo).  Bulk zip at epic.gs.washington.edu/downloads/zipped_CD_Files.zip.
                 THIS is the real "known lineage + measured position" worm resource.
  CShaper        Cao et al., Nat Commun 2020.  17 C. elegans embryos, 4->350 cells, measured
                 shape, volume, migration, nucleus position, cell-cell contacts.  No expression.
                 github.com/cao13jf/CShaper.
  ASTEC          Guignard et al., Science 2020.  Phallusia mammillata, 2 fully tracked embryos,
                 58,454 3D cell snapshots, 1,342 cells, 639 divisions, 2-min interval, measured
                 geometry + contact areas, lineage DIRECTLY OBSERVED.  No transcriptomics.

PROVENANCE CORRECTION TO OUR OWN C. ELEGANS RESULT (46_celegans.py).  I described packer19's
positions as "atlas-assigned".  Sharper: GEO GSE126954 has 86,024 cells and NO coordinate columns
at all -- the entire annotation is cell/n.umi/time.point/batch/Size_Factor/cell.type/subtype/
embryo.time/lineage/QC.  pycea ships a CURATED 988-cell subset indexed by INVARIANT LINEAGE NAME
(ABala, ABalp, ...) with x/y/z joined from an embryonic atlas.  So the coordinates are real
measured positions of NAMED cells, but not of the sequenced cells.  The r(a) measurement stands
(it needs lineage + position, both of which are real), but it is a measurement on the atlas, not
on Packer's cells, and 44% of Packer's cells have no lineage label anyway.  State it that way.

FIELD-WIDE GAP, CONFIRMED BY TWO INDEPENDENT SURVEYS: no published dataset combines a LIVE-IMAGED
tree with a transcriptome or spatial gene panel on the SAME cells at real tissue positions.  The
corners are TypeTracker (live tree + 3D position + protein), intMEMOIR (live tree + position +
tiny panel, colonies), trackSeq / Live-seq / Image-seq (transcriptome + live imaging, no tree,
dish coordinates), PEtracer / Spatio-DARLIN / SPACE-seq (barcode tree + position + expression, no
live imaging).  VERIFIED NEGATIVE: Zebrahub is NOT same-embryo -- the 120,444 sequenced cells and
the light-sheet tracking used SEPARATE embryos, linked only by developmental stage.

ASCIDIAN SURVEY (sixth and final): ASTEC IS THE BEST PUBLIC VALIDATION SET FOR r(a)
-------------------------------------------------------------------------------------
No ascidian dataset has measured position AND transcriptome on the same cells; there is no
spatial transcriptomics on any ascidian embryo at all (the only ascidian ST is an adult Styela
clava endostyle, Stereo-seq, CNGBdb CNP0004228).  But r(a) does not need expression, and on that
criterion ASTEC is the strongest resource found in any of the six surveys:

  ASTEC / Guignard et al., Science 2020.  Phallusia mammillata.  TEN wild-type embryos
  (ASTEC-Pm1..Pm10), multiview light-sheet every 2 min, 64-cell to late neurula.  ASTEC-Pm1 alone
  = 67,003 segmented 3D cell snapshots, 1,395 cells, 697 divisions; the pipeline missed 4 of 1,399
  cells and 3 of 700 divisions, 99% voxel accuracy.  Lineage DIRECTLY OBSERVED (not reconstructed).
  Positions, volumes, membrane shapes and cell-cell CONTACT AREAS all measured.
  PUBLIC: figshare project 64301, now NINETEEN datasets (raw/fused/segmented .inr/.nii, .obj
  meshes, per-cell geometric properties as .xml/.pkl).  Code github.com/astec-segmentation.
  Follow-up Biasuz et al., bioRxiv 2025.10.13.681089 automates Conklin naming to ~700 cells across
  8 more embryos and corrects earlier datasets.
  (An earlier survey said "2 fully tracked embryos" -- the correct figure is 10 wild-type, 19
  datasets on figshare including U0126-treated and a half-embryo.)

WHY IT BEATS TypeTracker AS A VALIDATION SET: it has an accession (TypeTracker's data is "in the
paper", no repository), ten embryos rather than nine organoids, exact division counts, and
contact-area geometry that would let us check whether r(a) is really about physical packing.
CAVEAT: Phallusia embryos are small and invariant -- by the C. elegans logic r(a) may well come
out flat.  That IS the test: if r(a) is flat in ascidian and worm embryos and steep in tumours,
the screen is measuring something real about tissue architecture rather than about recorders.

CONVERGENCE NOTE: six independent surveys now agree on the shape of the field and are returning
diminishing new information.  The tree-bearing spatial-transcriptomic set is PEtracer,
Spatio-DARLIN, SPACE-seq (+ small intMEMOIR / iTracer / baseMEMOIR).  The lineage-plus-geometry
set, which is all r(a) needs, adds ASTEC, CShaper, EPiC, TypeTracker and intMEMOIR's mESC arm.
Stop surveying; start downloading.

SEVENTH SURVEY (inferred clones + spatial): NOTHING CHANGES, BUT ONE USEFUL RESOURCE
--------------------------------------------------------------------------------------
Confirms the inferred-clone branch is SPOT-LEVEL everywhere (Visium 55 um, legacy ST 100 um) and
that per-spot clone labels are almost never deposited.  Does not affect our premise finding.

WORTH HAVING: CalicoST's Zenodo 10.5281/zenodo.13667151 (CC-BY) ships ACTUAL PER-SPOT CLONE
ASSIGNMENTS plus allele-specific CNAs for the HTAN WashU cohort (12 patients / 26 Visium slices,
CRC liver met + breast + PDAC) and the Erickson prostate slices -- no dbGaP or EGA approval needed.
That is the one place the territoriality screen could be run at scale on INFERRED clones.  Our
BCSA result (0.96x, n=2 sections) would go to n~26 across three cancer types.  If they all land
near 1.0 it converts "CNV clones have no spatial territory" from a caveat about one dataset into a
properly powered claim, and it is the strongest possible motivation for requiring a recorder.
CHEAP: the labels are a download; the screen is one pass over spot pairs.

CITATION TRAP TO AVOID: Schmid et al., "Benchmarking scRNA-seq copy number variation callers",
Nat Commun 2025 (PMC12491403) is scRNA-seq ONLY -- 6 methods, 21 scRNA-seq datasets.  It explicitly
says CNV callers "are currently being extended to spatial transcriptomics data".  Do NOT cite it as
a spatial benchmark.  There is no published benchmark spanning inferCNV / CopyKAT / SCEVAN /
SpatialInferCNV / STARCH / CalicoST / Numbat on spatial data; the closest consensus is
allele-aware (CalicoST ~ Numbat ~ XClone) > expression-only spatial (STARCH, SpatialInferCNV) >
scRNA-seq ports (inferCNV).  Also: CopyKAT and SCEVAN have NO spatial data of their own -- every
spatial application of them in the literature is third-party, running them off-label on spots.

EIGHTH SURVEY: KP-SPATIAL IS THE COMPARATIVE RESOURCE I SAID WAS MISSING
--------------------------------------------------------------------------
  KP-Spatial / Jones MG*, Sun D*, Min KHJ, Colgan WN, ... Chen F, Weissman JS, Yosef N, Yang D.
  "Spatiotemporal lineage tracing reveals the dynamic spatial architecture of tumour growth and
  metastasis."  NATURE GENETICS, IN PRESS.  bioRxiv 10.1101/2024.10.21.619529 (v2, 24 Oct 2025);
  PMC11526908.
    mouse KP-Tracer autochthonous lung adenocarcinoma AND metastases
    EVOLVING Cas9 target sites, expressed as polyA transcript, RECOVERED DIRECTLY FROM THE
      SPATIAL ASSAY (not a separate arm)
    Slide-seqV2, 10 um beads: 44 ARRAYS OVER MORE THAN 100 TUMOURS.  Plus Slide-tags, 5 arrays,
      50-70% of nuclei localised.
    whole transcriptome
    DATA: Zenodo 10.5281/zenodo.19771805 (25 Apr 2026, KPSpatial_Data.tar.gz, 6.1 GB) --
      AnnData spatial objects, ALLELE TABLES, RECONSTRUCTED TREES, plasticity scores.
      No GEO accession in the preprint; the trees are already built and deposited.

WHY THIS CHANGES THE PRIORITY.  I have been calling "one tumour" the binding constraint on the
comparative r(a) study.  KP-Spatial is >100 tumours in a single deposit with trees included, in a
second cancer model (autochthonous lung adeno, not transplanted 4T1), with primary AND metastatic
sites.  That is a better first target than downloading PEtracer's other nine tumours: more
samples, more biological contrast (primary vs met), and no tree reconstruction needed.
CAVEAT: Slide-seqV2 beads are 10 um, so positions are bead positions and a bead may mix cells --
coarser than PEtracer's MERFISH single cells.  For r(a) that is workable (we need pair distances)
but it adds mixing noise and should be checked against the bead-level version of the screen.

REVISED ORDER TO ACTUALLY RUN
  1. KP-Spatial (Zenodo 19771805) -- >100 tumours, trees included, primary vs metastasis.
     The comparative r(a) study, immediately.
  2. PEtracer's other nine tumours (GSE290975 / figshare 28473866) -- same recorder as ours,
     MERFISH single-cell, so it is the like-for-like replication of everything we have measured.
  3. Spatio-DARLIN intestine (GSE296896/7) -- tests the crypt prediction on independent data.
  4. ASTEC (figshare 64301) -- perfect trees, tests whether the screen correctly says no.

Also new and tree-bearing: CERAMIC (mouse liver tumour initiation, bioRxiv 10.64898/2026.07.13
.738237, barcodes recovered from the spatial data; platform and accessions unverified).
ORGANISM NEGATIVES, now backed by explicit hit counts: NO valid Drosophila or zebrafish spatial
lineage dataset 2024-2026.  Mouse embryo/gastrulation all fail on measured position -- including
the PEtracer embryo atlas (1.79M cells, position ASSIGNED from a published Stereo-seq atlas).

2026-09-01  HERITABLE EXPRESSION AS AN ENDOGENOUS LINEAGE MARK: NOT IN THIS DATA
  47_heritable_expression.py, heritable_expression.csv
----------------------------------------------------------------------------------
Pedro's proposal: find genes whose expression tracks the phylogeny but is unrelated to phenotype,
and use them for lineage tracing directly.  Tested on all three PEtracer sections: for 300k cell
pairs per section, corr(|x_gi - x_gk|, patristic distance) per gene, PLUS the partial correlation
controlling for spatial distance (clones occupy territory, so any spatially structured gene --
niche, gradient, field-of-view batch -- fakes phylogenetic signal).  Null permutes which cell
carries which tree row.

  section   raw signal (mean / max)   controlled (mean / max)   genes over 99th pct of |null|
  M3-4        +0.0026 / +0.0452         -0.0009 / +0.0485             33 / 175
  M3-1        +0.0023 / +0.0512         -0.0018 / +0.0438              4 / 175
  M3-2        -0.0007 / +0.0635         -0.0029 / +0.0521              3 / 175

THE ANSWER IS NO, on three grounds:
 1. EFFECT SIZE.  The best gene in any section reaches 0.05.  Reconstructing a tree from expression
    would need something like 0.5.  The mean across genes is indistinguishable from zero.
 2. IT DOES NOT REPRODUCE.  Per-gene controlled signal across sections: r = 0.25 / 0.33 / 0.44.
    (M3-4's "33 genes" is not more signal -- its null threshold is 0.0071 against 0.023 and 0.026,
    i.e. a tighter null from more cells.  Compare thresholds, not counts.)
 3. WHAT LITTLE SIGNAL EXISTS IS PHENOTYPE GENES -- the exact opposite of the proposal.  The only
    genes in the top 20 of all three sections are Cldn4, Nes and Sox9: epithelial claudin,
    nestin, and a progenitor TF.  State markers, not neutral heritable marks.

CAVEATS THAT KEEP THE IDEA ALIVE ELSEWHERE
  - 175-gene MERFISH panel SELECTED FOR CELL STATE.  This is the worst possible panel in which to
    look for phenotype-unrelated heritable genes.  The proper test is whole transcriptome:
    KP-Spatial (Zenodo 19771805) or SPACE-seq (GSE327687).  Run it there before closing the idea.
  - MERFISH counts are sparse per gene, which attenuates the correlations.

MECHANISTIC PRIOR, WHICH IS THE REAL ARGUMENT.  Transcript level has a memory of hours to days
(mRNA half-life, chromatin re-equilibration); lineage here spans five weeks.  There is no mechanism
for RNA abundance to carry five weeks of ancestry.  What IS heritable over that span is genetic and
epigenetic: allele-specific expression, DNA methylation, CNV dosage, mtDNA.  If an endogenous mark
is wanted, that is where to look -- and note the surveys already map those routes (SUMMIT for
mtDNA; the CNV route, which our own BCSA result shows has no spatial territory).  KP-Tracer's
headline finding is PLASTICITY, i.e. that state is NOT stably heritable within a clone, which is
the same statement from the other side.

2026-09-01  KP-SPATIAL DOWNLOADED AND CHECKED: THE TREES ARE NOT TIME-CALIBRATED
-----------------------------------------------------------------------------------
data/kpspatial/KPSpatial_Data.tar.gz -- Zenodo 19771805, 6,090,956,578 bytes,
md5 28cb8a0ba1e5d6e48d5f09b4b153f851 VERIFIED, 4,666 entries, gzip OK.
Layout: slideseq/{trees,expression,hotspot,fitness_signature.tsv,neighborhood_composition.tsv.gz},
slidetags/{trees,expression}, KP_2022/, simulations/, KPSpatial_meta.tsv,
hotspot_modules_consensus.tsv.  49 pucks (44 slide-seq trees + slidetags).

THE ANSWER TO THE QUESTION I FLAGGED: NO, r(a) DOES NOT TRANSFER.
  Trees are cassiopeia.data.CassiopeiaTree pickles (*_hybrid.pkl), from the hybrid solver.
  BRANCH LENGTHS TAKE ONLY 2-3 DISTINCT VALUES (1.0, 2.0, occasionally 3.0) -- unit-length
  topology, not measured time.
  NONE OF THE TREES IS ULTRAMETRIC.  Leaf depths within one tumour span e.g. 4-292 (B94_04),
  3-505 (C12_09), 3-226 (C12_04); leaf-depth sd ranges 0.5 to 100.3 across trees.
  NO time column anywhere: allele tables carry cellBC/intBC/allele/r1/r2/r3/lineageGrp/UMI/
  readCount; meta carries Puck/ID/Mouse/Area/Batch/Slide-seq ID/mESC.

WHY, and it is not a defect of the deposit.  PEtracer's tree is ultrametric with branch lengths
in DAYS because prime editing installs immutable marks over a known 35-day window -- that is an
unusual property, not the norm.  KP-Tracer uses evolving Cas9 scars over 3-5 MONTHS, so depth in
edges reflects how many edits a lineage happened to accumulate (rate x detection), not elapsed
time, and the recorder has likely saturated.

CONSEQUENCE FOR THE TWO PAPERS
  METHOD PAPER: stays on PEtracer, and this strengthens rather than weakens it.  r(a) requires a
    TIME-CALIBRATED tree; PEtracer is the dataset that has one.  Say so explicitly as a scope
    condition -- it is a real modelling requirement, not a convenience.
    If KP-Spatial is ever wanted, the options are (a) refit ultrametric branch lengths under a
    molecular clock (cassiopeia.tools branch-length estimation) and inherit its assumptions, or
    (b) reformulate as r(topological depth), which stays monotone and usable as a cost but loses
    the physical units and the diffusion reading.  Do not pretend depth is time.
  PLASTICITY PAPER: unaffected and still strong.  That question needs clone identity + expression
    + position, none of which require divergence times.  Assets confirmed present: 46 expression
    objects, per-tumour hotspot gene modules, a consensus module table, a fitness signature, and
    neighbourhood composition.
    CAVEAT ON THE SITE CONTRAST: the metastasis samples are NOT spread across mice.  Of 49 pucks,
    44 are Primary; the only LN1/Dia1/Rib1/Rib2 samples are FOUR pucks, all from a SINGLE mouse
    (SPC-11, which also contributes 22 of the primaries).  So primary-vs-metastasis is n=1 mouse.
    That is much weaker than ">100 tumours, primary and metastasis" implied.  The real strength
    here is 44 primary tumours across ~24 mice, not the metastatic contrast.
