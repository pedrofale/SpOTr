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
