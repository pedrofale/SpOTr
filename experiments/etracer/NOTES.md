# eTRACER — the first non-circular test of SpOTr's premise

Data: Zenodo [10.5281/zenodo.19230230](https://doi.org/10.5281/zenodo.19230230)
(`Tumor_Immune_Evasion_eTRACER.zip`, 2.1 GB, CC-BY-4.0). Paper:
[bioRxiv 2025.08.06.668639](https://www.biorxiv.org/content/10.1101/2025.08.06.668639).
Mouse EGFR-L858R;Trp53-/- LUAD under OT-I CD8 pressure; three stages (baseline, early
escape, late escape); Stereo-seq bin50 (25 um).

## Why this dataset settles the question the others could not

- The tree is a **real lineage phylogeny** (IQ-TREE2 ML on indel alleles written into
  endogenous 3'UTR target sites), not a CNV-derived clone graph.
- It is reconstructed **from barcodes only**, so it is not informed by position. That
  rules out the circularity that disqualifies KP-Tracer (trees built after 5 rounds of
  30 um spatial-neighbour imputation) and BaSISS (clone label = smoothed spatial field
  sampled at the cell's own coordinates).
- Its **leaves are Stereo-seq bins labelled by their own coordinates** (`10800_1700`),
  so tree metric and spatial metric live on identical units, at SpOTr's own resolution.
- Trees carry no branch lengths, so distances are topological (unit edges).

## Test

Per clone: does tree distance between two bins predict their spatial distance?
Null: permute the leaf-to-coordinate assignment within that clone (999 permutations,
one-sided). 54 clone trees (7 BL / 22 EE / 25 LE), leaves capped at 400 by random
subsample. `etracer_premise.py`.

## Result — the premise is real, and small

| stage | clones | mean r | median r | r > 0 | p < 0.05 |
|---|---|---|---|---|---|
| baseline | 7 | -0.001 | 0.007 | 4/7 | 0/7 |
| early escape | 22 | 0.064 | 0.070 | 15/22 | 11/22 |
| late escape | 25 | 0.060 | 0.067 | 20/25 | 7/25 |
| **all** | **54** | **0.054** | **0.039** | **39/54** | **18/54** |

Aggregate significance (`etracer_coherence.py`):

- positive r in **39/54** clones — binomial p = **7.5e-04**
- individually significant in **18/54** vs **2.7 expected** — binomial p = **6.5e-11**
- Wilcoxon r > 0 across clones — p = **1.5e-04**
- permutation null centred at **+0.0000** (sanity check passes)
- best clones: r = 0.268, 0.255, 0.250 (all p <= 0.002)

**So the tree genuinely predicts spatial position — this is the first unambiguous,
non-circular confirmation. But every individual effect is small: mean r = 0.054,
median 0.039, max 0.268.**

Baseline stage is flat (mean r ~ 0, 0/7 significant), consistent with its thin in-situ
coverage (~890 of 32,023 bins carry a clone barcode, vs ~33% at early escape).

## Clone spatial coherence, and a real biological signal

| stage | bins | clones (>=20 bins) | coherence |
|---|---|---|---|
| baseline | 677 | 15 | 0.205 |
| early escape | 13,304 | 23 | 0.258 |
| late escape | 7,140 | 24 | **0.309** |

Coherence rises monotonically as immune escape proceeds — clones become progressively
more spatially segregated. That matches the paper's own clone-aggregation finding and
means late-escape sections are the ones to work with.

## Where this leaves SpOTr — all four systems tested

| system | lineage source | coherence | tree-space signal |
|---|---|---|---|
| BCSA breast | Numbat CNV clones | 0.03-0.17 | r 0.015-0.115, **sign flips between sections of one tumour** |
| GBM (Greenwald) | expression CNV clones | 0.35-0.80 | Mantel +0.45 to +0.93 but underpowered; no matched scRNA-seq |
| PEtracer 4T1 | prime-editing recorder, in situ | 0.310 | cell-pair r = +0.092 (p=0.005); clade Mantel +0.199 (n.s.) |
| **eTRACER LUAD** | **Cas9 recorder, ML trees** | **0.205-0.309** | **mean r +0.054; 18/54 clones significant vs 2.7 expected** |

Two conclusions worth carrying into the writeup:

1. **The premise is real.** BCSA could never show this — there the signal was
   indistinguishable from noise and flipped sign between adjacent sections. In genuine
   lineage-traced data it is reproducible and highly significant in aggregate.
2. **The effect is weak everywhere it has been measured**, consistently across two
   independent lineage-tracing systems (PEtracer r ~ 0.09, eTRACER r ~ 0.05). That is
   biology, not a measurement artefact. It caps how much the Gromov-Wasserstein term can
   contribute, argues for a **small alpha**, and means the honest claim is "the tree adds
   a small but statistically robust amount", not "tree-informed OT transforms mapping".

## Next

- Run SpOTr proper: unmatched side = the 176-261 per-clone scRNA-seq ML trees
  (`LineageTracing/single_cell/sc_trees_{BL,EE,LE}/`), target = Stereo-seq bins
  (`SpatailTranscriptomics/spatial_RNA_count_{BL,EE,LE}.rds`), answer key = the 55
  spatial trees used above. Clone IDs are shared across modalities (7/7 BL, 18/22 EE,
  23/24 LE spatial clones have an scRNA counterpart).
- Use late-escape sections first (highest coherence, most clones).
- Read **LineageMap** (PMID 41648248, "Integrative Inference of Spatially Resolved Cell
  Lineage Trees") before writing up — directly adjacent prior work.

---

# SpOTr run on eTRACER late-escape — the first positive result

Setup: 23 clones that have BOTH an scRNA-seq tree (`sc_trees_LE/`) and a spatial tree.
Single-cell side = 1,515 cells with a real lineage tree; target = 2,000 Stereo-seq bins;
answer key = each bin's clone from `spatial_lineage_LE.csv` (majority clone = 41.2%).
Forward-only FGW Sinkhorn (eps=0.02), so no optimiser confounds. Tree distances are
block-diagonal: within-clone from the ML tree with branch lengths, cross-clone constant
(clones are independent lineages).

## Result: the tree helps

| α | real tree ARI | shuffled tree ARI (3 seeds) |
|---|---|---|
| 0.000 (no tree) | 0.3110 | — |
| 0.020 | 0.3333 | 0.3221 ± 0.0007 |
| 0.050 | 0.3471 | 0.3214 ± 0.0012 |
| **0.075** | **0.3475** | 0.3221 ± 0.0014 |
| 0.100 | 0.3464 | 0.3191 ± 0.0022 |
| 0.150 | 0.3339 | 0.3075 ± 0.0007 |
| 0.300 | 0.3030 | 0.1095 ± 0.0021 |

- Best real tree **α = 0.075, ARI 0.3475** vs **0.3110** with no tree — **gain +0.0365**.
- At that α the real tree is **18.5 sd above the shuffled control**; real > shuffled at
  **8/9 α values**.
- The shuffled curve **collapses** as α rises (0.32 → 0.11) while the real tree holds
  (0.35 → 0.30). Weighting a meaningless structure destroys the mapping; weighting a
  meaningful one does not. That asymmetry is the signature of genuine information.

**This is the first time in this project that the tree has been shown to help.** On BCSA a
shuffled tree did just as well; here it does not, by a wide margin.

The size of the gain is consistent with the premise measurement above (per-clone
tree-space r ~ 0.05): the tree carries a small amount of spatial information and SpOTr
extracts a correspondingly small but highly reproducible improvement.

## The critical condition: composition must match

The first run sampled ~70 cells per clone while the bins are 84% C4/C2/C3
(composition TVD 0.63). Under that mismatch the SAME tree on the SAME data was
**catastrophic**:

| | accuracy | ARI |
|---|---|---|
| no tree | 0.236 | 0.185 |
| shuffled tree | 0.166 | 0.210 |
| **real tree** | **0.016** | 0.151 |

Real lost to shuffled at 0/4 α, and accuracy collapsed to 0.002-0.038 against a 0.412
majority baseline. Re-sampling cells to match the spatial composition (TVD 0.63 → 0.011)
flipped it to real > shuffled at 5/5 α.

Interpretation: balanced OT forces every cell to carry equal mass, so an equal-per-clone
reference *imposes* an equal-per-clone spot composition. The GW term then enforces that
wrong composition through the tree's block geometry — which is why a correct tree is worse
than no tree when the reference is mis-composed. **This is the same finding as on BCSA
(`experiments/bcsa3/NOTES.md` finding 2), now demonstrated as a controlled A/B on one
dataset.** It is the strongest argument yet for unbalanced OT: the fix is to stop pinning
clone abundances to the reference.

## Honest limits

- Gain over expression-only is **+0.037 ARI (~12% relative)**, not transformative.
- Composition was matched *using the answer key*. In a real application you do not know
  the target composition — which is exactly why unbalanced OT (learning the marginal) is
  the necessary next step, not an optional refinement.
- Forward-only sweep; the learned-alpha path is untested here.
- One stage (late escape) of one preprint dataset; BL is unusable (thin coverage) and EE
  is untested.

## Reproduce

```bash
python select_matched.py            # composition-matched cell selection
Rscript subset_rds.R                # subset both Seurat objects (cnv env has Matrix)
python prep_etracer.py              # build etracer_inputs.npz
python run_etracer.py               # real vs shuffled vs no tree
python fine_sweep.py                # fine alpha grid + sd vs shuffled
```

---

# Does the learned per-clade alpha identify spatially-discordant clades?

**Tested directly. As currently implemented: no.**

eTRACER is the ideal testbed for this claim, because `etracer_premise.py` already measured,
independently and per clone, how well tree distance predicts spatial distance. So the
learned alpha_k can be checked against a real answer.

Setup (`learn_alpha.py`): unsupervised loss only — fit each spot's expression as a mixture
of per-clone mean expression profiles from the scRNA-seq. No answer key used in fitting.
400 iterations, lr 0.05, 23 clades.

| | |
|---|---|
| Pearson  alpha vs concordance r | **+0.038 (p = 0.86)** |
| Spearman | **-0.103 (p = 0.64)** |

No relationship. The most-concordant clones by measurement (C5 r=0.268, C8 r=0.198,
C2 r=0.164) received *low* alphas (0.060, 0.004, 0.019), while the highest alphas went to
C10 (r=0.041) and C12 (r=0.072).

## Why — the objective is nearly flat in alpha (`loss_alignment.py`)

| alpha | train loss | ARI |
|---|---|---|
| 0.000 | 0.220302 | 0.3110 |
| **0.020** | **0.220292** (min) | 0.3333 |
| **0.075** | 0.220375 | **0.3475** (max) |
| 0.300 | 0.221219 | 0.3030 |

- The loss varies by **0.76% of its own mean** across the entire alpha range.
- It is **minimised at alpha = 0.02** while mapping quality peaks at **alpha = 0.075**.
- corr(loss, ARI) = **-0.875** — directionally right, but the minimum sits in the wrong
  place and the signal is tiny.

So the expression-reconstruction loss barely notices alpha. Gradient descent on it recovers
noise, which is exactly what the failed correlation shows, and why the training loss moved
only 0.22122 -> 0.22085 over 400 iterations.

## What this means for the claim

The claim is **not dead — it is unsupported by the current objective.** Expression
reconstruction is nearly invariant to placement: many different couplings reconstruct spot
expression about equally well, so alpha is weakly identified. To make per-clade alpha
interpretable the objective has to be sensitive to *placement*, not just to reconstruction.

Options worth trying, in order of effort:
1. Add a spatial-coherence term on the inferred per-clone map (unsupervised, cheap).
2. Sharpen the deconvolution loss — per-clone weighting, or drop genes that are uninformative
   between clones, so alpha has more to bite on.
3. For the eTRACER validation case specifically, fit alpha against the known coupling
   (supervised) purely to establish that a *well-identified* alpha does track concordance.
   That separates "the idea is sound" from "the current loss finds it".

Whichever route, eTRACER should stay the proving ground: it is the only dataset here with an
independent per-clone concordance measurement to validate alpha against.

## Follow-up: three tests, and why they were all underpowered

| test | design | result |
|---|---|---|
| learned alpha (`learn_alpha.py`) | fit alpha by unsupervised expression loss | Pearson **+0.038** (p=0.86) |
| per-clade alpha sweep (`perclade_sweep.py`) | perturb one clade's alpha, objective-free | Pearson **-0.196** (p=0.37) |
| per-clade phi sweep (`perclade_phi_sweep.py`) | same, on the within/cross knob | Pearson **+0.135** (p=0.54) |
| **global sweep** (`global_sweep_perclade.py`) | uniform vector, read each clade's preferred value | alpha **+0.066**, phi **+0.108** |

**The per-clade sweeps were confounded.** Perturbing a single clade away from a uniform
vector is itself penalised by the per-clade equalizer inside `build_perclade_alpha_cost`
(the `s_k` rescaling keyed on `alpha_ref_k`), so 15 of 23 clades trivially "preferred" the
baseline value -- e.g. C36 scored -0.012, -0.012, **+0.058**, -0.011, -0.011 across the phi
grid, a clean spike at the baseline. The global sweep avoids this and is the design to use.

**All four tests are underpowered, and the reason is my own subsampling.** I capped the run
at 2,000 spatial bins for compute. That leaves **18 of 23 clades with fewer than 25 bins** --
their per-clade scores are noise, and they dominate an unweighted correlation.

Weighting by clade size tells a different story:

- size-weighted corr(preferred alpha, concordance) = **+0.603**
- size-weighted corr(preferred phi, concordance) = **+0.620**

And among the only well-powered clades the ordering is exactly as the claim predicts:

| clone | bins | concordance r | preferred alpha | preferred phi |
|---|---|---|---|---|
| C2 | 503 | +0.164 | 0.150 | 0.750 |
| C4 | 824 | +0.006 | 0.050 | 0.500 |
| C3 | 355 | -0.002 | 0.000 | 0.000 |

**Verdict: claim (b) is unresolved, not refuted.** Three clades ordering correctly is
suggestive and nothing more. The test needs clades with enough bins to measure, which means
rerunning on the full **7,119 LE bins** (not my 2,000 subsample) and ideally pooling the
**13,304 EE bins** as well. Until then neither a positive nor a negative claim is supportable.

## Full-scale rerun (`sweep_stages.py`) — the properly powered test

Redone on the **full bin sets**, both stages: LE 1,999 cells x 7,119 bins (23 clades),
EE 424 cells x 12,397 bins (18 clades). Composition matched at each stage (TVD 0.002 / 0.030;
cell counts sized to the largest total at which every clade can still hit its target share).
Global uniform sweep, per-clade readout. **30 clades with >= 40 bins**, up from 5.

| parameter | Pearson | Spearman | concordant vs discordant |
|---|---|---|---|
| preferred **alpha** | +0.125 (p = 0.51) | +0.118 (p = 0.54) | 0.183 vs 0.170 |
| preferred **phi** | +0.227 (p = 0.23) | **+0.294 (p = 0.11)** | **0.617 vs 0.500** |

**Conclusion on claim (b):**

- **alpha does not identify spatially-discordant clades.** Four tests, all null, and this is
  the parameter the claim was stated about. At full power the correlation is +0.125 (p=0.51).
  This should not go in a paper as stated.
- **phi is a live candidate but unproven.** It came out positive in every design tried
  (+0.135, +0.108, +0.227) and concordant clades prefer a higher within-clade weighting
  (0.617 vs 0.500), but it has never reached significance -- best is Spearman p = 0.11 at n = 30.

Two things would sharpen it before deciding:

1. **A finer phi grid.** The current grid is 5 points and a large share of clades land exactly
   on the middle value, so the estimate of "preferred phi" is coarse. A finer grid, or a
   parabolic fit to the score curve, would give a continuous optimum and less tie-breaking noise.
2. **More clades.** Detecting rho ~ 0.29 at 80% power needs n ~ 90; we have 30. BL is too thin
   to add much, but PEtracer contributes ~18 more clades with an independent concordance
   measurement, which would take it to ~48.

Until one of those changes the picture, the defensible statement is the mapping result
(tree helps, +0.037 ARI, 18.5 sd above shuffled), not the interpretability result.

## Finer grid + reliability filtering (`phi_fine.py`) — final state of claim (b)

The 5-point phi grid was replaced with 11 points plus a parabolic fit through the peak, giving
a continuous preferred phi per clade. **Ties at the grid midpoint went from 15/23 to 0/30**, so
the tie artifact is gone — but the correlation did not sharpen:

| grid | Pearson | Spearman | concordant vs discordant |
|---|---|---|---|
| 5-point (coarse) | +0.227 | +0.294 (p=0.11) | 0.617 vs 0.500 |
| 11-point + parabolic | +0.139 | +0.260 (p=0.17) | 0.576 vs 0.512 |

So part of the coarse-grid effect was the tie structure itself.

Restricting to clades whose concordance is better measured:

| filter | n | Pearson | p | Spearman | p |
|---|---|---|---|---|---|
| all (>= 40 bins) | 30 | +0.139 | 0.465 | +0.260 | 0.166 |
| concordance p < 0.05 | 15 | +0.177 | 0.529 | +0.246 | 0.376 |
| **>= 100 tree leaves** | 18 | **+0.384** | 0.116 | **+0.377** | 0.123 |
| >= 150 bins | 16 | +0.386 | 0.140 | +0.371 | 0.158 |

### Verdict

- **Per-clade alpha: dead.** Null in every design (four tests), and it is anyway just the
  expression-vs-structure weight — not the parameter the claim should rest on.
- **Per-clade phi: consistently positive, never significant.** Positive in all five designs
  tried, and the effect roughly **doubles as measurement quality improves** (r = 0.14 across
  all clades, 0.38 among the well-measured ones) — the signature of a real relationship
  attenuated by noise rather than of nothing. But the best p reached is 0.116.

### What it would take

At the observed r ~ 0.38 among well-measured clades, 80% power needs **n ~ 50**; we have 18.
PEtracer contributes roughly 18 more clades with an independent per-clade concordance
measurement (`experiments/petracer/`), which would take it to ~36 — close, and the single
cheapest way to find out. Pooling BL adds little (too few barcoded bins).

**Recommendation for the paper:** lead with the mapping result, which is solid. Hold the
interpretability claim until the PEtracer clades are pooled in; if it still sits at p ~ 0.1,
report it as an observation with the effect size, not as a claim.

---

# THE FULL-SCALE eTRACER LATE-ESCAPE RUN (2026-08-25)

Data: 4,120 dissociated cells (45 clones, composition-matched) x 5,000 Stereo-seq bins sampled
uniformly from all 35,835 LE bins, of which 1,026 (20.5%) carry a clone barcode. C_tree from
per-clone IQ-TREE ML trees with branch lengths (within-clone max 1.471, cross-clone 2.942).
Barcoded bins split fit/val/test; only fit labels drive the loss, only val selects, test is
scored. Every cell is scored by restricting its coupling row to test bins.

## Two prep problems caught before running

**Composition.** The prep initially used water-filling caps, giving a near-uniform cell-side
clone composition against bins that are 85% C4/C2/C3 -- TVD 0.774, WORSE than the 0.63 that
made a real tree score worse than no tree in the earlier work. Composition matching brought it
to 0.053. This was the single largest threat to the run and would have inverted the result.

**Scoring.** The first scoring counted only cells whose global argmax happened to land on a
test bin -- 318 of 4,120, and a different subset per configuration (318/315/289). Fixed by
restricting each cell's row to test bins so all configurations score the same 4,120 cells.

## Results

| configuration | accuracy | ARI |
|---|---|---|
| alpha = 0 | 0.3917 | 0.1865 |
| best global (val-selected, alpha=0) | 0.3917 | 0.1865 |
| learned per-clade (val early-stop) | 0.4189 | 0.2582 |
| control: uniform at the same mean alpha | 0.4153 | 0.2561 |
| control: shuffled per-clade | 0.4152 | 0.2552 |

Accuracy over the alpha grid, scored on all labelled bins:
0.0:0.3808  0.05:0.4083  0.075:0.4073  0.1:0.4085  0.15:0.4068  0.2:0.4100  0.3:0.4131  0.45:0.4034

**1. The tree helps.** +0.032 accuracy from alpha 0 to 0.30 on labelled bins, +0.027 to +0.039
on held-out test bins. Consistent with the earlier ARI +0.037 and with the simulated +0.037 at
a fixed alpha.

**2. The curve is FLAT above alpha 0.05.** The decision that matters is "tree or no tree", not
the precise value -- which is direct real-data support for a fixed default.

**3. Per-clade alpha adds nothing measurable.** Learned vs uniform-at-the-same-mean +0.0036;
learned vs shuffled +0.0037 (z = +2.6, but the learned vector spans only 0.091-0.109 so
shuffling barely changes it). Early stopping on validation fired at iteration 2 of 60, before
alpha could differentiate.

**4. The barcode loss overfits the bins it is fitted on.** On fit bins it is monotone
increasing in alpha and selects 0. Evaluated on held-out labelled bins it can find an interior
optimum -- but unreliably (`split_stability.csv`, 400 resamples per size):

| held-out bins | argmin distribution | picks alpha > 0 |
|---|---|---|
| 616 | 0.0:32% 0.05:66% 0.075:3% | 68% |
| 410 | 0.0:34% 0.05:54% 0.075:11% | 66% |
| 308 | 0.0:39% 0.05:47% 0.075:11% | 61% |
| 205 | 0.0:46% 0.05:34% 0.075:14% | 54% |

The accuracy-optimal alpha is 0.30 and the selector never picks it; modal pick is 0.05. It is
biased in the right direction but under-selects and fails outright in a third of resamples.
Because the accuracy curve is flat above 0.05, under-selection costs only ~0.005 -- so this is
survivable in practice, but it is not a reliable selector.

**5. alpha vs the independent premise measurement: Spearman +0.199 (p = 0.36, permutation
p = 0.18), n = 23.** Right direction and better than the -0.103 from the expression-loss run,
but not significant. Same power limit as before: |rho| under ~0.56 is undetectable at n = 23.

## Verdict

Claim (a) is confirmed on real data at full scale, with the leaks that threatened it caught and
fixed. Claim (b) is not: on this dataset per-clade alpha does not beat a uniform alpha, and the
alpha readout points the right way without reaching significance.

The one clearly novel methodological finding is #4: an anchor loss must be scored on anchor
bins it was not fitted on, or it is monotone and useless. That applies to the CNV case too and
was not visible in simulation, where the anchor covered every spot.

## Why per-clade alpha showed nothing here: data starvation

The learned alpha spans only **0.091-0.109** around its 0.10 init -- validation early stopping
fired at iteration 2 of 60, before any differentiation. The reason is visible in the labelled-bin
counts per clone:

- 10 of 45 clones draw ZERO labelled bins in the 5,000-bin sample
- of the 23 clones in the alpha-vs-premise comparison, 8 have fewer than 5 labelled bins
- three clones (C4 416, C2 270, C3 181) hold most of the mask; the next largest has 28

So for most clones there is essentially no data constraining alpha_k. The correlation does
strengthen as starved clones are dropped, but n collapses with it:

| restriction | n | Spearman | p |
|---|---|---|---|
| all | 23 | +0.199 | 0.36 |
| >= 1 labelled bin | 21 | +0.171 | 0.46 |
| >= 5 labelled bins | 15 | +0.286 | 0.30 |
| >= 10 labelled bins | 6 | +0.543 | 0.27 |

Raising `--n-bins` would help the large clones but not the starved ones: many hold only 1-3
barcoded bins in the ENTIRE tissue, so this is a property of the dataset, not of the sampling.
Per-clade alpha is not estimable on eTRACER LE for most of its clones.

## Verification

Three independent adversarial agents re-derived the prepared inputs from the raw files, each
with its own extraction and sampling implementation, and each obtained BIT-IDENTICAL arrays:
alignment (cell rows vs labels, bin rows vs coordinates), C_tree (patristic distances
recomputed by hand for sampled leaf pairs; block structure; no degenerate clones), and sampling
(uniformity of the bin draw, spatial coverage, no duplicates, no answer-key leakage into HVG
selection). All three passed.
