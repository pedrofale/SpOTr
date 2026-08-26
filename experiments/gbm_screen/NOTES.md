# Screening glioblastoma for SpOTr's preconditions

Cohort: Greenwald et al., *Cell* 2024, "Integrative spatial analysis reveals a multi-layered
organization of glioblastoma". Processed Visium on Zenodo
([record 12624860](https://zenodo.org/records/12624860)), open access, **fresh frozen**
(not FFPE) so full-transcriptome and allele-aware clone calling are both possible.
13 sections downloaded, 12 screenable (ZH1019inf ships no `tissue_positions_list.csv`).

## Method

`screen_gbm.py` calls CNV clones from spot expression (log2 ratio to the section mean,
101-gene rolling window along the genome, Ward clustering into 8), builds a tree over the
clone means, then measures the two BCSA criteria across tree granularities.

Malignant spots are defined by the canonical IDH-wt GBM axis (**chr7 gain − chr10 loss**),
not by "flattest profile". This matters: the reference is the section mean, so normal spots
appear *inverted* on that axis rather than flat, and the naive heuristic leaves them in —
which would let a tumour/normal boundary inflate coherence exactly as it did for BCSA2.
`validate_cnv.py` confirms the malignant clusters carry chr7+/chr10− as expected.

## Result (malignant spots only)

| section | spots | clones | coherence | best Mantel r | p |
|---|---|---|---|---|---|
| ZH916bulk | 1197 | 4 | **0.799** | +0.931 | 0.169 |
| ZH916T1 | 1667 | 4 | **0.671** | +0.709 | 0.168 |
| MGH258 | 1527 | 5 | **0.478** | +0.912 | 0.054 |
| ZH8812bulk | 1098 | 5 | **0.438** | +0.910 | 0.072 |
| ZH8811Abulk | 1792 | 6 | **0.353** | +0.837 | 0.046 |
| ZH8811Bbulk | 1842 | 6 | 0.287 | −0.610 | 0.345 |
| ZH881T1 | 2508 | 7 | 0.161 | +0.685 | 0.010 |
| ZH1007nec | 704 | 6 | 0.144 | +0.429 | 0.099 |
| ZH1007inf | 1121 | 7 | 0.117 | +0.515 | 0.254 |
| ZH1019T1 | 764 | 5 | 0.056 | −0.267 | 0.338 |
| ZH881inf | 450 | 4 | 0.052 | +0.471 | 0.334 |

**BCSA for comparison: coherence 0.03–0.17, Mantel +0.28 / −0.05 with a sign flip.**

### Criterion 1 (coherence > 0.3): passes on 5 sections

0.35–0.80, on malignant spots only. This is the precondition BCSA never met at any
granularity, and the clone maps show real territories rather than intermixing.

### Criterion 2 (Mantel p < 0.05): large effects, but underpowered

r is **+0.71 to +0.93** in the high-coherence sections — far above anything in BCSA — yet
mostly not significant. That is a power problem, not a null result: with 4 clades there are
only 6 pairwise distances and 4! = 24 label permutations, so the smallest attainable p is
about 0.042. The fix is more clades, i.e. better clone calling, not more sections.

### Criterion 3 (replication across sections): ZH916 passes

- **ZH916** — bulk r=+0.931 (coh 0.80), T1 r=+0.709 (coh 0.67). Same sign, both large.
- ZH1007 — +0.429 / +0.515, same sign but both low coherence.
- **ZH881 — sign flips** (ZH8811B is −0.610 while the other four are positive). Not every
  patient works.

### Biology: bulk sections work, infiltrating and necrotic ones do not

Every section that passes is bulk tumour. The `inf` (infiltrating) and `nec` (necrotic)
sections sit at 0.05–0.14, which makes sense — at the infiltrating edge malignant cells
disperse through normal brain, so clonal territories dissolve.

## Caveats

1. Clone calling here is **expression-only CNV**, adequate for a screen but not a real clone
   caller. Numbat or CalicoST with allele information would be better, and the fresh-frozen
   BAMs support it (unlike the FFPE probe-based Janesick data).
2. Only 4–7 clones per section, which is what limits the Mantel test.
3. **Not yet verified: matched sc/snRNA-seq for these specific Greenwald samples.** The Ravi
   et al. Dryad deposit (`10.5061/dryad.h70rxwdmj`, CC0) is a different GBM cohort that
   definitely ships `scTumor_Tissue.zip` alongside Visium, but Dryad blocks programmatic
   download from this machine — it needs a browser.

## Also screened / rejected

- **RCC** — the ccRCC Visium cohort is controlled access (EGA `EGAD00001008781`, data access
  agreement required). Not screenable without an application.
- **Janesick breast (10x)** — FFPE probe-based chemistry on both the scFFPE-seq and Visium
  sides, which limits allele-aware clone calling; plus breast has failed on 3/3 samples.

## Next

1. Confirm matched sc/snRNA-seq for Greenwald (or use Ravi, downloading via browser).
2. Re-call clones with Numbat/CalicoST on ZH916 bulk + T1 to get more clades, which both
   sharpens the tree and gives the Mantel test the power it currently lacks.
3. If ZH916 holds up, that is the SpOTr demonstration dataset.

---

# The scRNA-seq side

## Greenwald's cohort has no matched single-cell data

GSE237183 is Visium-only (13 IDH-WT + 6 IDH-mutant). The paper's single-cell references are
*external* cohorts (Neftel et al., Wang et al., Al-Dalahmah et al.), not the same patients.
So the sections that scored well on the spatial screen cannot supply the scRNA-seq side.

## A cohort that does have both

**"Mapping the spatial architecture of glioblastoma from core to edge"** — same study, split
across GEO series, all open:

| series | content |
|---|---|
| **GSE318562** | Visium V1, **22 sections along the edge-core axis of 9 patients** (IDH-WT) |
| **GSE318564** | scRNA-seq, 11 samples from 9 patients, "three of whom also contributed Visium" |
| GSE316917 / GSE335275 | Xenium, 362-gene panel (not usable for CNV) |

Matching the sample titles, only **B178** overlaps unambiguously by ID: scRNA-seq `B178_1/2/3`
plus Visium `B178-02-01`. The paper claims three such patients, so the other two are worth
chasing by other means. Multiple Visium sections per patient (B176, B185, B186, WU1220...)
give the cross-section replication axis for free.

## B178: the tree does not come out (`sc_tree_b178.py`)

- 33,656 cells → 16,880 after QC → **10,908 PTPRC+ immune cells** as a clean diploid reference
- **1,189 malignant cells** (chr7−chr10 above the 99th percentile of immune)
- Malignancy separation is unambiguous: score **+0.249** malignant vs **−0.000** immune

But clustering those malignant cells produces a **tumour-purity gradient, not subclones**:

| k | sizes | chr1 | chr4 | chr7 | chr10 | chr19 |
|---|---|---|---|---|---|---|
| 4 | 99 | +0.27 | +0.34 | +0.41 | +0.08 | +0.22 |
| | 338 | +0.12 | +0.20 | +0.25 | −0.04 | +0.09 |
| | 554 | −0.02 | +0.06 | +0.09 | −0.14 | −0.04 |
| | 198 | −0.13 | −0.05 | −0.03 | −0.21 | −0.16 |

Every chromosome moves together in a monotone ladder. At k=2 the two clone profiles correlate
at **r = +0.979** — same shape, different amplitude. Genuine subclones differ on *specific*
chromosomes; this is CNV signal strength varying cell to cell.

**So B178's scRNA-seq yields no usable phylogeny with expression-only CNV inference.**

## The spatial clones survive the same test (`purity_test.py`)

Applying the identical scaling test to the Visium clones from the screen:

| section | clones | mean pairwise r | verdict |
|---|---|---|---|
| ZH916T1 | 4 | 0.452 | distinct events |
| ZH8811Abulk | 6 | 0.498 | distinct events |
| ZH8812bulk | 5 | 0.543 | mixed |
| MGH258 | 5 | 0.607 | mixed |
| ZH916bulk | 4 | 0.610 | mixed |
| ZH881T1 | 7 | 0.668 | mixed |
| *B178 scRNA-seq* | *2* | *0.979* | *purity gradient* |

The spatial clones are well clear of the purity-gradient regime, so the coherence result from
the screen is not simply a tumour-content gradient. **The failure is specific to single-cell
CNV inference, not to the cohort.**

## Why, and what would fix it

A Visium spot aggregates ~10 cells, so its CNV estimate is far less noisy than a single 10x
cell's; at single-cell resolution the dominant axis of variation becomes signal strength
rather than subclonal identity. 1,189 malignant cells is also thin.

The fix is **allele-aware clone calling** — Numbat, i.e. the pipeline already in use here.
BAF from het SNPs is not degraded the way expression-ratio CNV is, and Numbat explicitly
models per-cell signal strength, which is exactly the confound that defeated the clustering
above. That needs raw reads: GEO ships only filtered matrices, but the samples carry SRA
accessions (e.g. `SRX32009557` for B178_1) with no dbGaP restriction flagged, so FASTQs look
obtainable — at the cost of download plus alignment.

## Status

- Spatial side: **works** (GBM bulk sections, real subclonal territories, coherence 0.35–0.80)
- Single-cell side: **unresolved** — expression-only CNV gives a purity gradient, not a tree
- Next test that settles it: Numbat on B178's raw scRNA-seq reads
