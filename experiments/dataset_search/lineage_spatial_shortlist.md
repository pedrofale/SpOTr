# Lineage-traced scRNA-seq with matched spatial data — verified shortlist

Produced by a 43-agent search+verification sweep (7 search angles, 35 candidate
verifications, 1 synthesis). Every accession below was checked against the live
archive by the verifying agent; the top ones were additionally spot-checked by hand
(GEO/ArrayExpress HTTP checks) before being reported.

## 1. Deduplication

The 32 verified records collapse to **16 distinct studies**. Six studies were submitted 3–5× under different names:

| Underlying study | Duplicate record names |
|---|---|
| **SPACE-seq** (Jia et al., Cell Stem Cell 2026) | "SPACE-seq (DARLIN Cas9 barcoding + Slide-seqV2)", "…mouse liver tumor, cortex, embryonic liver", "…mouse iCCA liver tumors", "…Camargo/Chen labs", "…developing mouse cortex, liver, NRAS/p53 tumours" — **5 records, 1 dataset** |
| **PEtracer** (Koblan/Yost/Zheng/Colgan et al., Science 2025) | "PEtracer 4T1 lung-metastasis atlas", "…MERFISH atlas", "PEtracer + MERFISH (4T1 metastasis)", "PEtracer syngeneic metastatic tumours" — **4 records** |
| **iTracer / Spatial iTracer** (He et al., Nat Methods 2022) | "iTracer cerebral organoids", "Spatial iTracer — human cerebral organoid" (×2), "iTracer / Spatial iTracer human cerebral organoids" — **4 records** |
| **KP-Tracer spatial** (Jones\*, Sun\* et al.) | "KP-Tracer spatial (Cas9 evolving tracer…)", "KP-Tracer spatial (Slide-seqV2 + Slide-tags)", "KP-Tracer Spatial (Jones & Sun et al.)" — **3 records** |
| **eTRACER** (Yang, Hou et al., Ji/Hu labs) | "eTRACER + Stereo-seq (immune-escape LUAD)", "eTRACER EGFR-mutant LUAD immune-escape", "eTRACER lung cancer immunotherapy (Ji lab)" — **3 records** |
| **Spatio-DARLIN** (Gao et al., Nat Methods 2026) | "Spatio-DARLIN (DARLIN mouse + BMKMANU S3000)", "Spatio-DARLIN mouse small intestine + brain", "Spatio-DARLIN — mouse small intestine and E10 brain" — **3 records** |
| **TREX / Space-TREX** (Ratz et al. 2022) | 2 records, **conflicting verdicts** (strong/yes vs possible/no) — resolved in §3 |
| **SpaceBar** (Kinsler et al. 2026) | 2 records, **conflicting verdicts** (strong/controlled vs possible/none) — resolved in §3 |
| **baseMEMOIR** (Chadly et al.) | 2 records |

---

## 2. Ranked shortlist — the STRONG candidates

**Critical framing point that applies to every one of them:** every strong candidate reads the lineage barcode *in situ*, off the same spots/beads/cells as the transcriptome. None is a naturally unmatched cells↔spots pair. The unmatched setting must be *simulated* by withholding the pairing — which is a feature (exact ground truth) but means none of these demonstrates SpOTr's stated use case unaided. The one partial exception is eTRACER (#2), which also has dissociated scRNA-seq trees over the same tumours.

The load-bearing axis for SpOTr is **tree depth**, not clone count. Evolving recorders (PEtracer, eTRACER, KP-Tracer) give graded ultrametrics; one-shot Dox-pulse recorders (DARLIN) and static barcodes (SPLINTR, TREX, SpaceBar) give a near-star metric where the GW term degenerates to clone-block matching.

---

### #1 — PEtracer 4T1 lung metastasis (Koblan, Yost, Zheng, Colgan et al., Science 2025)
**Best tree + best-packaged data. The validation dataset.**

- **Technology:** prime-editing evolving recorder (PE2). 3 orthogonalised edit sites (RNF2/HEK3/EMX1) × 8 predefined 5-nt lineage marks × ~10–15 intBC-tagged cassettes = **36–63 informative edit sites per clone**. NJ/UPGMA trees, validated against puro/blast static barcodes in 6 clones.
- **Organism/tissue:** mouse (BALB/c), 4T1 lung metastases, 3 mice (M1/M2/M3), 20 µm serial lung sections. Plus an in vitro coverslip colony dataset.
- **Spatial modality:** MERFISH, **true single-cell/subcellular**, 124-gene panel (M1/M2) and 175-gene (M3). Lineage decoded *in the same imaged cells* by in-gel T7 amplification after the RNA rounds, aligned with `fishtank`.
- **Lineages:** 11 founding clones / 10 tumours; **18 clades** within M1-T1 (19,029 cells); trees 1,265 → 19,029 cells; ~515k spatially resolved lineage-bearing cells; in vitro 64 clones / 18,675 cells; 6 barcoded ground-truth clones. Only ~4–5 clones clear the >50% edit-fraction bar for high-confidence phylogenies.
- **Accessions:** figshare **10.6084/m9.figshare.28473866** ("PEtracer Data", CC BY 4.0, ~10.6 GB) — `M1/M2/M3_tumor_tracing.h5td`, `M1_polygons_grid.json`, `M2_polygons.json`, `M3_polygons_grid.json`, `colony_tracing` + `colony_polygons.json`, `barcoded_tracing_clone_1..6.h5td`, `10x_4T1_primary.h5ad`. GEO **GSE290975** (scRNA-seq only). SRA BioProject **PRJNA1231108**. Code github.com/jweissmanlab/PEtracer-2025.
- **Paper:** https://www.science.org/doi/10.1126/science.adx3800 (PMID 40705858, PMC12766569; preprint 10.1101/2025.06.15.659774)
- **Fit:** the `.h5td` objects already ship `obst['tree']`/`obst['nj']`, `obsm['characters']`, `obsm['spatial']`/`spatial_grid`/`spatial_3d`, **and `obsp['tree_connectivities']` + `obsp['spatial_connectivities']`** — i.e. a tree metric and a spatial metric precomputed over the same cell set, exactly the pair a GW term consumes. Paper already reports tree–space distance correlation (Fig. 3N, 5H).
- **Doesn't fit:** 124–175 genes only, so the Wasserstein/expression term is panel-limited. Single-cell not spot resolution — bin into pseudo-spots to exercise the intended regime. **GEO contains NO in vivo lineage libraries** (GSM8826710/11/12 are GEX only) and no MERFISH; the entire spatial resource is on figshare. Raw images not public. Minor unresolved: one record lists `colony_tracing.h5ad`, another `.h5td`.

---

### #2 — eTRACER immune-escape LUAD (Yang, Hou et al.; Ji / Zheng Hu labs)
**The only dataset with a dissociated single-cell tree AND spatial data from the same tumours. The demonstration dataset.**

- **Technology:** CRISPR-Cas9 evolving recorder writing into neutral 3′UTRs of 14–15 highly expressed *endogenous* genes (28 target sites) + 14-bp static clone barcodes (libraries of 474,796 / 232,094). Maximum-likelihood phylogenies by **IQ-TREE2 v2.2.2.7**, 1,000 ultrafast bootstrap + 1,000 SH-aLRT.
- **Organism/tissue:** mouse; EP-OVA-eTRACER (EGFR-L858R; Trp53−/−) serially transplanted subcutaneously in Rag2−/− under OT-I CD8 pressure. 3 stages (baseline / early-escape / late-escape), 3 tumours each co-arrayed on one chip.
- **Spatial modality:** Stereo-seq (BGI v1.2), **bin50 = 25 µm**. Static + evolving barcodes recovered by amplicon-seq off the Stereo-seq cDNA library itself.
- **Lineages:** ~2,796–2,798 clones at baseline; **173–175 clones with >15 cells** used for main analyses; 103 clones for branching index. **55 spatially-resolved per-clone Newick trees** (7 BL / 23 EE / 25 LE, leaf counts 5 → 6,717 bins) and **176–261 per-clone scRNA Newick trees** (count differs between the two Zenodo versions — verify against the record you download). 91,411 QC-passed cells across 15 mice. Clone IDs are shared across modalities: 7/7 (BL), 18/22 (EE), 23/24 (LE) spatial clones have an scRNA counterpart.
- **Accessions:** Zenodo **10.5281/zenodo.19230230** (v2, `Tumor_Immune_Evasion_eTRACER.zip`, 2,147,819,399 bytes, CC-BY-4.0, 2026-03-26) and **10.5281/zenodo.16406856** (v1, `eTRACER.zip`, 2.1 GB). Key paths: `LineageTracing/spatial/spatial_trees_{BL,EE,LE}/*.phy.treefile`, `spatial_lineage_{BL,EE,LE}.csv` (CellID, lineageGrp, num, x, y), `spatial_*_allele_table.csv`, `LineageTracing/single_cell/sc_trees_*/`, `SpatailTranscriptomics/spatial_RNA_count_{BL,EE,LE}.rds`, `RCTD_results_*.rds`, `scRNA/sc_UMI_count.rds`. Code github.com/liangzhenhou/eTRACER.
- **Paper:** https://www.biorxiv.org/content/10.1101/2025.08.06.668639 (v2 retitled "Testing the plasticity-first theory of tumor evolution with spatiotemporal lineage tracing")
- **Fit:** Newick trees on both sides with **bin coordinates as leaf labels** (e.g. `10800_1700`), so tree metric and spatial metric are defined over identical units. Gives you both regimes at once: unmatched (sc tree → spatial bins) *and* in-situ ground truth to score against. Paper already reports clone spatial aggregation and a clone-distance-to-margin ↔ cell-state gradient.
- **Doesn't fit:** raw reads at NGDC **PRJCA040871 are NOT released** (page says so; both reviewer share links dead) — processed only. Unreviewed preprint; numbers drifted between v1/v2. bin50 ≠ single cell. Baseline stage has thin in-situ coverage (~890 of 32,023 bins carry a clone barcode, vs ~33% at EE). Spatial trees are per-clone, so a cross-clone spatial tree must be grafted onto the static-barcode partition. Subcutaneous allograft, not autochthonous lung.

---

### #3 — KP-Tracer spatial (Jones\*, Sun\*, Min et al.)
**Largest tissue-scale architecture: primary + metastasis in one animal.**

- **Technology:** KP-Tracer evolving Cas9 recorder, ~13 intBCs × 3 cut sites (~39 sites/cell), polyadenylated target sites. Cassiopeia NJ / Greedy-hybrid, LBIFitness, allele priors. inferCNV as orthogonal signal on Slide-tags.
- **Organism/tissue:** autochthonous *Kras*-LSL-G12D/+;*Trp53*-fl/fl mouse lung adenocarcinoma, SPC-Cre initiated, 12–16 weeks. 24 mice.
- **Spatial modality:** Slide-seqV2 (44 arrays, 10 µm beads) + **Slide-tags (5 arrays, true single nuclei)** = 49 arrays.
- **Lineages:** **124 distinct clonal tumours** (counted from `slideseq_fitness.tsv`: 167 `sample_tumor` groups − 43 "non-tumor"), 682,659 lineage-bearing spots, 41,185 Slide-tags nuclei. Mouse SPC-11 alone = 26 serial sections of one lung + LN, 2 rib and 1 diaphragm metastasis, all Slide-tags arrays from that animal.
- **Accessions:** Zenodo **10.5281/zenodo.19771805** (`KPSpatial_Data.tar.gz`, 6,090,956,578 bytes, md5 28cb8a0ba1e5d6e48d5f09b4b153f851, CC-BY-4.0, 2026-04-25; concept DOI 10.5281/zenodo.19771804). Code github.com/mattjones315/KPSpatial-release; Cassiopeia github.com/YosefLab/Cassiopeia. **No GEO/SRA exists** (confirmed absent). Companion non-spatial scRNA-seq phylogenies, different mice: Zenodo **10.5281/zenodo.5847462** (Yang\*/Jones\* et al., Cell 2022, PMID 35523183).
- **Paper:** https://www.biorxiv.org/content/10.1101/2024.10.21.619529v2 (Nature Genetics in press per the Yang lab site; not yet in Crossref/PubMed — do not cite a journal DOI)
- **Fit:** true primary→metastasis phylogeography in one specimen; 3D serial-layer structure for holding out planes.
- **Doesn't fit — one specific to your method:** Slide-seq trees were built after **5 iterations of 30 µm spatial-neighbour imputation**, so the tree metric is *already partly informed by spatial proximity* — a genuine circularity risk for a tree-vs-space GW claim. Use the non-imputed target-site tables (also provided) for any validation. Also: no raw FASTQs, tarball layout undocumented outside the archive, Slide-seq leaves are multi-cell spots (`allow_allele_conflicts`).

---

### #4 — Spatio-DARLIN (Gao et al., Nat Methods 2026)
**Most clones, cleanest deposition, shallowest tree.**

- **Technology:** DARLIN mouse (dox-inducible Cas9-TdT), three arrays CA/Col1a1, TA/Tigre, RA/Rosa26. Nested PCR off the same spatial cDNA; Read 1 = spatial barcode + UMI, Read 2 (350 bp) = array.
- **Organism/tissue:** mouse small intestine (1/2/3-week and PSI/DSI/MSI) + brain (L0927, L126 s1–s3; E10-induced, collected at 7 weeks). 10 µm cryosections.
- **Spatial modality:** **BMKMANU S3000** (6.8×6.8 mm capture, ~2.5–3.5 µm spot pitch), Cellpose v2.2.2 nuclear segmentation → per-cell matrix + per-cell xy.
- **Lineages:** **~7,000 clones with >1 cell per section** (intestine); 5,631 (brain L126). Reliable barcode in ~25–50% of cells.
- **Accessions:** GEO **GSE296896** (12 spatial samples, GSM8979564-71, GSM9213570, GSM9463728-30; BioProject PRJNA1261896) and GEO **GSE296897** (56 amplicon samples; BioProject PRJNA1261899). Both public 2026-04-12. Per sample: `matrix.mtx.gz`, `features.tsv.gz`, **`features_allele.tsv.gz`** (clone barcodes + mutation details), `barcodes.tsv.gz`, **`barcodes_pos.tsv.gz`** (xy), `HE.tif`, `FL.tif`, `Seurat_tidy_HQ.qs`. Code github.com/LLi-Lab/spatio_DARLIN; MosaicLineage github.com/ShouWenWang-Lab/MosaicLineage.
- **Paper:** https://www.nature.com/articles/s41592-026-03151-5 (Nat Methods 23(8):1588-1597, PMID 42373956)
- **Fit:** the only strong candidate where **expression + per-cell coordinates + per-cell allele strings are all directly downloadable from one open archive with no login and no reprocessing**, joined by a shared segmented-cell barcode key. Three adjacent sections 20 µm apart were collected specifically for clone verification — a ready-made held-out benchmark.
- **Doesn't fit:** single short Dox pulse ⇒ effectively a depth-1 star over ~7,000 clades; no published phylogeny, and the repo does allele calling only. Only 36 of the 56 amplicon samples pair with spatial (20 are L141 bulk-dissected brain regions). Lineage covers ~25% of cells ⇒ inherently unbalanced transport. Intestinal Swiss-roll geometry needs a crypt-villus prior.

---

### #5 — SPACE-seq (Jia et al., Cell Stem Cell 2026)
**Same recorder as #4, adds 3D brain and tumours — but the deposition has an unresolved conflict.**

- **Technology:** DARLIN (CA/RA/TA), alleles as 31-motif indel strings; homoplasy filtered against a ~300,000-granulocyte allele bank; ">60% of beads" carry ≥1 locus, 75%±4.5% rare-and-complex.
- **Organism/tissue:** mouse — 6 NRAS-G12V/shp53 iCCA liver tumours, 1-year cortex/hippocampus (E10 label, 3 coronal sections aligned with STitch3D), 4-week liver from E10/E12/E15.5-labelled mice.
- **Spatial modality:** Slide-seqV2, 10 µm beads; lineage read off the *same bead barcode*.
- **Lineages:** per tumour puck, **452–2,649 distinct TA alleles**; 62–528 on ≥2 beads; 12–172 at ≥5 beads. 58 hepatoblast clones in E10 liver across 19 lobules.
- **Accessions:** GEO SuperSeries **GSE327939** (79 samples, public 2026-06-08; PRJNA1453038) = **GSE327687** [SPACE-seq, 39 samples, PRJNA1451866 / SRP690926] + **GSE326431** [bulk lobe amplicon, 40 samples, PRJNA1445254]. Bead→allele tables **GSM9777579–GSM9777584** (HCC1-6_TA.txt.gz). Single Cell Portal **SCP3620** "Space-seq". Live code github.com/jiayuemeng930930/spatial-darlin_v0 (the paper's `carlin-spatial` URL 404s).
- **Paper:** https://doi.org/10.1016/j.stem.2026.05.017 (PMID 42263684, PMC13401910)
- **⚠️ UNRESOLVED CONFLICT — resolve before committing:** two independent verifications disagree about SCP3620. One reports it public but serving **0 cells / 0 genes / "This study has no data to view"**; another reports **62 files / 5.66 GB**, including 14 pucks each with `digital_expression` + **`*_barcode_matching.txt.gz`** (the bead→coordinate map) and 6 Seurat `.rds` (Brain_903/904 ×3, HCC_55-4, HCC_172-2, HCC_58-2). Both agree that **GEO contains no bead xy coordinates at all**, and that the one GEO puck matrix (`Puck_230425_34`, 49,308 beads) does **not** pair with any deposited lineage table (bead-barcode overlap 0–4). Downloads require a free Broad/Google sign-in (HTTP 401 anonymously). **Action: log into SCP and check SCP3620 before choosing this over #4.**
- **Also:** tree depth varies sharply — HCC2 is richly nested (298 distinct edit tokens, 94/172 clones sharing an edit), HCC6 is nearly star-like (5/81). Tumours are near-monoclonal (top clone 42–71% of beads). Prefer HCC2, cortex, and E10 liver. No dissociated scRNA-seq anywhere in this deposit. Naming collision: an unrelated PNAS 2025 spatial-epigenome method is also called SPACE-seq (PMID 40249782).

---

### #6 — Spatial iTracer, human cerebral organoid (He et al., Nat Methods 2022)
**Only human specimen among the strong set; the cleanest published precedent for your core assumption.**

- **Technology:** iTracer dual-channel — Sleeping Beauty ~11-mer reporter barcode (clone) + dox-inducible Cas9 scar (subclone), scarred at day 15. Explicit 3-level tree: root → barcode family → scar family → cell.
- **Tissue:** one 2-month-old organoid (Org14), 3 × 10 µm sections.
- **Spatial modality:** 10x Visium v1, **2,038 spots** (S1=688, S2=763, S3=587), ~1–10 cells/spot; barcode and scar amplicon libraries built off the same Visium cDNA.
- **Lineages:** 62 distinct barcodes / 190 barcode-composition strings / 248 barcode+scar families / 51 non-wildtype scars. **41.7% of spots barcoded per the paper (850 spots, 32 tracked barcodes in the authors' code); only 29.6% (603 spots) in the deposited metadata under the stricter joint call.** Scar layer in space is thin: 341 spots, 15 barcode+scar families spanning ≥3 spots.
- **Accessions:** ArrayExpress **E-MTAB-10972** (Visium; `counts.mtx.gz`, `features.tsv.gz`, `metadata.tsv.gz` with `GeneBarcodeMerge`+`ScarMerge`, `SLICE_1/2/3.json`), **E-MTAB-10974** (12 organoids scRNA), **E-MTAB-10973** (Org13 microdissected), **E-MTAB-10971** (perturb). ENA PRJEB47710/47712/47776, ERP132012. Mendeley Data doi:10.17632/nj3p3pxv6p. Code github.com/quadbio/itraceR + github.com/quadbio/iTracer_analysis (the paper's `quadbiolab/iTracer` URL was renamed). Working direct download: `https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/972/E-MTAB-10972/Files/metadata.tsv.gz`
- **Paper:** https://www.nature.com/articles/s41592-021-01344-8 (PMID 34969984, PMC8748197)
- **Fit:** Fig. 2f–h is essentially a direct empirical test of SpOTr's premise — spot pairs sharing a barcode are closer in space (Wilcoxon p<1e-4, 53,618 shared vs 197,444 exclusive pairs), pairs sharing barcode+scar closer still, **and it holds within a single annotated brain region**, so it is not a region-identity confound.
- **Doesn't fit:** **the Visium organoid is Org14; the 44,275-cell scRNA-seq atlas is Org1–12 and Org13.** Barcodes are organoid-private (77.9–97.6% unique to one organoid; only 2 of 62 Visium barcode atoms appear among the 2,208 scRNA atoms — chance collisions), so **you cannot join sibling-organoid cells to Org14 spots**. Tree depth = 2 ⇒ a 3-valued ultrametric. One barcode occupies 41% of lineage-bearing spots. Correct the widely-repeated error: **"237 scar families" is the scRNA-seq figure across 12 organoids, not the spatial sample.** Spot→xy needs the public Visium v1 barcode whitelist (`SLICE_*.json` gives row/col + image coords only).

---

### #7 — SPLINTR AML spleen + Stereo-seq (Holze, Talarmain, Fennell, Vassiliadis et al., Cell Rep Methods 2024)
- Static lentiviral SPLINTR barcodes (**flat partition, no tree**), MLL-AF9 AML in mouse spleen, Stereo-seq v1.0 bin10–bin50, barcodes recovered from SAW-unaligned reads.
- **140 clones shared between spatial and matched same-spleen scRNA-seq**; 263–370 spatial barcodes; 55 clones ≥10 UMIs; 95 clones spanning ≥10 bin50 bins.
- GEO **GSE246611** (PRJNA1033784) — GSM7872695 = `Mouse_04_spleen_stereoseq`; **note GSM7872694 is in vitro cultured cells, NOT the spleen**. Matched same-spleen data are the older **GSM4912416** (bulk barcode-seq) and **GSM4912482** (10x scRNA) in GSE161676/GSE161662/GSE161666 (Fennell & Vassiliadis, Nature 2022). Zenodo **10.5281/zenodo.10685805** (CC-BY-4.0, 10.8 GB) has bin-level h5ad + `barcode_composition_sc_bulk_spatial.tsv`. Code github.com/DaneVass/BARtab, /bartools.
- Paper: https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(24)00094-8 (PMC11133760)
- **Value:** the closest thing to a real unmatched sc↔spatial pair with a clone-level answer key. **Limit:** zero hierarchy. Also the same-animal identity of GSM4912482 is asserted by the authors but not verifiable from GEO metadata (Stereo-seq sample is "Mouse_04"; scRNA is "PtprcaDisease mouse 2"). Note the paper's "distinct spatial territory" claim is overstated for the two dominant clones (1.3–1.7× NN enrichment).

---

### #8 — intMEMOIR adult *Drosophila* brain (Chow et al., Science 2021)
- **Bxb1** integrase (not phiC31) editing 10 three-state "trits". 4 brains, **7,966 cells, 624 distinct edited barcodes, 79 clones ≥4 cells, 59 ≥6 cells**. Per-cell tables carry barcode + xpos/ypos + 8 marker genes + clone_ID in one file.
- CaltechDATA **10.22002/D1.1444** (https://data.caltech.edu/records/s3ejh-8rk72, CC-BY, 123 GB across 12 zips; brains in "Figure 5.zip", "Figure S13 B2.zip", "Figure S14 B3.zip", "Figure S15 B4.zip"). Ground-truth Newick trees exist **only for the mESC side**: github.com/raphael-group/intmemoir-processed-data (108 colonies, `true_tree.nwk` + `leaf_locations.txt`). Addgene 158387/158389/158390/158391.
- Paper: https://www.science.org/doi/10.1126/science.abb3099
- **Limits:** only a flat clone partition released for the brain; 10 ternary characters give ~25% topological accuracy by the authors' own benchmark; **8 genes only**, so the expression term has almost nothing to key on.

---

### #9 — baseMEMOIR mESC colonies (Chadly et al.)
- Base-editor recorder, **396 ternary characters (792 bits)/cell**, BEAST2 consensus Newick trees 4–7 generations deep. **7 colonies, 197 cells total** (~18–30 leaves each), 12-gene smFISH panel, per-cell xy.
- CaltechDATA **10.22002/327t7-ke088** (`BaseMEM_Magic.tar.gz`, 46.4 GB) and **10.22002/pmpby-gpj05** (images, 290 GB), CC-BY-4.0. Reprocessed trees + character matrices without the 46 GB download: github.com/ZhangLabGT/LineageMap/tree/HEAD/output (colony_2/4/5). BEAST2 package github.com/rbouckaert/irreversible.
- Paper: https://www.biorxiv.org/content/10.1101/2024.01.03.573434v1 (still preprint-only as of Aug 2026)
- **Use:** a 30×30 GW unit test with an exact answer, nothing more. The authors' own BEAST2 model already couples tree to space via a relaxed-clock diffusion term — read it as prior art.

---

## 3. Near-misses worth chasing

| Dataset | What exists | What's blocking | Action |
|---|---|---|---|
| **SpaceBar melanoma xenograft** (Kinsler et al., Nat Methods 23:328-333, 2026; PMID 41413345; https://www.nature.com/articles/s41592-025-02968-w) | seqFISH (Spatial Genomics GenePS), **single-cell resolution**, ~1,400 clones over ~67,000 cells in one WM989-A6-G3 xenograft section; 96 barcode probes imaged in the same run as the ~113–119-gene panel. Extended Data Fig. 5 has sections from several mice. | **No accession of any kind.** Published statement: *"Raw data, which consist of extremely large files, are available upon request."* github.com/grantkinsler/SpatialBarcodes ships only ~5 KB gene-level clone/space score tables for the in vivo tumour; the only per-cell file (`extractedData/in_vitro/combined_df.csv`, 9.2 MB) is the **in vitro** slide with xy + clone but **no expression**. Also: barcodes are static ⇒ flat 1,400-block star metric, no branch structure. | Email Grant Kinsler / Arjun Raj (UPenn) for the per-cell tumour matrix. Worth it: the authors already show 55 µm pseudo-gridding inflates apparent clonal signal (9 spurious clone-gene correlations vs 1 at true resolution) — a ready-made cautionary benchmark for any spot-level method. |
| **Human osteosarcoma mtDNA** (bioRxiv 10.64898/2026.07.09.737611; Europe PMC PPR1277594) | Same tumour: 11,203-cell 10x Chromium **and** one Visium section (1,849 spots) with MAESTER off the Visium WTA; 6 clones from 7 Moran's-I-filtered variants; MitoDrift WF-HMT phylogeny. Authors already compute cophenetic-vs-transcriptomic Mantel r=0.136 (p=0.001) and clone spatial coherence Z=10.07. | **No data availability statement and no accession** (Europe PMC hasData=N). The mtDNA tree's leaves are **spots**, not cells — the single cells carry no lineage. | Email Joshua W. K. Ho (HKU). Real rescue path: the retained variants sit at MT-RNR1 709 and an MT-RNR2 1861–1872 window, both well-captured in 10x 3′ data, so given the Chromium BAM you could genotype the same loci per cell → tree over cells + space over spots + spot-level answer key on one tumour. Caveat: 6 of 7 variants fall in a 12-nt window all substituting to A, which the authors flag as a possible strand-specific artefact. |
| **SUMMIT** (Bracht et al., bioRxiv 10.1101/2025.07.11.664452; PMID 40791354, PMC12338592) | mtDNA (MAESTER) read off the same Takara/Curio Seeker section as the transcriptome, in **intact human GI tissue** — Barrett's, small bowel, gastric cardia, CRC, plus a mouse xenograft. | No accession (verified four ways). Clone counts 2–6 per specimen, and **anti-correlated with spatial structure**: the only ≥6-clone sample (CRC) is explicitly "complete spatial intermixing" (LISI ≈ 2 for all pairs). No lineage-bearing single-cell side for any human sample. | Low priority. Human GI tissue would likely land in dbGaP/EGA anyway. Contact Sydney Shaffer / Nancy Zhang if you specifically need native human tissue. |
| **BaSISS breast cancer** (Lomakin et al., Nature 2022; https://www.nature.com/articles/s41586-022-05425-2, PMC9668746) | **Fully open, no application:** Sanger FTP https://ftp.sanger.ac.uk/pub/cancer/LomakinEtAl_BaSISS/submission/ (~60 GB) + BioImage Archive **S-BIAD537**. Real 6-branch SNV tree for P1 (PD9694) with ancestry encoded in branch names (`external_data/PD9694/WGS/PD9694_CCF.csv`), single-cell-resolved ISS with per-nucleus xy + 91-gene counts for all 8 sections. Code github.com/gerstung-lab/BaSISS, Zenodo **10.5281/zenodo.7037317** (note: 10.5281/zenodo.703731 is wrong). Browser cancerclonemaps.org. | **The tree comes from bulk multi-region WGS and no cell is genotyped** (0.82 BaSISS signals/nucleus). A cell's clone label is a lookup of a spatially smoothed clone field *at its own coordinates* — so the cell-level tree metric is a deterministic function of position. Aligning it to space with GW is circular. P2 has only 3–4 genotypes. | Use as a **non-circular region-level check**: 20 laser-capture regions in P1 with recorded slide coordinates + independent WGS clone reads (`PD9694_lcm_positions.csv`, `PD9694_lcm_reads.csv`). Also a good ~100 µm clone-territory reference to eyeball SpOTr output against. |
| **Space-TREX** (Ratz et al., Nat Neurosci 2022; https://www.nature.com/articles/s41593-022-01011-x, PMC8904259) | **Verdict conflict resolved:** the *Space-TREX arm alone* is genuinely same-specimen — 8 consecutive Visium sections of one brain (28,746 spots, atlas-registered via WholeBrain) paired with 8 cloneID amplicon libraries from the same sections, giving **162 clones**. GEO **GSE153424** (PRJNA642635, SRP269087): GEX **GSM4644079–GSM4644086**, cloneID **GSM4644087–GSM4644094** (note: not GSM4644090 as one record truncated). Code github.com/frisen-lab/TREX + github.com/ludvigla/TREXSeg. | The scRNA-seq brains (2,360 clones, GSM4644058-78) are **different animals** — GEO marks Visium as "replicate: 6" vs replicates 1–5 for scRNA, and each embryo is independently transduced from a ~0.94e6 library, so expected cross-animal barcode overlap is ~7 chance collisions. **No barcode bridge exists.** Lineage is a flat partition (89.6% of cells carry exactly one non-evolving cloneID). 55 µm spots ≈ 4 cells. GEO provides no processed spot→clone table. | Usable as a fully-public flat-clone validation set or synthetic-benchmark generator. Check the Seurat RDS objects (SharePoint link + password published in the Data Availability statement) before committing to re-running TREX over the Visium BAMs. |
| **Unscreened leads** | **LineageMap** (PMID 41648248, bioRxiv 10.64898/2026.01.19.700383) — "Integrative Inference of Spatially Resolved Cell Lineage Trees". Directly adjacent prior/competing work; benchmarks on baseMEMOIR colonies and ships their trees. **Read this for novelty positioning before writing SpOTr up.** | | |

---

## 4. Negative results — modality families with NO usable matched spatial data

These are worth stating plainly because they close off search directions:

1. **Native mtDNA variants (MAESTER-family) — zero usable matches.** Four independent attempts: SUMMIT (no accession, 2–6 clones, and the one ≥6-clone sample has zero spatial segregation), the HKU osteosarcoma preprint (no accession; tree leaves are spots, cells unlabelled), Gier et al. Barrett's (**seqFISH sections are from patients 15/16 = A197/A200, who appear in neither GSE291080 nor GSE287855** — different patients, decisively), and HEMO/chondrosarcoma (only 3 clones in the organoid arm; the 16-clone chondrosarcoma has no single-cell arm at all). In every case the lineage sits *on the spots* or in *different specimens*. **There is no public dataset with a many-clone mtDNA phylogeny over single cells plus same-specimen spatial transcriptomics.**

2. **Somatic SNV / CNV phylogenies — one instance, and it is circular.** Only BaSISS, where the tree is bulk multi-region WGS and no cell is genotyped. No single-cell-DNA phylogeny with same-specimen spatial data surfaced.

3. **Transposon / L1 retrotransposon lineage tracing — nothing surfaced at all.** No candidate in either verified or rejected sets.

4. **Polylox — nothing surfaced.** The only integrase recorder with spatial data is intMEMOIR (Bxb1), and its tissue instance is a fly brain with an 8-gene panel and no released phylogeny.

5. **Multicolour / Confetti / Brainbow clonal labelling — nothing surfaced** with matched spatial transcriptomics.

6. **Engineered recorders in native human tissue — impossible by construction, and confirmed empty.** Every human entry is either an organoid (iTracer), a xenograft (SpaceBar), or native tissue read with mtDNA/SNVs (all of which fail per points 1–2).

7. **Xenium — zero candidates.** The imaging-based spatial modalities represented are MERFISH, seqFISH/GenePS, sequential-hybridisation smFISH and in-situ sequencing (BaSISS); sequencing-based are Slide-seqV2, Slide-tags, Visium, Stereo-seq and BMKMANU S3000.

8. **Naturally unmatched cells↔spots pairs — essentially none.** Every strong candidate reads lineage in situ. The single partial exception is **eTRACER**, where dissociated scRNA-seq trees exist for the same tumours as the Stereo-seq sections with shared clone IDs. Everything else requires simulating the unmatched setting by withholding a known pairing. **Frame this as a finding, not a gap** — it means the field currently produces answer keys, and SpOTr can be scored rather than merely demonstrated.

---

## 5. The two picks

### (a) VALIDATION with ground-truth lineage + position → **PEtracer (figshare 10.6084/m9.figshare.28473866)**

Nothing else combines all four: a genuinely *evolving* recorder (36–63 edit sites/clone, so a graded ultrametric rather than a clone-block indicator), **true single-cell resolution** (MERFISH + in-gel T7 on the same imaged cells, not 10–25 µm beads or 55 µm spots), **pre-built trees and coordinates in one object** (`obst['tree']`, `obsm['spatial']`, `obsm['characters']`), and — decisively — **`obsp['tree_connectivities']` and `obsp['spatial_connectivities']` already computed over the same cell set**, which is literally the input pair a Gromov-Wasserstein term consumes. Hold out `obsm['spatial']`, run SpOTr, score against the known assignment. Start on `colony_tracing` (64 clones / 18,675 cells, in vitro, clean) before M1-T1 (19,029 cells, 18 clades). Fall back to baseMEMOIR only if you want a 30-leaf unit test with a BEAST2 answer key.

*Runner-up for validation:* Spatio-DARLIN, purely on deposition quality — it is the only candidate where expression, per-cell xy and per-cell allele strings all download from open GEO with no login, joined by a shared cell-barcode key, plus three adjacent 20 µm sections as a built-in held-out benchmark. Its weakness is tree depth, which matters less if you are only validating recovery of a clone partition.

### (b) REAL DEMONSTRATION with many clones → **eTRACER (Zenodo 10.5281/zenodo.19230230)**

This is the only dataset that gives you SpOTr's actual problem *and* its answer key from the same specimens. You get **176–261 per-clone maximum-likelihood scRNA-seq phylogenies** (IQ-TREE2, bootstrapped) as the unmatched single-cell side, **Stereo-seq bin50 count matrices with coordinates** as the spot side, and **55 spatially-resolved per-clone trees whose leaf labels are bin coordinates** as ground truth — with clone IDs shared across the two modalities (7/7, 18/22, 23/24 spatial clones have an scRNA counterpart). Scale is ample: ~2,800 baseline clones, 173–175 with >15 cells, spatial trees up to 6,717 bins. And the biology motivates the method directly: clone spatial aggregation at early escape, plus a monotone clone-distance-to-margin ↔ cell-state relationship.

Two things to accept going in: it is an unreviewed preprint whose numbers shifted between v1 and v2, and **raw reads at NGDC PRJCA040871 are not released** (both reviewer share tokens are dead), so you cannot re-call barcodes from FASTQs. The processed objects are sufficient.

*Runner-up for demonstration:* **KP-Tracer** (Zenodo 10.5281/zenodo.19771805) if you want native tissue architecture and a primary→metastasis story — 124 clonal tumours, 49 arrays, one mouse (SPC-11) carrying 26 serial sections plus LN/rib/diaphragm metastases with a shared phylogeny. But use its **non-imputed** target-site tables: the shipped Slide-seq trees were built with 5 rounds of 30 µm spatial-neighbour imputation, and validating a tree-vs-space method on a tree that was smoothed using space is a circularity you cannot argue your way out of in review.
