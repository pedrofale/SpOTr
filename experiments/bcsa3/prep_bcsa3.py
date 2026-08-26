"""
Build SpOTr inputs for BCSA3 from the joint Numbat run.

SpOTr's deconvolution loss (deconvolution_loss_ls) models

    Y[spot, gene] ~ sum_k prop[spot, k] * clone_cnvs[k, gene] / 2

so Y must be an expression *ratio* (neutral ~ 1) and clone_cnvs a total copy
number (neutral = 2). Everything below is built to satisfy that contract.

Requires bcsa3_ss3_tree_dist.csv (see export_tree.R).
Writes an .npz consumed by run_bcsa3.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import sparse
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
DATA = Path("/mnt/data/pedro/BCSA/BCSA3")
JOINT = DATA / "numbat_out"
SECS = {"B1": "V19T26-031_B1", "D1": "V19T26-031_D1", "C1": "V19T26-031_C1"}
BIOMART = "/home/pedro/projects/cancer/clonal_drug_response/pipeline/ss3/resources/biomart_gene_names.txt"

# Genomic rolling window (genes). Numbat smooths expression this way to make CNV
# signal visible in sparse data; without it per-gene ratios are essentially noise.
ROLL = 101
# genes below this mean CP10K in the reference have too noisy a ratio to use
MIN_REF_CP10K = 0.25
# per-gene ratios above this are dropout artefacts rather than copy number
RATIO_CLIP = 4.0


def _roll_by_chrom(M, chrom, window=ROLL):
    """Rolling mean along genomic order, restarting at each chromosome."""
    out = np.empty_like(M, dtype=np.float32)
    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        blk = pd.DataFrame(M[:, idx].T)
        sm = blk.rolling(window, center=True, min_periods=1).mean().to_numpy().T
        out[:, idx] = sm
    return out


def main(section="B1", out=None, n_genes=0):
    out = Path(out or HERE / f"bcsa3_{section}_inputs.npz")

    # ---- Numbat clone calls (same source the baselines and readiness used) ----
    cp = pd.read_csv(JOINT / "clone_post_2.tsv", sep="\t")
    meta = pd.read_csv(JOINT / "cell_metadata.csv")
    nb = cp.merge(meta, on="cell")
    clone_cols = [c for c in cp.columns if c.startswith("p_") and c[2:].isdigit()]

    # ---- gene order + per-clone CNV from the Numbat bulk profiles ----
    bulk = pd.read_csv(JOINT / "bulk_clones_final.tsv.gz", sep="\t", low_memory=False)
    bulk = bulk[bulk["gene"].notna()]
    gpos = (bulk[["gene", "CHROM", "gene_start"]].drop_duplicates("gene")
            .set_index("gene").sort_values(["CHROM", "gene_start"]))
    # copy number = 2 * expression fold change, clipped to a sane range
    cn = (bulk.pivot_table(index="sample", columns="gene", values="phi_mle", aggfunc="first")
          .mul(2.0).clip(0.2, 6.0))

    # ---- tree distances among SS3 cells (Numbat cell-level phylogeny) ----
    td = pd.read_csv(HERE / "bcsa3_ss3_tree_dist.csv", index_col=0)

    # ---- SS3 reference (two plates) ----
    mats = []
    for plate in ["BCSA3_1", "BCSA3_2"]:
        df = pd.read_csv(DATA / f"{plate}.dgecounts.csv", index_col=0)
        df.columns = [f"{plate}_{c}" for c in df.columns]
        mats.append(df)
    umi = pd.concat(mats, axis=1).fillna(0)
    bm = (pd.read_csv(BIOMART, sep="\t", index_col=1)[["Gene stable ID", "Gene name"]]
          .drop_duplicates().set_index("Gene stable ID"))
    sc_ad = sc.AnnData(umi.T)
    sc_ad.X = sparse.csr_matrix(sc_ad.X)
    sc_ad.var["gene_name"] = bm.loc[bm.index.isin(sc_ad.var.index), "Gene name"]
    sc_ad = sc_ad[:, sc_ad.var["gene_name"].notna()].copy()
    sc_ad.var = sc_ad.var.reset_index(names="ENSG").set_index("gene_name")
    sc_ad.var_names_make_unique()
    ss = nb[nb["source"] == "SS3"].set_index("cell")
    sc_ad.obs["clone"] = ss["clone_opt"].reindex(sc_ad.obs_names).astype("str")

    # keep only cells that have BOTH a clone call and a tree position, in one order
    keep = [c for c in sc_ad.obs_names
            if c in td.index and str(sc_ad.obs["clone"].get(c)) not in ("nan", "None")]
    sc_ad = sc_ad[keep].copy()
    C_tree = td.loc[keep, keep].to_numpy(dtype=np.float32)

    # ---- Visium section ----
    st = sq.read.visium(DATA / SECS[section] / "outs")
    st.var_names_make_unique()
    v = nb[nb["source"] == f"Visium_{section}"].copy()
    v["bc"] = v["cell"].str.replace(f"^{section}_", "", regex=True)
    v = v.set_index("bc")
    st.obs["clone_numbat"] = v["clone_opt"].reindex(st.obs_names).astype("str")
    st = st[st.obs["clone_numbat"].notna() & (st.obs["clone_numbat"] != "nan")].copy()
    soft = v[clone_cols].reindex(st.obs_names)
    soft.columns = [c[2:] for c in clone_cols]

    # ---- shared, CNV-informative, genomically ordered gene set ----
    genes = [g for g in gpos.index
             if g in set(sc_ad.var_names) and g in set(st.var_names) and g in cn.columns]
    if n_genes:  # optional thinning for quick runs; keeps genomic spread
        genes = genes[:: max(1, len(genes) // n_genes)]
    sc_ad = sc_ad[:, genes].copy()
    st = st[:, genes].copy()
    chrom = gpos.loc[genes, "CHROM"].to_numpy()

    # ---- expression ratios (neutral ~ 1) ----
    sc.pp.normalize_total(sc_ad, target_sum=1e4)
    sc.pp.normalize_total(st, target_sum=1e4)
    Xsc = np.asarray(sc_ad.X.todense(), dtype=np.float32)
    Xst = np.asarray(st.X.todense(), dtype=np.float32)

    # reference = SS3 normal cells (Numbat clone 1); fall back to all cells
    normal = (sc_ad.obs["clone"] == "1").to_numpy()
    ref = Xsc[normal].mean(0) if normal.sum() >= 10 else Xsc.mean(0)

    # Genes too quiet in the reference give meaningless ratios (a near-zero
    # denominator sends a handful of genes to 100x+ and swamps the loss), so
    # restrict to genes the reference actually expresses.
    expressed = (ref >= MIN_REF_CP10K) & (Xst.mean(0) >= MIN_REF_CP10K)
    genes = [g for g, k in zip(genes, expressed) if k]
    chrom = chrom[expressed]
    sc_ad = sc_ad[:, genes].copy()
    st = st[:, genes].copy()
    Xsc, Xst, ref = Xsc[:, expressed], Xst[:, expressed], ref[expressed]

    # Clip before smoothing: residual outliers are dropout artefacts, not CNV.
    Rsc = _roll_by_chrom(np.clip(Xsc / ref, 0.0, RATIO_CLIP), chrom)
    Rst = _roll_by_chrom(np.clip(Xst / ref, 0.0, RATIO_CLIP), chrom)

    # The loss predicts a neutral level of 1 (clone_cnvs/2), so put every cell
    # and spot on that scale: dropout depresses the raw ratio by a per-library
    # factor that has nothing to do with copy number.
    Rsc /= np.maximum(np.median(Rsc, axis=1, keepdims=True), 1e-6)
    Rst /= np.maximum(np.median(Rst, axis=1, keepdims=True), 1e-6)

    clones = sorted(cn.index.astype(int))
    Ccn = _roll_by_chrom(cn.loc[clones, genes].to_numpy(dtype=np.float32), chrom)

    # ---- clone assignments for the SS3 cells (one-hot on Numbat clone_opt) ----
    cell_clone = sc_ad.obs["clone"].astype(int).to_numpy()
    A = np.zeros((sc_ad.n_obs, len(clones)), dtype=np.float32)
    for i, c in enumerate(cell_clone):
        A[i, clones.index(c)] = 1.0

    # ---- spatial distances ----
    xy = np.asarray(st.obsm["spatial"], dtype=np.float32)
    C_space = cdist(xy, xy).astype(np.float32)

    # The GW term aligns (C_tree[i,i'] - C_space[j,j'])^2 directly, and
    # prepare_ot_inputs normalises only C_feature. Visium coordinates are in
    # pixels (~1e4) while tree distances are ~1e1, so without this the spatial
    # term outweighs the tree by ~500x and the phylogeny is ignored.
    C_space = C_space / max(C_space.max(), 1e-12)
    C_tree = C_tree / max(C_tree.max(), 1e-12)

    np.savez_compressed(
        out,
        R_sc=Rsc, R_st=Rst, clone_cnvs=Ccn, clone_assignments=A,
        C_tree=C_tree, C_space=C_space, xy=xy,
        cells=np.array(sc_ad.obs_names), spots=np.array(st.obs_names),
        genes=np.array(genes), chrom=chrom, clones=np.array(clones),
        cell_clone=cell_clone,
        spot_clone_numbat=st.obs["clone_numbat"].astype(int).to_numpy(),
        spot_soft=soft.reindex(columns=[str(c) for c in clones]).to_numpy(dtype=np.float32),
    )
    print(f"[prep] {section}: {Rsc.shape[0]} cells x {Rst.shape[0]} spots x {len(genes)} genes")
    print(f"[prep] SS3 clones: {pd.Series(cell_clone).value_counts().sort_index().to_dict()}")
    print(f"[prep] ratio sc  min {Rsc.min():.2f} med {np.median(Rsc):.2f} max {Rsc.max():.2f}")
    print(f"[prep] ratio st  min {Rst.min():.2f} med {np.median(Rst):.2f} max {Rst.max():.2f}")
    print(f"[prep] clone_cnvs range {Ccn.min():.2f}-{Ccn.max():.2f}")
    print(f"[prep] tree dist range {C_tree.min():.2f}-{C_tree.max():.2f}")
    print(f"[prep] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="B1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-genes", type=int, default=0, dest="n_genes")
    main(**vars(ap.parse_args()))
