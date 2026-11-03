"""HT260C1, rebuilt (Pedro, 2026-10-31: genes that separate the annotated types; B's subclones; "go ahead with everything else").

Genes.  The genes shared by the nuclei and the section are split at random in two halves (seed 0).  SpOTr and every method fit genes of the
first half chosen by cell2location's filter_genes rule on the nuclei (detected in > 3% of nuclei, or in > 5 nuclei with a mean non-zero count
> 1.12; the cutoffs of 355_), of which 10% are held out as before; the copy-number references use only the second half, so no method sees them.
(The earlier problem fitted the nuclei's 2,000 highly variable genes, which among nuclei that are 95% tumour hold four stromal markers.)

Subclones.  A and B as before (HT260C1_sn_assign_k2.tsv).  B's nuclei are split into B1 and B2 along B's own copy-number axis (360_: the first
principal component of their arm scores, log UMI regressed out; chr4 / 5q up and chr3 / 6p down in B1), by a two-component Gaussian mixture
(every B nucleus to its more likely component).  The cells' copy-number distances (Euclidean, over the standardised depth-regressed arm scores
of the traced nuclei) give the lineage arm's tree over single nuclei.

References (second-half genes, the section's spots with the authors' purity >= its median, as before): ref_ba, the B-minus-A copy-number
signal (projection on the nuclei's k = 2 split effect, + = B); ref_wb, B's within-clone signal (projection on B's axis, + = B1), defined on
the spots where ref_ba > 0.

Usage: python 362_htan_v2_prep.py   (benchmark_env; writes data/htan/HT260C1-Th1K1Fc2U1Z1Bs1_v2.npz and _v2_reference.tsv)
"""
import sys, gzip, json
from pathlib import Path
import numpy as np, pandas as pd, scipy.io, scipy.sparse
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; SCR = HERE.parent / "htan_screen"; RAW = SCR / "raw" / "HT260C1"
CASE, SEC = "HT260C1", "HT260C1-Th1K1Fc2U1Z1Bs1"; OUTD = ROOT / "data" / "htan"; SEED, MIN_SPOT = 0, 200
CEN = {"1": 123.4, "2": 93.9, "3": 90.9, "4": 50.0, "5": 48.8, "6": 59.8, "7": 60.1, "8": 45.2, "9": 43.0, "10": 39.8, "11": 53.4, "12": 35.5,
       "13": 17.7, "14": 17.2, "15": 19.0, "16": 36.8, "17": 25.1, "18": 18.5, "19": 26.2, "20": 28.1, "21": 12.0, "22": 15.0}


def read10x(prefix):
    M = scipy.sparse.csr_matrix(scipy.io.mmread(gzip.open(f"{prefix}-matrix.mtx.gz")).T)
    f = pd.read_csv(f"{prefix}-features.tsv.gz", sep="\t", header=None); b = pd.read_csv(f"{prefix}-barcodes.tsv.gz", header=None)[0].to_numpy()
    if f.shape[1] > 2:
        g = (f[2] == "Gene Expression").to_numpy(); M, f = M[:, g], f[g].reset_index(drop=True)
    return M, f, b


def c2l_filter(X, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12):
    """cell2location.utils.filtering.filter_genes's rule (0.1.5), without its plot"""
    n = np.asarray((X > 0).sum(0)).ravel().astype(float); nz = np.asarray(X.sum(0)).ravel() / np.maximum(n, 1.0)
    with np.errstate(divide="ignore"):
        return (np.log10(n) > np.log10(X.shape[0] * cell_percentage_cutoff2)) | ((np.log10(n) > np.log10(cell_count_cutoff)) & (np.log10(nz) > np.log10(nonz_mean_cutoff)))


def b_axis(bs, cl):
    """B's within-clone copy-number axis (360_), each B nucleus's position on it, B1 / B2, and the traced nuclei's standardised arm scores"""
    from sklearn.mixture import GaussianMixture
    S = pd.read_csv(SCR / f"{CASE}_sn_own_arm_scores.tsv", sep="\t", index_col=0); asg = pd.read_csv(SCR / f"{CASE}_sn_assign_k2.tsv", sep="\t").set_index("barcode")
    tr = np.isin(cl, ["A", "B"]); idx = bs[tr]; X = S.reindex(idx).to_numpy(float); u = np.log(asg.loc[idx, "umi"].to_numpy(float))
    Dm = np.column_stack([np.ones(len(u)), u]); X = X - Dm @ np.linalg.lstsq(Dm, X, rcond=None)[0]; Z = (X - X.mean(0)) / X.std(0)
    b = cl[tr] == "B"; ZB = (Z[b] - Z[b].mean(0)) / Z[b].std(0); vt = np.linalg.svd(ZB, full_matrices=False)[2][0]
    if vt[list(S.columns).index("chr4p")] < 0:
        vt = -vt                                                                                    # + = chr4 up (B1)
    y = ZB @ vt; g = GaussianMixture(2, n_init=5, random_state=0).fit(y[:, None]); hi = int(np.argmax(g.means_.ravel())); post = g.predict_proba(y[:, None])[:, hi]
    sub = np.where(post >= 0.5, "B1", "B2"); return dict(Z=Z, arms=list(S.columns), load=pd.Series(vt, index=S.columns), y=y, post=post, sub=sub, traced_idx=np.where(tr)[0])


def main():
    import scanpy as sc, anndata as ad
    rng = np.random.default_rng(SEED)
    Xs, fs, bs = read10x(RAW / "HT260C1-Th1K1"); ids_s, names_s = fs[0].to_numpy(), fs[1].to_numpy(); Xv, fv, bv = read10x(RAW / SEC); ids_v = fv[0].to_numpy()
    meta = pd.read_csv(SCR / f"{CASE}_sn_meta.tsv", sep="\t").set_index("Original_barcode"); asg = pd.read_csv(SCR / f"{CASE}_sn_assign_k2.tsv", sep="\t").set_index("barcode")
    keep = np.isin(bs, meta.index[meta.cell_type != "Low quality"]); Xs, bs = Xs[keep], bs[keep]; ct = meta.loc[bs, "cell_type"].to_numpy(); ctype = np.where(ct == "Tumor", "tumour", ct)
    pos = pd.read_csv(RAW / f"{SEC}-tissue_positions_list.csv", header=None, names=["bc", "in", "r", "c", "py", "px"]).set_index("bc").reindex(bv)
    um = 55.0 / json.load(open(RAW / f"{SEC}-scalefactors_json.json"))["spot_diameter_fullres"]; stm = pd.read_csv(SCR / f"{SEC}_st_meta.tsv", sep="\t").set_index("barcode").reindex(bv)
    shared = [g for g in ids_v if g in set(ids_s)]; js = {g: i for i, g in enumerate(ids_s)}; jv = {g: i for i, g in enumerate(ids_v)}; name = dict(zip(ids_s, names_s))
    Xs_sh = Xs[:, [js[g] for g in shared]].tocsc(); Xv_sh = Xv[:, [jv[g] for g in shared]].tocsc(); col = {g: i for i, g in enumerate(shared)}
    half = rng.permutation(len(shared)); poolF = set(shared[i] for i in half[: len(shared) // 2]); poolR = set(shared) - poolF
    F0 = [g for g in shared if g in poolF]; sel = c2l_filter(Xs_sh[:, [col[g] for g in F0]]); fit_all = [g for g, s_ in zip(F0, sel) if s_]
    ev = sorted(rng.choice(fit_all, size=len(fit_all) // 10, replace=False).tolist()); fitg = [g for g in fit_all if g not in set(ev)]
    intis = (pos["in"].to_numpy() == 1) & stm["Tumor_purity_estimate"].notna().to_numpy()
    kv = intis & (np.asarray(Xv_sh[:, [col[g] for g in fitg]].sum(1)).ravel() >= MIN_SPOT); Xv_k, bv_k = Xv_sh[kv], bv[kv]; xy = pos.loc[bv_k, ["px", "py"]].to_numpy(float) * um
    seen = np.asarray(Xv_k.sum(0)).ravel() > 0; fitg, ev = [g for g in fitg if seen[col[g]]], [g for g in ev if seen[col[g]]]; F, V = [col[g] for g in fitg], [col[g] for g in ev]
    cl = asg["clone"].reindex(bs).fillna("").to_numpy(); cl = np.where(ctype == "tumour", cl, ""); traced = np.isin(cl, ["A", "B"])
    clone = np.array([{"B": 0, "A": 1}[c] if tr else 2 + (0 if t == "tumour" else 1) for c, tr, t in zip(cl, traced, ctype)])
    bx = b_axis(bs, cl); bsub = np.full(len(bs), "", dtype=object); bpost = np.full(len(bs), np.nan); ti = bx["traced_idx"]; isb = cl[ti] == "B"
    bsub[ti[isb]] = bx["sub"]; bpost[ti[isb]] = bx["post"]; Zt = bx["Z"]; dcn = np.sqrt(((Zt[:, None, :] - Zt[None, :, :]) ** 2).sum(2))
    names_fit = np.array([name[g] for g in fitg]); stromal = [g for g in ["COL1A1", "COL1A2", "COL3A1", "COL6A3", "DCN", "LUM", "FBLN1", "SPARC", "FN1", "TAGLN"] if g in set(names_fit)]
    epi = [g for g in ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20", "CDH17", "CEACAM5", "CEACAM6", "LGALS4", "PHGR1"] if g in set(names_fit)]
    np.savez_compressed(OUTD / f"{SEC}_v2.npz", clone=clone, clone_informative=np.arange(4) < 2, clone_dist=1.0 - np.eye(4), clone_names=np.array(["B", "A", "untraced tumour", "not tumour"]),
                        Xc_fit=Xs_sh[:, F].toarray().astype(np.float32), Ys_fit=Xv_k[:, F].toarray().astype(np.float32), Xc_ev=Xs_sh[:, V].toarray().astype(np.float32), Ys_ev=Xv_k[:, V].toarray().astype(np.float32),
                        lib=np.asarray(Xs.sum(1)).ravel().astype(float), spot_total=np.asarray(Xv[kv].sum(1)).ravel().astype(float), ev_genes=np.array([name[g] for g in ev]), fit_genes=names_fit,
                        cell_type=ctype, xy=xy, barcodes=bs, spot_barcodes=bv_k, sn_clone=cl, traced=traced, b_sub=bsub.astype(str), b_post=bpost, d_cn=dcn.astype(np.float32),
                        traced_order=ti, b_load=bx["load"].to_numpy(), arms=np.array(bx["arms"]))
    print(f"{SEC} v2: nuclei {len(bs)}, traced {traced.sum()} (A {np.sum(cl == 'A')}, B {np.sum(cl == 'B')}: B1 {np.sum(bsub == 'B1')}, B2 {np.sum(bsub == 'B2')}); spots {kv.sum()} of {intis.sum()}; "
          f"shared genes {len(shared)}, fit pool {len(F0)} -> cell2location's filter {len(fit_all)} -> fitted {len(fitg)}, held out {len(ev)}; stromal markers fitted {stromal}; epithelial {epi}")
    # references from the second half of the genes
    gp = pd.read_csv(ROOT / "data" / "b189" / "gene_positions.csv").drop_duplicates("gene_id").set_index("gene_id"); Xr = Xv_sh[kv]; det = np.asarray((Xr > 0).mean(0)).ravel() >= 0.10
    gi = [i for i, g in enumerate(shared) if det[i] and g in poolR and g in gp.index and str(gp.loc[g, "chrom"]) in CEN]
    chs = np.array([str(gp.loc[shared[i], "chrom"]) for i in gi]); sts = np.array([gp.loc[shared[i], "start"] for i in gi], float)
    o = np.lexsort((sts, chs.astype(int))); gi = [gi[k] for k in o]; chs, sts = chs[o], sts[o]; arm = np.array([f"chr{c}{'p' if s < CEN[c] * 1e6 else 'q'}" for c, s in zip(chs, sts)])
    pur = stm.loc[bv_k, "Tumor_purity_estimate"].to_numpy(float)
    L = np.log1p(np.asarray(Xr[:, gi].todense(), float) / np.asarray(Xv[kv].sum(1), float) * 1e4); L -= L[pur <= np.percentile(pur, 20)].mean(0)
    for c in np.unique(chs):
        j = chs == c; L[:, j] = uniform_filter1d(L[:, j], size=min(101, j.sum()), axis=1, mode="nearest")
    L -= np.median(L, axis=1, keepdims=True)
    scr = json.load(open(SCR / f"{CASE}_screen.json")); e = pd.Series(scr["splits"]["2"]["pairs"][0]["effect"]); clu = pd.read_csv(SCR / f"{CASE}_sn_clusters.tsv", sep="\t")
    small = int(clu.k2[clu.k2 > 0].value_counts().idxmin()); e = e if small == 2 else -e
    arms = [x for x in pd.unique(arm) if (arm == x).sum() >= 20 and x in e.index]; S = np.stack([L[:, arm == x].mean(1) for x in arms], 1); tum = pur >= np.median(pur); T = S[tum]
    T = T - np.outer(pur[tum] - pur[tum].mean(), np.polyfit(pur[tum], T, 1)[0]); T = (T - T.mean(0)) / T.std(0)
    ev_ = e.reindex(arms).to_numpy(); ref_ba = T @ (ev_ / np.linalg.norm(ev_)); wl = bx["load"].reindex(arms).fillna(0.0).to_numpy(); ref_wb = T @ (wl / np.linalg.norm(wl))
    old = pd.read_csv(OUTD / f"{SEC}_reference.tsv", sep="\t").set_index("barcode")["ref_c2"]; both = old.index.intersection(bv_k[tum])
    rho_old = pd.Series(ref_ba, index=bv_k[tum]).loc[both].corr(old.loc[both], method="spearman")
    pd.DataFrame({"barcode": bv_k[tum], "x_um": xy[tum, 0], "y_um": xy[tum, 1], "purity": pur[tum], "ref_ba": ref_ba, "ref_wb": np.where(ref_ba > 0, ref_wb, np.nan)}).to_csv(OUTD / f"{SEC}_v2_reference.tsv", sep="\t", index=False)
    print(f" references: {len(gi)} second-half genes, {len(arms)} arms, {tum.sum()} tumour spots; B-minus-A against the earlier reference (Spearman) {rho_old:.3f}; B territory {int((ref_ba > 0).sum())} spots")


if __name__ == "__main__":
    main()
