"""HT260C1 (a colorectal cancer liver metastasis, Mo et al. 2024), with the dissociated nuclei, the Visium section, their copy number clones and the
CNA signal of the spots that the scores use. Writes $SPOTR_DATA/htan/sc.h5ad and st.h5ad.

Inputs, in $SPOTR_DATA/htan/raw (see README.md for where to download them):
  HT260C1-Th1K1-{matrix.mtx.gz,features.tsv.gz,barcodes.tsv.gz}              the multiome nuclei (expression and ATAC peaks)
  HT260C1-Th1K1Fc2U1Z1Bs1-{matrix.mtx.gz,features.tsv.gz,barcodes.tsv.gz}    the Visium section
  HT260C1-Th1K1Fc2U1Z1Bs1-tissue_positions_list.csv, -scalefactors_json.json, B1-HT260C1-Th1K1Fc2U1.tif (the H&E)
  HT260C1-Th1K1XBmn1-1_arm_level.tsv                                           the copy number table of the authors for each nucleus
  HT260C1_sn_meta.tsv, HT260C1-Th1K1Fc2U1Z1Bs1_st_meta.tsv                    the cell types and spot purity of the authors (metadata.R)
  gene_positions.csv.gz (in this folder)                                       GRCh38 gene coordinates

The Supplementary Methods of the paper (Clones and tree of HT260C1) describe these steps.

Genes. We split the genes measured in both the nuclei and the section at random into two halves (seed 0). SpOTr and every other method fit the
genes of the first half that the filter_genes rule of cell2location keeps on the nuclei (detected in more than 3% of the nuclei, or in more than
5 nuclei with a mean non-zero count above 1.12), minus a random tenth that we hold out. We compute the CNA signal of the spots on the second half only.
Spots and nuclei. We keep the spots in the tissue that have a tumor purity estimate of the authors (from ESTIMATE) and at least 200 counts on the
fitted genes, and the nuclei that the authors did not annotate as "Low quality".

Clones. We compute the CNA scores of the nuclei (copynumber.py) and call the clones in the following steps.
  1. We regress the log total counts out of the CNA scores of the tumor nuclei, and split the 60% of tumor nuclei with the most counts into two
  groups by Ward clustering.
  2. We select the chromosome arms that differ between the two groups, with an effect of at least 0.5 standard deviations and with most of their
  genes changing in the same direction.
  3. We call each of the 60% of tumor nuclei with the most counts A or B by the multinomial likelihood ratio of its counts on these chromosome
  arms under the two groups, with the proportions of counts estimated from the 40% of nuclei with the most counts, equal priors and a posterior
  of at least 0.9.
  4. We split B into B1 and B2 with a Gaussian mixture model (GMM) with two components along the first principal component of its CNA scores,
  with the log total counts regressed out.
  5. We call the remaining tumor nuclei by adding the evidence of their CNA scores from ATAC, from logistic regressions trained on the called
  nuclei, to the evidence from expression, first for A against B and then for B1 against B2, at a combined posterior of at least 0.9 or at most
  0.1.
  6. We leave out the tumor nuclei in Leiden expression clusters of which at least 30% are non-tumor nuclei, and the tumor nuclei that remain
  uncalled.
  7. We split B1 into the sub-lineages B1a and B1b with a GMM with two components along the first principal component of the unsmoothed CNA
  scores of its nuclei, with the log total counts regressed out. No method is given B1a and B1b.

The CNA profile of each cancer nucleus is its 39 CNA scores with the log total counts regressed out and standardized (sc.obsm["cn_profile"]), and
fit.py builds the tree of SpOTr from the distances between these profiles.

CNA signal of the spots. We compute the CNA scores of the spots on the second half of the genes, with the 20% of spots with the lowest purity as
the reference, and regress the purity out.
  ref_ba         The signal for B against A, the projection on the difference between A and B in the CNA scores of the nuclei (on the spots with
                 a purity above the median).
  ref_wb         The signal within B, the projection on the axis between B1 and B2 (on the spots with ref_ba > 0).
  b1_signal      The signal for B1b against B1a, the projection on the difference between the CNA scores of the sub-lineages.
  b1_territory   The territory of B1, the spots with ref_ba above 0.5 standard deviations and ref_wb > 0.

Usage: python prepare.py [raw folder]   (benchmark_env, CPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import paths  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent)); import copynumber as cn  # noqa: E402

NUCLEI, SECTION = "HT260C1-Th1K1", "HT260C1-Th1K1Fc2U1Z1Bs1"
SEED, MIN_SPOT = 0, 200


def c2l_filter(X, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12):
    """Return the genes that the rule of cell2location.utils.filtering.filter_genes keeps."""
    n = np.asarray((X > 0).sum(0)).ravel().astype(float); nz = np.asarray(X.sum(0)).ravel() / np.maximum(n, 1.0)
    with np.errstate(divide="ignore"):
        return (np.log10(n) > np.log10(X.shape[0] * cell_percentage_cutoff2)) | ((np.log10(n) > np.log10(cell_count_cutoff)) & (np.log10(nz) > np.log10(nonz_mean_cutoff)))


def logit(p):
    return np.log(np.clip(p, 1e-12, 1 - 1e-12)) - np.log(np.clip(1 - p, 1e-12, 1 - 1e-12))


def load(raw):
    """Load the nuclei (expression and peaks), the section, the annotations of the authors and the gene positions."""
    M, f, b = cn.read10x(raw / NUCLEI); gex = (f[2] == "Gene Expression").to_numpy(); pk = (f[2] == "Peaks").to_numpy()
    Mv, fv, bv = cn.read10x(raw / SECTION); gv = (fv[2] == "Gene Expression").to_numpy()
    return dict(G=M[:, gex], gf=f[gex].reset_index(drop=True), P=M[:, pk], pf=f[pk].reset_index(drop=True), b=b, Xv=Mv[:, gv], fv=fv[gv].reset_index(drop=True), bv=bv,
                meta=pd.read_csv(raw / "HT260C1_sn_meta.tsv", sep="\t").set_index("Original_barcode"),
                stm=pd.read_csv(raw / f"{SECTION}_st_meta.tsv", sep="\t").set_index("barcode"),
                tableA=pd.read_csv(raw / f"{NUCLEI}XBmn1-1_arm_level.tsv", sep="\t", index_col=0),
                gp=cn.gene_positions(Path(__file__).resolve().parent / "gene_positions.csv.gz"),
                pos=pd.read_csv(raw / f"{SECTION}-tissue_positions_list.csv", header=None, names=["bc", "in", "r", "c", "py", "px"]).set_index("bc"),
                sf=json.load(open(raw / f"{SECTION}-scalefactors_json.json")), raw=raw)


# ------------------------------------------------------------------------------------------------------------------------- the clones
def profiles(d):
    """Return the smoothed CNA scores of every annotated nucleus (the profiles for the axis of B and for the tree), with the non-tumor nuclei as
    reference.
    """
    rows = np.isin(d["b"], d["meta"].index); ct = d["meta"].loc[d["b"][rows], "cell_type"].to_numpy()
    S = cn.arm_scores(d["G"][rows], d["gf"][0].to_numpy(), d["gp"], ct != "Tumor", 0.05, total="used"); S.index = d["b"][rows]; return S


def nuclei(d):
    """Return the nuclei kept (those that the authors did not annotate as "Low quality"), in the order of the matrix, and their cell types."""
    keep = np.isin(d["b"], d["meta"].index[d["meta"].cell_type != "Low quality"]); b = d["b"][keep]; ct = d["meta"].loc[b, "cell_type"].to_numpy()
    return keep, b, np.where(ct == "Tumor", "tumour", ct)


def split_ab(d):
    """Split the tumor nuclei with the most counts in two by Ward clustering of their CNA scores with depth regressed out, and return the effect (in
    standard deviations, rounded to 3 decimals) and the breadth of each chromosome arm. The breadth is the fraction of the genes of the
    chromosome arm that change in the same direction, for the six chromosome arms with the largest effects.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    keep, b, ct = nuclei(d); X = d["G"][keep]; ids = d["gf"][0].to_numpy(); tum = ct == "tumour"; umi = np.asarray(X.sum(1)).ravel()
    S = cn.arm_scores(X, ids, d["gp"], ~tum, 0.05); S.index = b
    St = S[tum].copy(); lu = np.log(umi[tum]); Rz = St - np.outer(lu - lu.mean(), [np.polyfit(lu, St[a], 1)[0] for a in St.columns])
    deep = lu >= np.quantile(lu, 0.4); lab = np.zeros(len(St), int)
    lab[deep] = fcluster(linkage(((Rz[deep] - Rz[deep].mean()) / Rz[deep].std()).to_numpy(), "ward"), 2, "maxclust")
    e = (St[lab == 2].mean() - St[lab == 1].mean()) / St.std(); top = e.abs().sort_values(ascending=False).index[:6].tolist()
    Xt = X[tum]; detg = np.asarray((Xt > 0).mean(0)).ravel() >= 0.05
    gsel = np.where(detg & np.array([i in d["gp"].index and str(d["gp"].loc[i, "chrom"]) in cn.CENTROMERE_MB for i in ids]))[0]
    Lg = np.log1p(scipy.sparse.diags(1e4 / np.asarray(Xt.sum(1)).ravel()) @ Xt[:, gsel]).tocsr()
    garm = np.array([cn.arm(d["gp"].loc[ids[k], "chrom"], d["gp"].loc[ids[k], "start"]) for k in gsel])
    dg = np.asarray(Lg[lab == 2].mean(0)).ravel() - np.asarray(Lg[lab == 1].mean(0)).ravel()
    breadth = {a: round(float(np.mean(np.sign(dg[garm == a]) == np.sign(dg[garm == a].mean()))), 2) for a in top}
    return dict(barcodes=b[tum], umi=umi[tum], k2=lab, effect=e.round(3), breadth=breadth)


def call_ab(d, sp):
    """Call A or B for every tumor nucleus by the multinomial likelihood ratio of its counts on the differing chromosome arms under the proportions
    of counts of the two groups.
    """
    sizes = pd.Series(sp["k2"][sp["k2"] > 0]).value_counts(); small, large = int(sizes.idxmin()), int(sizes.idxmax())
    eff, brd = sp["effect"], sp["breadth"]
    arms = [a for a in eff.index if abs(eff[a]) >= 0.5 and brd.get(a, 0) >= 0.7] or [a for a in brd if brd[a] >= 0.7]
    t = np.isin(d["b"], sp["barcodes"]); X, b = d["G"][t], d["b"][t]; lab = pd.Series(sp["k2"], index=sp["barcodes"]).reindex(b).to_numpy()
    umi = np.asarray(X.sum(1)).ravel(); ids, names = d["gf"][0].to_numpy(), d["gf"][1].to_numpy(); gp = d["gp"]
    arm = np.array([cn.arm(gp.loc[i, "chrom"], gp.loc[i, "start"]) if i in gp.index and str(gp.loc[i, "chrom"]) in cn.CENTROMERE_MB else "" for i in ids])
    det = np.asarray((X > 0).mean(0)).ravel() >= 0.05; mean_cp = np.asarray((scipy.sparse.diags(1e4 / umi) @ X).mean(0)).ravel()
    hk = pd.Series(names).str.match(r"^(RPL|RPS|MT-|MRPL|MRPS)").to_numpy() | (mean_cp >= np.quantile(mean_cp[det], 0.95))
    use = det & ~hk & np.isin(arm, arms)
    Ma = scipy.sparse.csr_matrix((np.ones(use.sum()), (np.arange(use.sum()), [arms.index(a) for a in arm[use]])), shape=(use.sum(), len(arms)))
    Xa = np.asarray((X[:, use] @ Ma).todense()); n = Xa.sum(1); deep40 = umi >= np.quantile(umi, 0.6)
    sh = lambda m: Xa[m].sum(0) / n[m].sum()
    ll = Xa @ (np.log(sh(deep40 & (lab == small))) - np.log(sh(deep40 & (lab == large)))); w = 1 / (1 + np.exp(-np.clip(ll, -50, 50)))
    clone = np.where(umi < np.quantile(umi, 0.4), "untraced", np.where(w >= 0.9, "B", np.where(w <= 0.1, "A", "untraced")))
    return pd.DataFrame({"barcode": b, "umi": umi, "llr": ll, "post_b": w, "clone": clone}).set_index("barcode")


def split_b(S, asg, b, cl):
    """Split B into B1 and B2 with a GMM with two components along the first principal component of its CNA scores with depth regressed out
    (oriented with chr4p up).
    """
    from sklearn.mixture import GaussianMixture
    tr = np.isin(cl, ["A", "B"]); idx = b[tr]; X = S.reindex(idx).to_numpy(float); u = np.log(asg.loc[idx, "umi"].to_numpy(float))
    Dm = np.column_stack([np.ones(len(u)), u]); X = X - Dm @ np.linalg.lstsq(Dm, X, rcond=None)[0]; Z = (X - X.mean(0)) / X.std(0)
    isb = cl[tr] == "B"; ZB = (Z[isb] - Z[isb].mean(0)) / Z[isb].std(0); vt = np.linalg.svd(ZB, full_matrices=False)[2][0]
    if vt[list(S.columns).index("chr4p")] < 0:
        vt = -vt
    y = ZB @ vt; g = GaussianMixture(2, n_init=5, random_state=0).fit(y[:, None]); post = g.predict_proba(y[:, None])[:, int(np.argmax(g.means_.ravel()))]
    sub = np.full(len(b), "", dtype=object); sub[np.where(tr)[0][isb]] = np.where(post >= 0.5, "B1", "B2")
    return sub.astype(str), pd.Series(vt, index=S.columns)


def expression_clusters_mixed(Xc, ct):
    """Return the tumor nuclei in Leiden clusters (resolution 1 on 2,000 highly variable genes, 30 principal components and 15 neighbors) of which
    at least 30% are non-tumor nuclei.
    """
    import scanpy as sc, anndata as ad
    a = ad.AnnData(np.array(Xc, np.float32, copy=True)); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000); a = a[:, a.var["highly_variable"]].copy()
    sc.pp.pca(a, n_comps=30, random_state=0); sc.pp.neighbors(a, n_neighbors=15, random_state=0); sc.tl.leiden(a, resolution=1.0, random_state=0)
    ld = a.obs["leiden"].astype(str).to_numpy(); me = pd.Series(ct != "tumour").groupby(ld).mean()
    return np.isin(ld, me.index[me >= 0.3]) & (ct == "tumour")


def call_rest(d, S, asg, b, ct, lab, drop):
    """Call the remaining tumor nuclei by adding the evidence from ATAC to that from expression, first for A against B and then for B1 against B2
    (combined posterior >= 0.9 or <= 0.1).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    tum = (ct == "tumour") & ~drop; tb = b[tum]; lab = lab[tum]; a = asg.reindex(tb); umi = a["umi"].to_numpy(float)
    rows = {x: i for i, x in enumerate(d["b"])}; allb = d["tableA"].index.to_numpy()
    cta = d["meta"]["cell_type"].reindex(allb); L, tot = cn.atac_arms(d["P"][[rows[x] for x in allb]], d["pf"], (cta != "Tumor").to_numpy())
    L.index = allb; Lt = L.reindex(tb); tt = pd.Series(tot, index=allb).reindex(tb).to_numpy(float); ok = Lt.notna().all(1).to_numpy()
    Za = np.full((len(tb), L.shape[1]), np.nan); Za[ok] = cn.regress_out(Lt[ok].to_numpy(), np.log(np.maximum(tt[ok], 1.0)))
    called = lab != ""; un = ~called; trn = called & ok; y = (lab[trn] != "A").astype(int)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(Za[trn], y); pa = np.full(len(tb), np.nan); pa[ok] = clf.predict_proba(Za[ok])[:, 1]
    with np.errstate(invalid="ignore"):
        pc = 1 / (1 + np.exp(-np.clip(a["llr"].to_numpy(float) + logit(pa) - np.log(y.mean() / (1 - y.mean())), -50, 50)))
        cAB = np.where(un & (pc >= 0.9), "B", np.where(un & (pc <= 0.1), "A", ""))
    St = S.reindex(tb); okr = St.notna().all(1).to_numpy(); oldb = np.isin(lab, ["B1", "B2"])
    isB = (oldb | (cAB == "B")) & okr; Zr = cn.regress_out(St[isB].to_numpy(), np.log(umi[isB])); old = oldb[isB]
    vt = np.linalg.svd(Zr[old] - Zr[old].mean(0), full_matrices=False)[2][0]; vt = vt if vt[list(S.columns).index("chr4p")] > 0 else -vt; yb_ = Zr @ vt
    g = GaussianMixture(2, n_init=5, random_state=0).fit(yb_[old][:, None]); pr1 = g.predict_proba(yb_[:, None])[:, int(np.argmax(g.means_.ravel()))]
    trb = oldb & okr & ok; yb = (lab[trb] == "B1").astype(int); clfB = LogisticRegression(max_iter=3000, C=0.1).fit(Za[trb], yb)
    iB = np.where(isB)[0]; pcb = 1 / (1 + np.exp(-np.clip(logit(pr1) + logit(clfB.predict_proba(Za[iB])[:, 1]) - np.log(yb.mean() / (1 - yb.mean())), -50, 50)))
    sub = np.where(pcb >= 0.9, "B1", np.where(pcb <= 0.1, "B2", "")); new = np.where(cAB == "A", "A", "").astype(object); new[iB[~old]] = sub[~old]
    return pd.DataFrame({"barcode": tb, "label": lab, "new_label": new.astype(str)}).set_index("barcode")


def sublineages(d, b1, gp_arms_peaks):
    """Split B1 into B1a and B1b with a GMM with two components along the first principal component of the unsmoothed CNA scores of its nuclei, with
    depth regressed out.

    Returns the label of each nucleus of B1 (1 for B1b) and the difference between the CNA scores of the sub-lineages (B1b minus B1a).
    """
    from sklearn.mixture import GaussianMixture
    keep, b, _ = nuclei(d); ct = d["meta"].loc[b, "cell_type"].to_numpy()
    R, umi = cn.arm_means(d["G"][keep], d["gf"][0].to_numpy(), d["gp"], ct != "Tumor"); R.index = b
    arms = [a for a in R.columns if a in gp_arms_peaks]; m = np.isin(b, b1)
    Z = cn.regress_out(R.loc[b[m], arms].to_numpy(), np.log(np.maximum(umi[m], 1))); vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    yr = (Z - Z.mean(0)) @ vt; h = GaussianMixture(2, n_init=5, random_state=0).fit(yr[:, None]).predict(yr[:, None])
    return pd.Series(h, index=b[m]), pd.Series(Z[h == 1].mean(0) - Z[h == 0].mean(0), index=arms)


# ------------------------------------------------------------------------------------------------------------------------- assembly
def main(raw=None):
    import anndata as ad
    raw = Path(raw) if raw else paths.DATA / "htan" / "raw"; out = paths.DATA / "htan"; out.mkdir(parents=True, exist_ok=True)
    d = load(raw); gp = d["gp"]; rng = np.random.default_rng(SEED)
    # genes and spots
    keep, bs, ctype = nuclei(d); Xs = d["G"][keep]; ids_s, names_s = d["gf"][0].to_numpy(), d["gf"][1].to_numpy(); Xv, bv = d["Xv"], d["bv"]; ids_v = d["fv"][0].to_numpy()
    pos = d["pos"].reindex(bv); um = 55.0 / d["sf"]["spot_diameter_fullres"]; stm = d["stm"].reindex(bv)
    shared = [g for g in ids_v if g in set(ids_s)]; js = {g: i for i, g in enumerate(ids_s)}; jv = {g: i for i, g in enumerate(ids_v)}; name = dict(zip(ids_s, names_s))
    Xs_sh = Xs[:, [js[g] for g in shared]].tocsc(); Xv_sh = Xv[:, [jv[g] for g in shared]].tocsc(); col = {g: i for i, g in enumerate(shared)}
    half = rng.permutation(len(shared)); poolF = set(shared[i] for i in half[: len(shared) // 2]); poolR = set(shared) - poolF
    F0 = [g for g in shared if g in poolF]; fit_all = [g for g, s_ in zip(F0, c2l_filter(Xs_sh[:, [col[g] for g in F0]])) if s_]
    held = set(rng.choice(fit_all, size=len(fit_all) // 10, replace=False).tolist()); fitg = [g for g in fit_all if g not in held]
    intis = (pos["in"].to_numpy() == 1) & stm["Tumor_purity_estimate"].notna().to_numpy()
    kv = intis & (np.asarray(Xv_sh[:, [col[g] for g in fitg]].sum(1)).ravel() >= MIN_SPOT); Xv_k, sb = Xv_sh[kv], bv[kv]
    xy = pos.loc[sb, ["px", "py"]].to_numpy(float) * um; seen = np.asarray(Xv_k.sum(0)).ravel() > 0; fitg = [g for g in fitg if seen[col[g]]]
    Fi = [col[g] for g in fitg]; Xc = Xs_sh[:, Fi].toarray().astype(np.float32); Y = Xv_k[:, Fi].toarray().astype(np.float32)
    print(f"nuclei {len(bs)}, spots {kv.sum()}, fitted genes {len(fitg)}", flush=True)
    # clones A / B, B1 / B2
    S = profiles(d); sp = split_ab(d); asg = call_ab(d, sp)
    cl = np.where(ctype == "tumour", asg["clone"].reindex(bs).fillna("").to_numpy(), ""); bsub, bload = split_b(S, asg, bs, cl)
    lab = np.where(cl == "A", "A", np.where(bsub == "B1", "B1", np.where(bsub == "B2", "B2", "")))
    # the CNA signal of the spots
    pur = stm.loc[sb, "Tumor_purity_estimate"].to_numpy(float)
    Sv = cn.arm_scores(Xv[kv], ids_v, gp, pur <= np.percentile(pur, 20), 0.10, allowed=poolR)
    small = int(pd.Series(sp["k2"][sp["k2"] > 0]).value_counts().idxmin()); e = sp["effect"] if small == 2 else -sp["effect"]
    arms = [a for a in Sv.columns if a in e.index]; tum = pur >= np.median(pur); T = Sv[arms].to_numpy()[tum]
    T = T - np.outer(pur[tum] - pur[tum].mean(), np.polyfit(pur[tum], T, 1)[0]); T = (T - T.mean(0)) / T.std(0)
    ev_ = e.reindex(arms).to_numpy(); wl = bload.reindex(arms).fillna(0.0).to_numpy()
    ref_ba = np.full(len(sb), np.nan); ref_ba[tum] = T @ (ev_ / np.linalg.norm(ev_))
    ref_wb = np.full(len(sb), np.nan); ref_wb[tum] = T @ (wl / np.linalg.norm(wl)); ref_wb[~(ref_ba > 0)] = np.nan
    # the remaining tumor nuclei, then the nuclei kept
    drop = expression_clusters_mixed(Xc, ctype); calls = call_rest(d, S, asg, bs, ctype, lab, drop)
    final = pd.Series(np.where(calls["label"] != "", calls["label"], calls["new_label"]), index=calls.index).reindex(bs).fillna("").to_numpy().astype(str)
    kn = (ctype != "tumour") | (np.isin(bs, calls.index) & (final != "")); print(f"dropped {int((~kn).sum())} tumor nuclei; kept {kn.sum()}", flush=True)
    # the sub-lineages of B1 and their signal in the spots
    h, sig = sublineages(d, bs[bsub == "B1"], set(cn.peak_arms(d["pf"])[1]))
    Sa = Sv.to_numpy().copy(); Sa = Sa - np.outer(pur - pur.mean(), np.polyfit(pur, Sa, 1)[0]); Sa = (Sa - Sa.mean(0)) / Sa.std(0); Sa = pd.DataFrame(Sa, columns=Sv.columns)
    a_ = [a for a in sig.index if a in Sa.columns]; b1_signal = Sa[a_].to_numpy() @ (sig[a_].to_numpy() / np.linalg.norm(sig[a_].to_numpy()))
    ok = np.isfinite(ref_ba); zz = np.full(len(sb), np.nan); zz[ok] = (ref_ba[ok] - ref_ba[ok].mean()) / ref_ba[ok].std()
    with np.errstate(invalid="ignore"):
        terr = ok & (zz > 0.5) & np.isfinite(ref_wb) & (ref_wb > 0)
    # the CNA profiles of the cancer nuclei
    kb, kct, kl = bs[kn], ctype[kn], final[kn]; tr = kct == "tumour"; ti = np.where(tr)[0]
    X = S.reindex(kb[ti]).to_numpy(float); u = np.log(asg.loc[kb[ti], "umi"].to_numpy(float)); Dm = np.column_stack([np.ones(len(u)), u])
    X = X - Dm @ np.linalg.lstsq(Dm, X, rcond=None)[0]; prof = np.full((len(kb), X.shape[1]), np.nan); prof[ti] = (X - X.mean(0)) / X.std(0)
    sub_l = np.array([("B1b" if h[x] == 1 else "B1a") if (l == "B1" and x in h.index) else "" for x, l in zip(kb, kl)])
    # AnnData
    var = lambda: pd.DataFrame({"gene_name": [name[g] for g in fitg]}, index=pd.Index(fitg, name="gene_id"))
    cat = lambda v, m, cats: pd.Categorical(pd.Series(v).where(m).to_numpy(), categories=cats)
    obs = pd.DataFrame({"cell_type": kct, "clone": cat(kl, tr, ["A", "B1", "B2"]), "sublineage": cat(sub_l, sub_l != "", ["B1a", "B1b"])},
                       index=pd.Index(kb, name="barcode"))
    sc_ = ad.AnnData(scipy.sparse.csr_matrix(Xc[kn]), obs=obs, var=var()); sc_.obsm["cn_profile"] = prof.astype(np.float64); sc_.uns["cn_arms"] = np.array(S.columns)
    from PIL import Image
    import tifffile
    Image.MAX_IMAGE_PIXELS = None; img = np.asarray(Image.fromarray(tifffile.imread(raw / "B1-HT260C1-Th1K1Fc2U1.tif")).reduce(10))
    so = pd.DataFrame({"purity": pur, "ref_ba": ref_ba, "ref_wb": ref_wb, "b1_signal": b1_signal, "b1_territory": terr}, index=pd.Index(sb, name="barcode"))
    st_ = ad.AnnData(scipy.sparse.csr_matrix(Y), obs=so, var=var()); st_.obsm["spatial"] = xy
    st_.uns["spatial"] = {SECTION: dict(images=dict(lowres=img), scalefactors=dict(tissue_lowres_scalef=0.1, spot_diameter_fullres=float(d["sf"]["spot_diameter_fullres"])))}
    st_.uns["um_per_pixel"] = um
    sc_.write_h5ad(out / "sc.h5ad", compression="gzip"); st_.write_h5ad(out / "st.h5ad", compression="gzip")
    print(f"wrote {out}/sc.h5ad ({sc_.n_obs} nuclei: " + ", ".join(f"{k} {int((kl[tr] == k).sum())}" for k in ("A", "B1", "B2"))
          + f"; B1a {int((sub_l == 'B1a').sum())}, B1b {int((sub_l == 'B1b').sum())}) and st.h5ad ({st_.n_obs} spots, {int(terr.sum())} in the territory of B1)")
    return dict(Xc=Xc, Y=Y, xy=xy, bs=bs, ctype=ctype, sb=sb, fitg=fitg, S=S, sp=sp, asg=asg, bsub=bsub, bload=bload, ref_ba=ref_ba, ref_wb=ref_wb,
                drop=drop, calls=calls, kn=kn, h=h, sig=sig, b1_signal=b1_signal, terr=terr, prof=prof, sub_l=sub_l, final=final)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main(*sys.argv[1:2])
