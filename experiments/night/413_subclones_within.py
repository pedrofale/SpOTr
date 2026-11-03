"""Subclonal structure inside the main clones, in both HTAN cases, from the nuclei's RNA and ATAC (Pedro, 2026-11-01: "it looks like in both of these
samples the two main clones are clearly spatially segregated in terms of glands. I wonder if we can find evidence in the snRNA / snATAC data for
subclonal structure inside these main clones?").  383_ did this for HT260C1 (B splits into B1 / B2, confirmed; A shows nothing); here both cases,
the same way, one level deeper as well.

Per nucleus (multiome): RNA arm scores (mean log normalised expression of each arm's genes detected in >= 5% of nuclei, centred on the non-tumour
nuclei, log UMI regressed out) and ATAC arm scores (each arm's share of the nucleus's peak counts, log2 ratio to the non-tumour nuclei, log total
regressed out).  For each group (main clone A, B; then HT260C1's B1, B2; HT268B1's A1, A2 from the screen's k = 3 split): within the group,
standardised, the first principal component's share of variance in each modality against 20 permutations of each arm across nuclei; two groups or
one along it (Gaussian mixture BIC gain against the null's); the per-nucleus correlation of the ATAC and RNA axes and the correlation of their arm
loadings (|r|).  For HT268B1's A1 / A2: cross-validated AUC of the ATAC arm profile for the RNA split, against 5 label permutations.
A sub-split is supported when the ATAC axis beats its null (p <= 0.05) and agrees with the RNA axis (per nucleus |r| >= 0.3, loadings |r| >= 0.5).
Stronger, as 383_ (an ATAC main axis can be technical and hide shared structure): ATAC projected on the RNA axis's arm weights, per nucleus, against
the weights shuffled across arms (200 times); and the RNA axis split in two (Gaussian mixture), the halves' arm differences in ATAC against RNA,
against arms shuffled -- supported when both p <= 0.01 and the arm-effect |r| >= 0.5.

Usage: python 413_subclones_within.py   (benchmark_env; CPU; writes spotr_out/subclones_within.json)
"""
import gzip, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, scipy.io, scipy.sparse

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; SCR = HERE.parent / "htan_screen"; RAWD = SCR / "raw"; O = HERE / "spotr_out"
CASES = {"HT260C1": "HT260C1-Th1K1", "HT268B1": "HT268B1-Th1H3"}


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load(case):
    m362 = mod("m362", "362_htan_v2_prep.py"); CEN = m362.CEN; P = RAWD / case / CASES[case]
    M = scipy.sparse.csr_matrix(scipy.io.mmread(gzip.open(f"{P}-matrix.mtx.gz")).T); f = pd.read_csv(f"{P}-features.tsv.gz", sep="\t", header=None); b = pd.read_csv(f"{P}-barcodes.tsv.gz", header=None)[0].to_numpy()
    meta = pd.read_csv(SCR / f"{case}_sn_meta.tsv", sep="\t").set_index("Original_barcode"); keep = np.isin(b, meta.index[meta.cell_type != "Low quality"]); M, b = M[keep], b[keep]
    ct = meta.loc[b, "cell_type"].to_numpy(); asg = pd.read_csv(SCR / f"{case}_sn_assign_k2.tsv", sep="\t").set_index("barcode"); clone = asg["clone"].reindex(b).fillna("").to_numpy()
    clu = pd.read_csv(SCR / f"{case}_sn_clusters.tsv", sep="\t").set_index("barcode"); k3 = clu["k3"].reindex(b).fillna(0).astype(int).to_numpy()
    # RNA arm scores
    ge = (f[2] == "Gene Expression").to_numpy(); G = M[:, ge].tocsc(); gid = f[0][ge].to_numpy(); umi = np.asarray(G.sum(1)).ravel().astype(float)
    gp = pd.read_csv(ROOT / "data" / "b189" / "gene_positions.csv").drop_duplicates("gene_id").set_index("gene_id"); det = np.asarray((G > 0).mean(0)).ravel() >= 0.05
    arm_g = np.array([(f"chr{gp.loc[g, 'chrom']}{'p' if gp.loc[g, 'start'] < CEN[str(gp.loc[g, 'chrom'])] * 1e6 else 'q'}" if (g in gp.index and str(gp.loc[g, "chrom"]) in CEN and d) else "") for g, d in zip(gid, det)])
    Lg = G.multiply(1.0 / np.maximum(umi, 1)[:, None]).tocsc() * 1e4; Lg.data = np.log1p(Lg.data); arms_r = [a for a in pd.unique(arm_g) if a and (arm_g == a).sum() >= 20]
    R = np.column_stack([np.asarray(Lg[:, np.where(arm_g == a)[0]].mean(1)).ravel() for a in arms_r])
    # ATAC arm scores
    pk = (f[2] == "Peaks").to_numpy(); A_ = M[:, pk].tocsc(); fp = f[pk].reset_index(drop=True); chrom = fp[3].astype(str).str.replace("chr", ""); start = fp[4].to_numpy(float)
    arm_p = np.array([f"chr{c}{'p' if s < CEN[c] * 1e6 else 'q'}" if c in CEN else "" for c, s in zip(chrom, start)]); tot = np.asarray(A_.sum(1)).ravel().astype(float)
    arms_a = [a for a in pd.unique(arm_p) if a and (arm_p == a).sum() >= 200]; F = np.column_stack([np.asarray(A_[:, np.where(arm_p == a)[0]].sum(1)).ravel() for a in arms_a]) / np.maximum(tot[:, None], 1.0)
    norm = ct != "Tumor"
    Rn = R - R[norm].mean(0); An = np.log2(np.maximum(F / np.maximum(F[norm].mean(0), 1e-12), 1e-3))
    return dict(b=b, ct=ct, clone=clone, k3=k3, umi=umi, tot=tot, R=pd.DataFrame(Rn, index=b, columns=arms_r), A=pd.DataFrame(An, index=b, columns=arms_a))


def resid_std(X, u):
    D = np.column_stack([np.ones(len(u)), np.log(np.maximum(u, 1))]); Rr = X - D @ np.linalg.lstsq(D, X, rcond=None)[0]; return (Rr - Rr.mean(0)) / np.maximum(Rr.std(0), 1e-12)


def pc1(Z):
    u, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); return float((s[0] ** 2) / (s ** 2).sum()), vt[0], (Z - Z.mean(0)) @ vt[0]


def bic_gain(y):
    from sklearn.mixture import GaussianMixture
    return float(GaussianMixture(1, random_state=0).fit(y[:, None]).bic(y[:, None]) - GaussianMixture(2, n_init=5, random_state=0).fit(y[:, None]).bic(y[:, None]))


def axis_test(Z, rng):
    ev, load, y = pc1(Z); nv, nb = [], []
    for _ in range(20):
        Zp = np.column_stack([rng.permutation(Z[:, k]) for k in range(Z.shape[1])]); e_, _, y_ = pc1(Zp); nv.append(e_); nb.append(bic_gain(y_))
    return dict(pc1_share=ev, null_max=float(max(nv)), p=float((np.sum(np.array(nv) >= ev) + 1) / 21), two_group_bic_gain=bic_gain(y), null_bic_max=float(max(nb))), load, y


def main():
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); out = {}
    for case in ("HT268B1", "HT260C1"):
        d = load(case); tum = d["ct"] == "Tumor"; arms = [a for a in d["A"].columns if a in d["R"].columns and not a.startswith("chrX") and not a.startswith("chrY")]
        groups = {"A": d["clone"] == "A", "B": d["clone"] == "B"}
        if case == "HT260C1":
            z = np.load(ROOT / "data" / "htan" / "HT260C1-Th1K1Fc2U1Z1Bs1_v2.npz", allow_pickle=True); bs = pd.Series(z["b_sub"].astype(str), index=z["barcodes"].astype(str)).reindex(d["b"]).fillna("").to_numpy()
            groups.update({"B1": bs == "B1", "B2": bs == "B2"})
        else:
            # the screen's k = 3 split of the deep tumour nuclei: within A, the two larger k = 3 clusters
            ka = pd.Series(d["k3"][groups["A"]]).value_counts(); top2 = ka.index[:2].tolist(); groups.update({"A1": groups["A"] & (d["k3"] == top2[0]), "A2": groups["A"] & (d["k3"] == top2[1])})
        out[case] = {"nuclei": {g: int(m.sum()) for g, m in groups.items()}}
        for g, m in groups.items():
            if m.sum() < 60:
                continue
            Zr = resid_std(d["R"].loc[d["b"][m], arms].to_numpy(), d["umi"][m]); Za = resid_std(d["A"].loc[d["b"][m], arms].to_numpy(), d["tot"][m])
            ta, la, ya = axis_test(Za, rng); tr_, lr, yr = axis_test(Zr, rng)
            r_n = abs(float(np.corrcoef(ya, yr)[0, 1])); r_l = abs(float(np.corrcoef(la, lr)[0, 1])); top = [arms[i] for i in np.argsort(-np.abs(la))[:6]]
            # does ATAC reproduce the RNA axis?  (1) per nucleus: ATAC projected on the RNA axis's arm weights against the RNA position, null: the
            # weights shuffled across arms; (2) split the nuclei in two along the RNA axis (Gaussian mixture): the two halves' arm differences in ATAC
            # against those in RNA, null: arms shuffled
            from sklearn.mixture import GaussianMixture
            pr = float(np.corrcoef(Za @ lr, yr)[0, 1]); nul = [abs(float(np.corrcoef(Za @ rng.permutation(lr), yr)[0, 1])) for _ in range(200)]
            gm = GaussianMixture(2, n_init=5, random_state=0).fit(yr[:, None]); h = gm.predict(yr[:, None]); ea = Za[h == 1].mean(0) - Za[h == 0].mean(0); er = Zr[h == 1].mean(0) - Zr[h == 0].mean(0)
            re = abs(float(np.corrcoef(ea, er)[0, 1])); nule = [abs(float(np.corrcoef(rng.permutation(ea), er)[0, 1])) for _ in range(200)]
            rep = dict(per_nucleus_r=abs(pr), per_nucleus_p=float((np.sum(np.array(nul) >= abs(pr)) + 1) / 201), split_sizes=[int((h == 0).sum()), int((h == 1).sum())], arm_effects_r=re,
                       arm_effects_p=float((np.sum(np.array(nule) >= re) + 1) / 201), top_rna_arms=[arms[i] for i in np.argsort(-np.abs(lr))[:6]])
            sup = ta["p"] <= 0.05 and r_n >= 0.3 and r_l >= 0.5; sup2 = rep["per_nucleus_p"] <= 0.01 and rep["arm_effects_p"] <= 0.01 and rep["arm_effects_r"] >= 0.5
            out[case][f"within {g}"] = dict(nuclei=int(m.sum()), atac=ta, rna=tr_, atac_vs_rna_per_nucleus=r_n, atac_vs_rna_loadings=r_l, atac_top_arms=top, supported=bool(sup),
                                            atac_reproduces_rna_axis=rep, supported_rna_axis=bool(sup2))
            print(f"   ATAC along the RNA axis: per nucleus |r| {rep['per_nucleus_r']:.2f} (p {rep['per_nucleus_p']:.3f}); split {rep['split_sizes']}: arm effects |r| {rep['arm_effects_r']:.2f} (p {rep['arm_effects_p']:.3f}); RNA arms {rep['top_rna_arms']} | supported: {sup2}", flush=True)
            print(f"{case} within {g} ({m.sum()} nuclei): ATAC PC1 {ta['pc1_share']:.3f} (null max {ta['null_max']:.3f}, p {ta['p']:.2f}; two groups BIC gain {ta['two_group_bic_gain']:.0f} vs null {ta['null_bic_max']:.0f}) | "
                  f"RNA PC1 {tr_['pc1_share']:.3f} (null {tr_['null_max']:.3f}) | ATAC vs RNA per nucleus |r| {r_n:.2f}, loadings |r| {r_l:.2f} | top ATAC arms {top} | supported: {sup}", flush=True)
        if case == "HT268B1":
            m = groups["A1"] | groups["A2"]; y = groups["A1"][m].astype(int); Za = resid_std(d["A"].loc[d["b"][m], arms].to_numpy(), d["tot"][m]); cv = StratifiedKFold(5, shuffle=True, random_state=0); clf = LogisticRegression(max_iter=3000)
            auc = roc_auc_score(y, cross_val_predict(clf, Za, y, cv=cv, method="predict_proba")[:, 1]); null = [roc_auc_score(yp, cross_val_predict(clf, Za, yp, cv=cv, method="predict_proba")[:, 1]) for yp in (rng.permutation(y) for _ in range(5))]
            out[case]["A1 vs A2 from ATAC"] = dict(auc=float(auc), null_max=float(max(null)), nuclei=dict(A1=int(groups["A1"].sum()), A2=int(groups["A2"].sum())))
            print(f"{case}: the RNA split A1 / A2 ({groups['A1'].sum()} / {groups['A2'].sum()}) predicted from ATAC: AUC {auc:.3f} (null max {max(null):.3f})", flush=True)
    json.dump(out, open(O / "subclones_within.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
