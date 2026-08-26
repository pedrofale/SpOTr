"""Does the learned per-clade alpha recover which clades are spatially concordant?

The interpretability claim: alpha_k weights the structural (tree) term for clade k, so a
high alpha_k should mean "the tree is worth using for this clade" -- i.e. that clade is
spatially concordant.

We can test that here, because `etracer_premise.py` already measured, INDEPENDENTLY and
per clone, how well tree distance predicts spatial distance (the per-clone r). If the
learned alpha_k tracks that r, the interpretation is real.

The loss used is unsupervised: fit each spot's expression as a mixture of per-clone mean
expression profiles. No use of the answer key anywhere in fitting.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))


def main(n_iters=400, lr=0.05):
    import spotr.spotr as S
    d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
    X_sc, X_st = d["X_sc"], d["X_st"]
    A, clones, cell_clone = d["A"], d["clones"], d["cell_clone"]

    sc_ad = ad.AnnData(np.asarray(X_sc, dtype=np.float32))
    sc_ad.obs["clade"] = pd.Categorical([str(c) for c in cell_clone])
    st_ad = ad.AnnData(np.asarray(X_st, dtype=np.float32))
    st_ad.layers["expr"] = st_ad.X.copy()

    # clone "signatures" = mean expression profile per clone, from the scRNA-seq only.
    # deconvolution_loss_ls divides signatures by 2 (a CNV convention), so pre-multiply.
    sig = np.stack([X_sc[cell_clone == c].mean(0) for c in clones]).astype(np.float32) * 2.0
    print(f"[alpha] {sc_ad.shape[0]} cells x {st_ad.shape[0]} bins, {len(clones)} clades")

    res = S.run_spotr_perclade_alpha(
        sc_ad, st_ad, np.asarray(d["C_tree"]), np.asarray(d["C_space"]),
        A, sig,
        spot_layer="expr", clade_column="clade", clade_to_ignore="__none__",
        eps=0.02, T_sinkhorn=50, J_alt=3, n_iters=n_iters, lr=lr,
        alpha=0.1, use_loss_ls=True,
    )
    alpha = {str(k): float(v) for k, v in res["alpha_by_clade"].items()}
    loss = np.asarray(res["loss_hist"])
    print(f"[alpha] loss {loss[0]:.5f} -> {loss[-1]:.5f}  (nan: {bool(np.isnan(loss).any())})")

    # independent per-clone spatial concordance, measured from the SPATIAL trees
    prem = pd.read_csv(HERE / "etracer_premise.csv")
    prem = prem[prem.stage == "LE"].copy()
    prem["clone_id"] = prem["clone"].str.split("-").str[0]
    prem = prem.groupby("clone_id").agg(r=("r", "mean"), p=("p", "min"),
                                        n_leaves=("n_leaves", "max")).reset_index()
    df = prem[prem.clone_id.isin(alpha)].copy()
    df["alpha"] = df.clone_id.map(alpha)
    df = df.sort_values("alpha", ascending=False)

    print(f"\n[test] clades with both a learned alpha and a measured concordance: {len(df)}\n")
    print(df[["clone_id", "alpha", "r", "p", "n_leaves"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if len(df) >= 5:
        pr, pp = stats.pearsonr(df.alpha, df.r)
        sr, sp = stats.spearmanr(df.alpha, df.r)
        print(f"\n  Pearson  alpha vs concordance r = {pr:+.3f}  (p = {pp:.4f})")
        print(f"  Spearman                        = {sr:+.3f}  (p = {sp:.4f})")
        print(f"\n  -> {'learned alpha TRACKS spatial concordance' if pp < 0.05 and pr > 0 else 'no usable relationship'}")
    df.to_csv(HERE / "alpha_vs_concordance.csv", index=False)


if __name__ == "__main__":
    main()
