"""Cleaner test: sweep alpha and phi GLOBALLY, and read off each clade's preferred value.

The per-clade sweep was confounded -- perturbing one clade away from a uniform vector is
itself penalised by the per-clade equalizer in build_perclade_alpha_cost, so most clades
trivially "preferred" the baseline. Sweeping globally keeps the vector uniform at every
step, so each clade's score curve is comparable across settings.

If the interpretability claim holds, clades that are spatially concordant should peak at a
higher global alpha (use structure) and higher global phi (use WITHIN-clade structure).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

A_GRID = [0.0, 0.05, 0.075, 0.15, 0.30]
P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS, T_SINK, J_ALT = 0.02, 50, 3


def main():
    d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)

    def scores(g):
        prop = np.asarray(g).T @ A
        prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        out = []
        for k in range(K):
            t = (truth == clones[k]).astype(float)
            out.append(np.nan if (t.std() == 0 or prop[:, k].std() == 0)
                       else float(np.corrcoef(prop[:, k], t)[0, 1]))
        return np.array(out)

    def run(alpha_v, phi_v, decomp):
        ak = jnp.asarray(np.full(K, alpha_v, dtype=np.float32))
        pk = jnp.asarray(np.full(K, phi_v, dtype=np.float32)) if decomp else None
        g = gamma0
        for _ in range(J_ALT):
            C = S.build_perclade_alpha_cost(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            use_decomp=decomp, Omega=Omega if decomp else None,
                                            phi_k=pk)
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
        return g

    alpha_curves = {v: scores(run(v, 0.5, False)) for v in A_GRID}
    phi_curves = {v: scores(run(0.075, v, True)) for v in P_GRID}

    prem = pd.read_csv(HERE / "etracer_premise.csv")
    prem = prem[prem.stage == "LE"].copy()
    prem["clone_id"] = prem["clone"].str.split("-").str[0]
    prem = prem.groupby("clone_id").agg(r=("r", "mean")).reset_index()

    rows = []
    for k in range(K):
        av = np.array([alpha_curves[v][k] for v in A_GRID])
        pv = np.array([phi_curves[v][k] for v in P_GRID])
        rows.append(dict(clone_id=str(clones[k]),
                         pref_alpha=A_GRID[int(np.nanargmax(av))] if np.isfinite(av).any() else np.nan,
                         pref_phi=P_GRID[int(np.nanargmax(pv))] if np.isfinite(pv).any() else np.nan,
                         n_bins=int((truth == clones[k]).sum())))
    df = pd.DataFrame(rows).merge(prem, on="clone_id", how="inner").dropna()
    df.to_csv(HERE / "global_sweep_perclade.csv", index=False)

    print(df.sort_values("r", ascending=False)[["clone_id","n_bins","r","pref_alpha","pref_phi"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for col, name in [("pref_alpha","alpha"), ("pref_phi","phi")]:
        pr, pp = stats.pearsonr(df[col], df.r); sr, sp = stats.spearmanr(df[col], df.r)
        hi = df[df.r >= df.r.median()][col].mean(); lo = df[df.r < df.r.median()][col].mean()
        print(f"\n  preferred {name} vs within-clone concordance:")
        print(f"    Pearson {pr:+.3f} (p={pp:.4f})   Spearman {sr:+.3f} (p={sp:.4f})")
        print(f"    mean preferred {name}: concordant {hi:.3f} vs discordant {lo:.3f}")
    # weight by clade size -- small clades are noisy
    w = df.n_bins.to_numpy(float)
    for col, name in [("pref_alpha","alpha"), ("pref_phi","phi")]:
        cm = np.cov(np.stack([df[col], df.r]), aweights=w)
        wr = cm[0,1]/np.sqrt(cm[0,0]*cm[1,1])
        print(f"  size-weighted corr(preferred {name}, r) = {wr:+.3f}")


if __name__ == "__main__":
    main()
