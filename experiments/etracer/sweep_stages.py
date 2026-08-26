"""Global alpha/phi sweep with per-clade readout, on the FULL bin sets, pooled over stages.

Only clades with enough bins can be measured at all, so results are filtered by MIN_BINS
and the pooled correlation is what the interpretability claim rests on.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

A_GRID = [0.0, 0.05, 0.10, 0.20, 0.35]
P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS, T_SINK, J_ALT = 0.02, 50, 3
MIN_BINS = 40


def run_stage(stage):
    d = np.load(HERE / f"etracer_inputs_{stage}.npz", allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    print(f"[{stage}] {A.shape[0]} cells x {len(truth)} bins, {K} clades")

    def scores(g):
        prop = np.asarray(g).T @ A
        prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        out = []
        for k in range(K):
            t = (truth == clones[k]).astype(float)
            out.append(np.nan if (t.std() == 0 or prop[:, k].std() == 0)
                       else float(np.corrcoef(prop[:, k], t)[0, 1]))
        return np.array(out)

    def run(av, pv, dec):
        ak = jnp.asarray(np.full(K, av, dtype=np.float32))
        pk = jnp.asarray(np.full(K, pv, dtype=np.float32)) if dec else None
        g = gamma0
        for _ in range(J_ALT):
            C = S.build_perclade_alpha_cost(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            use_decomp=dec, Omega=Omega if dec else None, phi_k=pk)
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
        return g

    ac = {v: scores(run(v, 0.5, False)) for v in A_GRID}
    pc = {v: scores(run(0.10, v, True)) for v in P_GRID}

    prem = pd.read_csv(HERE / "etracer_premise.csv")
    prem = prem[prem.stage == stage].copy()
    prem["clone_id"] = prem["clone"].str.split("-").str[0]
    prem = prem.groupby("clone_id").agg(r=("r", "mean")).reset_index()

    rows = []
    for k in range(K):
        av = np.array([ac[v][k] for v in A_GRID]); pv = np.array([pc[v][k] for v in P_GRID])
        if not (np.isfinite(av).any() and np.isfinite(pv).any()):
            continue
        rows.append(dict(stage=stage, clone_id=str(clones[k]),
                         n_bins=int((truth == clones[k]).sum()),
                         pref_alpha=A_GRID[int(np.nanargmax(av))],
                         pref_phi=P_GRID[int(np.nanargmax(pv))]))
    return pd.DataFrame(rows).merge(prem, on="clone_id", how="inner")


def main():
    df = pd.concat([run_stage(s) for s in ["LE", "EE"]], ignore_index=True)
    df.to_csv(HERE / "sweep_stages.csv", index=False)
    sub = df[df.n_bins >= MIN_BINS]
    print(f"\npooled: {len(df)} clades, {len(sub)} with >= {MIN_BINS} bins\n")
    print(sub.sort_values("r", ascending=False)
            [["stage","clone_id","n_bins","r","pref_alpha","pref_phi"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for col, nm in [("pref_alpha","alpha"), ("pref_phi","phi")]:
        pr, pp = stats.pearsonr(sub[col], sub.r); sr, sp = stats.spearmanr(sub[col], sub.r)
        hi = sub[sub.r >= sub.r.median()][col].mean(); lo = sub[sub.r < sub.r.median()][col].mean()
        print(f"\n  preferred {nm} vs within-clone concordance  (n = {len(sub)})")
        print(f"    Pearson  {pr:+.3f}  (p = {pp:.4f})")
        print(f"    Spearman {sr:+.3f}  (p = {sp:.4f})")
        print(f"    mean preferred {nm}: concordant {hi:.3f} vs discordant {lo:.3f}")


if __name__ == "__main__":
    main()
