"""Candidate: composition-conditional-excess  (S2)

Definition (author):  gamma0[i,j] = r_i * M[j,k_i] / R_{k_i}  is the composition-matched
max-entropy null coupling: same row masses, same column marginals, same clade x spot mass M,
ZERO within-clade information.  With per-cell anchor profiles y_i and spot anchor Y_j,

    D(gamma) = sum_j b_j || Y_j - (1/b_j) sum_i gamma[i,j] y_i ||^2
    S2       = ( D(gamma0) - D(gamma) ) / D(gamma0)

REPAIRS APPLIED (from the review's "fixes"):
  COSMETIC-1 (mandatory): the ratio normalisation divides by D(gamma0), which IS the coarse
    composition loss -- maximising S2 then rewards a WORSE composition and imports a push
    toward alpha=1.  So the PRIMARY criterion here is the raw excess numerator normalised by
    the ALPHA-INDEPENDENT total anchor variance:
        S_var(alpha) = [ D(gamma0_a) - D(gamma_a) ] / sum_j b_j ||Y_j - Ybar||^2
    We record all three curves: (2) raw excess, (3) S_var (PRIMARY), (4) author's ratio S2,
    plus D(gamma0) alone.
  COSMETIC-2: gamma0 uses the REALISED row masses r_i = sum_j gamma[i,j] (not a_i), so the
    construction survives semi-relaxed / unbalanced couplings.

SCALE (obstacle 3): C is renormalised to a fixed median magnitude at every alpha
    (q29_scalefixed.py one-liner), so cost-scale drift is not a confound.

CONTROLS:
  * SHUFFLED TREE (obstacle 2, mandatory): C_tree replaced by C_tree[perm][:,perm].  Exact
    distance spectrum, destroys which cell is which.
  * ANCHOR REGIME: the shipped simulator makes within-clade anchor variation iid measurement
    noise on a population DISJOINT from the spots -> zero placement information.  We run the
    criterion (a) on the shipped simulator, and (b) on the review's minimal simulator patch
    (heritable within-clade CNV + heritable within-clade layout for compact clades), which is
    the only regime where S2 could be non-degenerate.  Mode chosen by --mode.
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]        # the required grid
GRID_EXT = GRID + [0.90]                            # + top-endpoint diagnostic


def anchor_curves(g, A_hard, y, Y, b):
    """Return (D_gamma, D_gamma0) for one coupling.

    g : [n,m] coupling.  A_hard : [n,K] clade indicator.  y : [n,B] per-cell anchor.
    Y : [m,B] spot anchor.  b : [m] spot masses.
    """
    g = np.asarray(g, dtype=np.float64)
    r = g.sum(1)                                   # realised row masses (COSMETIC-2)
    lab = A_hard.argmax(1)
    M = g.T @ A_hard                               # [m,K] clade x spot mass
    R = np.array([r[lab == k].sum() for k in range(A_hard.shape[1])])
    R = np.maximum(R, 1e-12)
    # gamma0[i,j] = r_i * M[j,k_i] / R_{k_i}   -- built implicitly for the prediction
    g0 = (r[:, None] / R[lab][:, None]) * M[:, lab].T   # [n,m]
    bb = np.maximum(np.asarray(b, np.float64), 1e-12)

    def pred(gm):
        return (gm.T @ y) / bb[:, None]
    p1, p0 = pred(g), pred(g0)
    D1 = float((bb * ((Y - p1) ** 2).sum(1)).sum())
    D0 = float((bb * ((Y - p0) ** 2).sum(1)).sum())
    # exact decomposition of the excess:  D0 - D1 = 2*sum_j b_j <e_j,v_j> - sum_j b_j ||v_j||^2
    e, v = Y - p0, p1 - p0
    cross = float(2.0 * (bb * (e * v).sum(1)).sum())
    quad = float((bb * (v ** 2).sum(1)).sum())
    return D1, D0, cross, quad


def run_seed(d, seed, shuffle_tree, rng, cnv_noise=0.25):
    cidx, spot_ci = np.asarray(d["cell_clade"]), np.asarray(d["spot_clade"])
    K = len(d["clades"])
    n = len(cidx)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    X_sc = np.asarray(d["X_sc"])
    sig = jnp.asarray(np.stack([X_sc[cidx == k].mean(0) for k in range(K)]).astype(np.float32) * 2.0)

    Ct = np.asarray(d["C_tree"], np.float32).copy()
    if shuffle_tree:
        perm = rng.permutation(n)
        Ct = Ct[np.ix_(perm, perm)]
        np.fill_diagonal(Ct, 0.0)

    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(np.asarray(d["X_st"])),
                                             Ct, np.asarray(d["C_space"]))
    Y_expr = jnp.asarray(np.asarray(d["X_st"], np.float32))
    omega = jnp.asarray(A); g0m = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0m, eps=EPS)

    # ---- per-cell anchor y_i and spot anchor Y_j ----
    if "cell_cnv_sc" in d.files:                       # patched simulator: heritable anchor
        y = np.asarray(d["cell_cnv_sc"], np.float64)
    else:                                              # shipped simulator: clone mean + iid noise
        y = np.asarray(d["cnv_clone"], np.float64)[cidx] + \
            rng.normal(0, cnv_noise, size=(n, np.asarray(d["cnv_clone"]).shape[1]))
    Yst = np.asarray(d["cnv_spot"], np.float64)
    bnp = np.asarray(b, np.float64)
    Ybar = (bnp[:, None] * Yst).sum(0) / bnp.sum()
    TOTVAR = float((bnp * ((Yst - Ybar) ** 2).sum(1)).sum())

    def cost(av, g):
        return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                            Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            equalize="measured")
    REF = float(jnp.median(jnp.abs(cost(0.0, g0m))))

    out = []
    for av in GRID_EXT:
        g = g0m
        for _ in range(J):
            C = cost(av, g)
            C = C * (REF / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))   # SCALE FIX (q29)
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        gn = np.asarray(g)
        Dg, D0, cross, quad = anchor_curves(gn, A, y, Yst, bnp)
        out.append(dict(seed=seed, alpha=av, shuffled=bool(shuffle_tree),
                        acc=float((spot_ci[gn.argmax(1)] == cidx).mean()),
                        D_gamma=Dg, D_gamma0=D0, cross=cross, quad=quad,
                        cross_var=cross / TOTVAR, quad_var=quad / TOTVAR,
                        excess_raw=D0 - Dg,                 # curve (2)
                        S_var=(D0 - Dg) / TOTVAR,           # curve (3) PRIMARY
                        S_ratio=(D0 - Dg) / max(D0, 1e-12), # curve (4) author's S2
                        expr_loss=float(S.deconvolution_loss_ls(Y_expr, g, jnp.asarray(A), sig)),
                        medC=float(jnp.median(jnp.abs(C)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="shipped", choices=["shipped", "patched"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pref = "sim_ctrl_s" if args.mode == "shipped" else "sim_cce_s"
    rows = []
    for seed in range(14):
        f = HERE / f"{pref}{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        if not bool(d["in_band"][0]):
            continue
        for shuf in (False, True):
            rows += run_seed(d, seed, shuf, np.random.default_rng(1234 + seed + 7000 * shuf))
        print(f"  seed {seed} done", flush=True)
    r = pd.DataFrame(rows)
    tag = args.tag or args.mode
    r.to_csv(HERE / f"cand_composition-conditional-excess_{tag}.csv", index=False)
    report(r, tag)


def mono(x):
    dd = np.diff(np.asarray(x, float))
    return bool((dd > 0).all() or (dd < 0).all())


def report(r, tag):
    print("\n" + "=" * 110)
    print(f"CANDIDATE composition-conditional-excess  [{tag}]   grid={GRID}  (+0.90 diagnostic)")
    for shuf in (False, True):
        sub = r[r.shuffled == shuf]
        if not len(sub):
            continue
        lab = "SHUFFLED-TREE CONTROL" if shuf else "REAL TREE"
        print("\n" + "-" * 110)
        print(f"### {lab}")
        for curve in ("S_var", "excess_raw", "S_ratio"):
            per = []
            for s_, gsub in sub.groupby("seed"):
                gg = gsub.set_index("alpha").sort_index()
                g6 = gg.loc[GRID]
                per.append(dict(seed=s_,
                                sel=float(g6[curve].idxmax()),
                                sel_ext=float(gg[curve].idxmax()),
                                opt=float(g6.acc.idxmax()),
                                mono=mono(g6[curve].values),
                                acc_sel=float(g6.acc[g6[curve].idxmax()]),
                                acc_a0=float(g6.acc[0.0]), acc_02=float(g6.acc[0.20]),
                                acc_opt=float(g6.acc.max()),
                                rng_=float(np.ptp(g6[curve].values))))
            p = pd.DataFrame(per)
            print(f"\n  curve {curve:<11} n={len(p)}")
            print(f"    selected alpha (6-pt grid) : {', '.join(f'{v:.2f}' for v in p.sel)}")
            print(f"    selected alpha (+0.90)     : {', '.join(f'{v:.2f}' for v in p.sel_ext)}")
            print(f"    argmax(accuracy)           : {', '.join(f'{v:.2f}' for v in p.opt)}")
            print(f"    agreement                  : {int((np.abs(p.sel-p.opt)<1e-9).sum())}/{len(p)}")
            print(f"    mean |sel - opt|           : {np.abs(p.sel-p.opt).mean():.3f}")
            print(f"    monotone seeds             : {int(p.mono.sum())}/{len(p)}")
            print(f"    top-endpoint (0.90) pinned : {int((p.sel_ext>=0.90).sum())}/{len(p)}")
            print(f"    acc@sel {p.acc_sel.mean():.4f} | a=0 {p.acc_a0.mean():.4f} | "
                  f"a=0.2 {p.acc_02.mean():.4f} | oracle {p.acc_opt.mean():.4f}")
            print(f"    beats a=0: {p.acc_sel.mean()>p.acc_a0.mean()}   "
                  f"beats a=0.2: {p.acc_sel.mean()>p.acc_02.mean()}   "
                  f"mean ptp(S) over grid = {p.rng_.mean():.3e}")
    # mechanism: which of the two excess terms drives S?
    sub = r[~r.shuffled].groupby("alpha")[["cross_var", "quad_var", "S_var"]].mean()
    print("\n" + "-" * 110)
    print("MECHANISM (real tree, mean over seeds): excess = cross - quad, all / anchor variance")
    print(f"{'alpha':>7}{'2<e,v>/V':>14}{'||v||^2/V':>14}{'S_var':>14}")
    for aa, row in sub.iterrows():
        print(f"{aa:>7.2f}{row.cross_var:>+14.5f}{row.quad_var:>14.5f}{row.S_var:>+14.5f}")

    # per-seed table on the required grid, real tree
    print("\n" + "-" * 110)
    print("PER-SEED  S_var / accuracy  on the required grid (REAL TREE)")
    sub = r[~r.shuffled]
    print(f"{'seed':>5} " + "".join(f"{'a='+str(v):>18}" for v in GRID))
    for s_, gsub in sub.groupby("seed"):
        gg = gsub.set_index("alpha")
        print(f"{s_:>5} " + "".join(f"{gg.S_var[v]:>+10.5f}/{gg.acc[v]:.3f}" for v in GRID))


if __name__ == "__main__":
    main()
