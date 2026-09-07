"""Candidate: permuted-tree-excess-agreement  (S(alpha) = A_T(alpha) - mean_r A_{P,r}(alpha))

IDEA (as proposed). Split the genes into two halves. Map each half independently against the
SAME tree (arm T). Measure the multivariate intraclass concordance A of the two halves'
posterior-mean spatial placements. Repeat with the tree's leaf labels permuted (arm P): same
topology, same edge-length spectrum, same GW magnitude, no true cell-to-lineage correspondence.
S = A_T - A_P. S(0)=0 because the tree does not enter the cost; S(1)=0 because the expression
channel does not enter and both halves solve the identical problem in BOTH arms. Interior
extremum forced by Rolle.

WHAT I CHANGED vs the spec, following the review's "fixes" (all reported):
  * GRID EXTENDED TO 1.0 (review: "the structural protection lives in the unevaluated tail").
    Grid = [0, .10, .20, .35, .50, .70, .85, 1.0]; the required 6-point subset is reported too.
  * SCALE HELD FIXED at every alpha (obstacle 3 / q29_scalefixed.py one-liner), identically in
    both arms, so the arms differ only in which tree they see.
  * IDENTICAL, DETERMINISTIC SOLVER INIT across arms and halves (g0 = a b^T), plus a
    MULTI-RESTART CONTROL from a second fixed init, to rule out the review's confound (b):
    a frustrated arm-P landscape deflating A_P for numerical reasons.
  * SECOND NULL: whole-clade label swaps (permute which clade sits where, keeping cells within
    a clade together). The review says within-clade permutation kills the PER-CLADE variant;
    the per-clade variant is therefore WITHDRAWN as the review demands, and the clade-swap null
    is run instead as an alternative, more conservative GLOBAL null.
  * REPAIR (2), MOMENT MATCHING: instead of argmax S, fit (V_e, V_t, V_n, c) to the two whole
    curves A_T(a), A_P(a) under the review's own linearisation and compute
    a* = V_n / (V_n + V_et), V_et = V_t - 2c + V_e   (identifying assumption: the truth is the
    expression channel's common signal m_e, so the tree's error is Var(x_t - m_e)).
  * REPAIR (1), DETECTOR: max_a S is reported with its permutation-to-permutation sd.
  * MANDATORY SHUFFLED-TREE CONTROL (obstacle 2): the entire criterion re-run with the real
    C_tree replaced by a leaf-label-shuffled one in arm T as well.

Reduced from the proposal to fit the compute cap: R=6 cell-permutation nulls (not 8) and R=3
clade-swap nulls; 2 gene halves; 10 in-band seeds.
"""
import sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax, jax.numpy as jnp
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID6 = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
GRID = GRID6 + [0.85, 1.0]
R_CELL, R_CLADE = 6, 3


# ---------------------------------------------------------------- agreement statistic
def posterior_mean(gn, xy):
    w = gn / np.maximum(gn.sum(1, keepdims=True), 1e-30)
    return w @ xy


def concordance(muA, muB, w):
    """Multivariate intraclass concordance. 1 = identical placements, 0 = independent."""
    num = float((w * ((muA - muB) ** 2).sum(1)).sum())
    dA = float((w * ((muA - (w[:, None] * muA).sum(0)) ** 2).sum(1)).sum())
    dB = float((w * ((muB - (w[:, None] * muB).sum(0)) ** 2).sum(1)).sum())
    return 1.0 - num / max(dA + dB, 1e-30)


# ---------------------------------------------------------------- one solve
def solve(alpha, Cf, C_T, C_S, a, b, omega, sF, sL, K, g_init, ref):
    g = g_init
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.full((K,), alpha, dtype=jnp.float32),
                                         Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                         equalize="measured")
        C = C * (ref / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))   # q29 scale fix
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    return np.asarray(g)


def main():
    rows, curves = [], []
    seeds = []
    for seed in range(14):
        f = HERE / f"sim_ctrl_s{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        if not bool(d["in_band"][0]):
            continue
        seeds.append(seed)
        t0 = time.time()
        cidx, spot_ci = np.asarray(d["cell_clade"]), np.asarray(d["spot_clade"])
        K, n, G = len(d["clades"]), len(cidx), d["X_sc"].shape[1]
        Amat = np.zeros((n, K), np.float32); Amat[np.arange(n), cidx] = 1.0
        omega = jnp.asarray(Amat)
        xy = np.asarray(d["xy_st"], np.float64)
        w = np.full(n, 1.0 / n)

        rng = np.random.default_rng(1000 + seed)
        perm_genes = rng.permutation(G)
        halves = [perm_genes[: G // 2], perm_genes[G // 2:]]

        # ---- tree variants
        Ctree_real = np.asarray(d["C_tree"], np.float32)

        def cell_perm_tree(r):
            p = np.random.default_rng(7000 + 100 * seed + r).permutation(n)
            return Ctree_real[np.ix_(p, p)]

        def clade_swap_tree(r):
            g_ = np.random.default_rng(9000 + 100 * seed + r)
            kperm = g_.permutation(K)
            # map every cell of clade k to a cell of clade kperm[k] (same within-clade order)
            idx_by_k = [np.where(cidx == k)[0] for k in range(K)]
            p = np.arange(n)
            for k in range(K):
                src, dst = idx_by_k[k], idx_by_k[kperm[k]]
                m = min(len(src), len(dst))
                p[src[:m]] = dst[:m]
            return Ctree_real[np.ix_(p, p)]

        shuffled_base = cell_perm_tree(99)     # the obstacle-2 shuffled "real" tree

        # ---- full-gene accuracy sweep (real tree) -------------------------------
        Cf_all, C_T, C_S, a, b = S.prepare_ot_inputs(
            ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]), Ctree_real, d["C_space"])
        g0 = a[:, None] * b[None, :]
        sF, sL = S.make_cladexfgw_scales(Cf_all, C_T, C_S, a, b, g0, eps=EPS)
        C0 = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cf_all, C_T, C_S,
                                          a, b, g0, omega, sF, sL, equalize="measured")
        ref_all = float(jnp.median(jnp.abs(C0)))
        acc = {}
        for av in GRID:
            gn = solve(av, Cf_all, C_T, C_S, a, b, omega, sF, sL, K, g0, ref_all)
            acc[av] = float((spot_ci[gn.argmax(1)] == cidx).mean())

        # ---- per-half fixed inputs ---------------------------------------------
        half_in = []
        for h in halves:
            Cfh, _, _, _, _ = S.prepare_ot_inputs(
                ad.AnnData(np.ascontiguousarray(d["X_sc"][:, h])),
                ad.AnnData(np.ascontiguousarray(d["X_st"][:, h])), Ctree_real, d["C_space"])
            half_in.append(Cfh)

        # a second deterministic init for the restart control (identical across all arms)
        pert = np.random.default_rng(31337 + seed).gamma(2.0, 0.5, size=(n, len(spot_ci)))
        g_alt = jnp.asarray((np.asarray(g0) * pert / (np.asarray(g0) * pert).sum()).astype(np.float32))

        def arm_A(Ctree, init, tag):
            """concordance curve of the two gene halves against tree `Ctree`."""
            CT = jnp.asarray(Ctree)
            scales, refs = [], []
            for Cfh in half_in:
                sFh, sLh = S.make_cladexfgw_scales(Cfh, CT, C_S, a, b, g0, eps=EPS)
                Ch0 = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cfh, CT, C_S,
                                                   a, b, g0, omega, sFh, sLh, equalize="measured")
                scales.append((sFh, sLh)); refs.append(float(jnp.median(jnp.abs(Ch0))))
            out = {}
            for av in GRID:
                mus = []
                for Cfh, (sFh, sLh), rf in zip(half_in, scales, refs):
                    gn = solve(av, Cfh, CT, C_S, a, b, omega, sFh, sLh, K, init, rf)
                    mus.append(posterior_mean(gn, xy))
                out[av] = concordance(mus[0], mus[1], w)
            return out

        blocks = {}
        blocks[("real", "T", "g0")] = arm_A(Ctree_real, g0, "T")
        for r in range(R_CELL):
            blocks[("real", f"P{r}", "g0")] = arm_A(cell_perm_tree(r), g0, "P")
        for r in range(R_CLADE):
            blocks[("real", f"Q{r}", "g0")] = arm_A(clade_swap_tree(r), g0, "Q")
        # obstacle-2 control: arm T sees a shuffled tree; nulls are further shuffles
        blocks[("shuf", "T", "g0")] = arm_A(shuffled_base, g0, "T")
        for r in range(R_CELL):
            blocks[("shuf", f"P{r}", "g0")] = arm_A(cell_perm_tree(50 + r), g0, "P")
        # restart control
        blocks[("real", "T", "alt")] = arm_A(Ctree_real, g_alt, "T")
        for r in range(2):
            blocks[("real", f"P{r}", "alt")] = arm_A(cell_perm_tree(r), g_alt, "P")

        for (variant, arm, init), c in blocks.items():
            for av in GRID:
                curves.append(dict(seed=seed, variant=variant, arm=arm, init=init,
                                   alpha=av, A=c[av], acc=acc[av]))
        print(f"  seed {seed} done in {time.time()-t0:.1f}s", flush=True)

    cu = pd.DataFrame(curves)
    cu.to_csv(HERE / "cand_permuted-tree-excess-agreement.csv", index=False)
    report(cu, seeds)


# ============================================================================ reporting
def _S(cu, seed, variant, init="g0", nullpref="P"):
    sub = cu[(cu.seed == seed) & (cu.variant == variant) & (cu.init == init)]
    AT = sub[sub.arm == "T"].set_index("alpha").A.reindex(GRID)
    nulls = sub[sub.arm.str.startswith(nullpref)]
    per_r = {r: g.set_index("alpha").A.reindex(GRID) for r, g in nulls.groupby("arm")}
    AP = pd.concat(per_r.values(), axis=1)
    return AT, AP.mean(1), AP, AT - AP.mean(1)


def moment_match(AT, AP, grid):
    ga = np.asarray(grid)

    def model(th):
        Ve, Vt, Vn = np.exp(th[:3]); rho = 1 / (1 + np.exp(-th[3]))
        c = rho * np.sqrt(Ve * Vt)
        u = (1 - ga) ** 2 * Ve + ga ** 2 * Vt
        x = 2 * ga * (1 - ga) * c
        N = (1 - ga) ** 2 * Vn
        return (u + x) / (u + x + N), u / (u + N)

    def res(th):
        aT, aP = model(th)
        return np.concatenate([aT - AT.values, aP - AP.values])

    best = None
    for th0 in [np.array([0., 0., 0., 0.]), np.array([0., 1., -1., 1.]),
                np.array([0., -1., 1., -1.]), np.array([0., .5, .5, 2.])]:
        try:
            r = least_squares(res, th0, max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    Ve, Vt, Vn = np.exp(best.x[:3]); rho = 1 / (1 + np.exp(-best.x[3]))
    c = rho * np.sqrt(Ve * Vt)
    Vet = max(Vt - 2 * c + Ve, 1e-9)
    return float(Vn / (Vn + Vet)), dict(Ve=Ve, Vt=Vt, Vn=Vn, c=c, Vet=Vet, cost=float(best.cost))


def nearest(x, grid):
    return grid[int(np.argmin(np.abs(np.asarray(grid) - x)))]


def report(cu, seeds):
    print("\n" + "=" * 110)
    print("S(alpha) = A_T - mean_r A_P   (REAL tree)   and TRUE accuracy, per seed")
    print("=" * 110)
    hdr = "".join(f"{'a=' + str(v):>17}" for v in GRID6)
    print(f"{'seed':>5}{hdr}     |  a=0.85          a=1.00")
    print(f"{'':>5}" + "".join(f"{'S / acc':>17}" for _ in GRID6))
    print("-" * 110)
    per = []
    for s_ in seeds:
        AT, APm, AP, Sc = _S(cu, s_, "real")
        acc = cu[(cu.seed == s_) & (cu.variant == "real") & (cu.arm == "T") &
                 (cu.init == "g0")].set_index("alpha").acc.reindex(GRID)
        line = f"{s_:>5}" + "".join(f"{Sc[v]:>+9.4f}/{acc[v]:.3f}" for v in GRID6)
        line += f"     |{Sc[0.85]:+9.4f}/{acc[0.85]:.3f} {Sc[1.0]:+9.4f}/{acc[1.0]:.3f}"
        print(line)
        d_ = np.diff(Sc.values)
        per.append(dict(
            seed=s_, sel=float(Sc.idxmax()), sel6=float(Sc.reindex(GRID6).idxmax()),
            opt=float(acc.idxmax()), opt6=float(acc.reindex(GRID6).idxmax()),
            acc_sel=float(acc[Sc.idxmax()]), acc_a0=float(acc[0.0]),
            acc_a02=float(acc[0.20]), acc_opt=float(acc.max()),
            acc_opt6=float(acc.reindex(GRID6).max()),
            S0=float(Sc[0.0]), S1=float(Sc[1.0]), Smax=float(Sc.max()),
            Ssd=float(AP.std(1).max()),
            mono=bool((d_ > 0).all() or (d_ < 0).all())))
    p = pd.DataFrame(per)

    print("\n--- construction checks (the candidate's own prediction #1) ---")
    print(f"  |S(0)|  max over seeds : {p.S0.abs().max():.2e}   (<1e-10 in "
          f"{int((p.S0.abs()<1e-10).sum())}/{len(p)})")
    print(f"  |S(1)|  max over seeds : {p.S1.abs().max():.2e}   (<1e-10 in "
          f"{int((p.S1.abs()<1e-10).sum())}/{len(p)})")
    print(f"  argmax S strictly interior : "
          f"{int(((p.sel>0)&(p.sel<1)).sum())}/{len(p)}")
    print(f"  S MONOTONE in alpha        : {int(p.mono.sum())}/{len(p)}  (monotone = failed)")

    print("\n--- SELECTION (argmax S over the extended grid incl. 1.0) ---")
    print(f"  selected alpha per seed : {', '.join(f'{v:.2f}' for v in p.sel)}")
    print(f"  optimal  alpha per seed : {', '.join(f'{v:.2f}' for v in p.opt)}")
    ag = int((np.abs(p.sel - p.opt) < 1e-9).sum())
    print(f"  AGREEMENT argmax S == argmax acc : {ag}/{len(p)}")
    print(f"  mean |selected - optimal|        : {np.abs(p.sel-p.opt).mean():.3f}")
    print(f"  selected inside [0.10, 0.35]     : "
          f"{int(((p.sel>=0.10)&(p.sel<=0.35)).sum())}/{len(p)}")
    print(f"  sd(selected alpha) across seeds  : {p.sel.std():.4f}")
    print(f"\n  MEAN ACCURACY  at selected {p.acc_sel.mean():.4f} | "
          f"alpha=0 {p.acc_a0.mean():.4f} | fixed a=0.2 {p.acc_a02.mean():.4f} | "
          f"oracle {p.acc_opt.mean():.4f}")
    print(f"  beats alpha=0   : {p.acc_sel.mean() > p.acc_a0.mean()}  "
          f"(per-seed {int((p.acc_sel>p.acc_a0).sum())}/{len(p)})")
    print(f"  beats a=0.2     : {p.acc_sel.mean() > p.acc_a02.mean()}  "
          f"(per-seed {int((p.acc_sel>p.acc_a02).sum())}/{len(p)})")
    print(f"  [6-point grid only] selected {', '.join(f'{v:.2f}' for v in p.sel6)}  "
          f"agreement {int((np.abs(p.sel6-p.opt6)<1e-9).sum())}/{len(p)}")

    # ---------- detector view (review fix 1)
    print("\n--- DETECTOR view (review fix 1): max_a S vs its permutation-to-permutation sd ---")
    print(f"  max_a S per seed : {', '.join(f'{v:+.4f}' for v in p.Smax)}")
    print(f"  perm sd at that alpha : {', '.join(f'{v:.4f}' for v in p.Ssd)}")
    print(f"  max_a S > 3 sd in {int((p.Smax > 3*p.Ssd).sum())}/{len(p)} seeds "
          f"(candidate predicted >=9/11)")

    # ---------- shuffled-tree control (obstacle 2)
    print("\n" + "=" * 110)
    print("MANDATORY SHUFFLED-TREE CONTROL (obstacle 2): arm T sees a leaf-shuffled tree")
    print("=" * 110)
    sh = []
    for s_ in seeds:
        AT, APm, AP, Sc = _S(cu, s_, "shuf")
        sh.append(dict(seed=s_, sel=float(Sc.idxmax()), Smax=float(Sc.max()),
                       Ssd=float(AP.std(1).max())))
    q = pd.DataFrame(sh)
    print(f"  selected alpha (shuffled) : {', '.join(f'{v:.2f}' for v in q.sel)}")
    print(f"  selected alpha (real)     : {', '.join(f'{v:.2f}' for v in p.sel)}")
    same = int((np.abs(q.sel.values - p.sel.values) < 1e-9).sum())
    print(f"  SAME selected alpha as real tree in {same}/{len(p)} seeds")
    print(f"  mean |sel_shuf - sel_real| : {np.abs(q.sel.values-p.sel.values).mean():.3f}")
    print(f"  max_a S  real {p.Smax.mean():+.4f}  vs  shuffled {q.Smax.mean():+.4f}   "
          f"(ratio {p.Smax.mean()/max(abs(q.Smax.mean()),1e-9):.2f}x)")
    print(f"  shuffled max_a S > 3sd in {int((q.Smax > 3*q.Ssd).sum())}/{len(q)} seeds")

    # ---------- clade-swap null
    print("\n--- ALTERNATIVE NULL: whole-clade label swaps (per-clade variant WITHDRAWN) ---")
    cl = []
    for s_ in seeds:
        AT, APm, AP, Sc = _S(cu, s_, "real", nullpref="Q")
        cl.append(dict(seed=s_, sel=float(Sc.idxmax()), Smax=float(Sc.max())))
    cq = pd.DataFrame(cl)
    print(f"  selected alpha : {', '.join(f'{v:.2f}' for v in cq.sel)}   "
          f"max_a S mean {cq.Smax.mean():+.4f}")

    # ---------- restart control (review confound b)
    print("\n--- MULTI-RESTART CONTROL (review confound b: frustrated arm-P landscape) ---")
    rs = []
    for s_ in seeds:
        sub = cu[(cu.seed == s_) & (cu.variant == "real") & (cu.init == "alt")]
        AT = sub[sub.arm == "T"].set_index("alpha").A.reindex(GRID)
        AP = pd.concat([g.set_index("alpha").A.reindex(GRID)
                        for _, g in sub[sub.arm.str.startswith("P")].groupby("arm")], axis=1)
        Sc = AT - AP.mean(1)
        rs.append(dict(seed=s_, sel=float(Sc.idxmax()), Smax=float(Sc.max())))
    rq = pd.DataFrame(rs)
    print(f"  selected alpha, alt init : {', '.join(f'{v:.2f}' for v in rq.sel)}")
    print(f"  selected alpha, g0  init : {', '.join(f'{v:.2f}' for v in p.sel)}")
    print(f"  same selection in {int((np.abs(rq.sel.values-p.sel.values)<1e-9).sum())}/{len(p)}"
          f"  -> selection is {'init-robust' if (np.abs(rq.sel.values-p.sel.values)<1e-9).mean()>0.6 else 'INIT-DEPENDENT'}")

    # ---------- moment matching (review fix 2)
    print("\n" + "=" * 110)
    print("REVIEW FIX (2): MOMENT MATCHING instead of argmax  ->  a* = Vn/(Vn+Vet)")
    print("=" * 110)
    mm = []
    for s_ in seeds:
        AT, APm, AP, Sc = _S(cu, s_, "real")
        astar, fit = moment_match(AT, APm, GRID)
        acc = cu[(cu.seed == s_) & (cu.variant == "real") & (cu.arm == "T") &
                 (cu.init == "g0")].set_index("alpha").acc.reindex(GRID)
        g = nearest(astar, GRID)
        mm.append(dict(seed=s_, astar=astar, grid=g, acc_sel=float(acc[g]),
                       opt=float(acc.idxmax()), **fit))
    m = pd.DataFrame(mm)
    print(f"  a*  per seed : {', '.join(f'{v:.3f}' for v in m.astar)}")
    print(f"  snapped      : {', '.join(f'{v:.2f}' for v in m.grid)}")
    print(f"  optimal      : {', '.join(f'{v:.2f}' for v in m.opt)}")
    print(f"  agreement {int((np.abs(m.grid-m.opt)<1e-9).sum())}/{len(m)}   "
          f"mean|a*-opt| {np.abs(m.astar-m.opt).mean():.3f}   sd(a*) {m.astar.std():.4f}")
    print(f"  MEAN ACC at a* {m.acc_sel.mean():.4f} vs a=0 {p.acc_a0.mean():.4f} "
          f"vs a=0.2 {p.acc_a02.mean():.4f} vs oracle {p.acc_opt.mean():.4f}")
    print(f"  fitted c: {', '.join(f'{v:.3f}' for v in m.c)}")

    # co-movement (review's test that matters): does the selection track the moving optimum?
    print("\n--- CO-MOVEMENT across seeds (argmax S vs argmax acc) ---")
    rho = spearmanr(p.sel, p.opt)
    print(f"  Spearman rho(selected, optimal) = {rho.statistic:+.3f} (p={rho.pvalue:.3f});"
          f"  sd(selected)={p.sel.std():.4f}, sd(optimal)={p.opt.std():.4f}")

    p.to_csv(HERE / "cand_permuted-tree-excess-agreement_summary.csv", index=False)


if __name__ == "__main__":
    main()
