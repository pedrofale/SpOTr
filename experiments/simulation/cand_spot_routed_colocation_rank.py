"""Candidate `spot-routed-colocation-rank` -- empirical settlement on the controlled simulator.

WHAT IS TESTED
--------------
As specified (S3): from the coupling build the cell x cell co-location kernel
    W = (g D_b^{-1/2})(g D_b^{-1/2})^T,   W[i,i'] = sum_j g[i,j] g[i',j] / b_j
from the anchor build the spot-routed cell x cell kernel
    p[i,j] prop. b_j exp(-||y_i - Y_j||^2 / (2 sigma^2)),  E = sum_j p[i,j]p[i',j]/b_j
and score S3 = rank corr(W, E) over WITHIN-CLADE pairs.

The review calls this fatal for three reasons and names repairs. Implemented here:

  * S3_within   -- the candidate exactly as specified.
  * S3_all      -- same statistic over ALL pairs. On this simulator the anchor (CNV) is
                   clonal, so within-clade pairs carry no anchor signal at all; the
                   all-pairs version is the candidate's best available shot and is
                   reported so the negative is not manufactured by the restriction.
  * S3_cc       -- the candidate's own PAIRED NEGATIVE CONTROL (i): E rebuilt from
                   cell-cell anchor similarity instead of spot-routed posteriors.
                   Review's algebra says this is S3 up to a scalar density factor.
  * REPAIR (A)  -- "keep the spot index". First-order, CELL-resolution scoring of the
                   coupling against the anchor posterior:
                       S_A     = (1/n) sum_i sum_j ghat[i,j] log p[i,j]     (cross-entropy)
                       S_Arank = mean_i spearman(g[i,:], p[i,:])            (sigma-free)
                   Not spot-permutation invariant, so it can see misregistration.
  * REPAIR (B)  -- "keep a spatial index on one side": rank agreement between
                   g[i,j]g[i',j] and p[i,j]p[i',j] over TRIPLES (i,i',j), j not summed.

CONTROLS
  1. SPOT-PERMUTATION INVARIANCE (review's decisive test): b is uniform here, so every
     spot is in one equal-b_j group; permuting g's columns must leave W (hence S3)
     numerically identical while accuracy collapses to the random floor.
  2. SHUFFLED TREE (mandatory, obstacle 2): C_tree leaf labels permuted, whole sweep
     re-run. The anchor is a measured modality and is left alone.
  3. COUPLING ROW ENTROPY vs alpha, to detect a hedging artefact.
  4. Anchor-only map accuracy (argmax_j p[i,j]) -- the Q18 crossover check.

SCALE is held fixed at every alpha by the q29_scalefixed.py one-liner.

ANCHOR REGIME (stated explicitly, per review fix (C)): the simulator's anchor is CNV.
Per-cell CNV is not stored for the dissociated cells (they are disjoint from the spatial
cells), so it is regenerated exactly as simulate_controlled.py generates it:
    y_i = cnv_clone[clade(i)] + N(0, cnv_noise=0.25)
Spot side is the simulator's own measured cnv_spot. This is the DISCRETE/CLONAL regime.
It is the only anchor regime this simulator (or any real dataset in this project) offers.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax, jax.numpy as jnp
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
CNV_NOISE = 0.25          # simulate_controlled.py default
N_PAIR_SUB = 200_000      # subsample for the all-pairs rank statistic
N_TRIPLE = 400_000        # subsample for repair (B)
# anchor-degradation sweep: multiplier on BOTH the per-cell and per-spot anchor noise.
# mult=1 is the simulator's own setting, at which the anchor alone already maps every cell
# to a correct-clade spot (anchor-only accuracy 1.000) -- i.e. it is an oracle-grade answer
# key, so a criterion scored against it is not a fair test. Higher multipliers degrade it.
NOISE_MULT = [1, 4, 8, 16, 32]


def spearman(x, y):
    if len(x) < 10:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def criteria(gn, p, logp, d2, Wpairs_idx, wc_idx, all_idx, tri, cidx):
    """All candidate variants for one coupling."""
    out = {}
    ghat = gn / np.maximum(gn.sum(1, keepdims=True), 1e-300)
    n, m = gn.shape
    # --- S3 family (b uniform -> D_b^{-1/2} is a constant, drops out of any rank stat) ---
    W = gn @ gn.T
    E = p @ p.T
    i0, i1 = wc_idx
    out["S3_within"] = spearman(W[i0, i1], E[i0, i1])
    j0, j1 = all_idx
    out["S3_all"] = spearman(W[j0, j1], E[j0, j1])
    # paired negative control (i): E from cell-cell anchor similarity
    out["S3_cc_within"] = spearman(W[i0, i1], Ecc[i0, i1])
    out["S3_cc_all"] = spearman(W[j0, j1], Ecc[j0, j1])
    # --- repair (A): keep the spot index, cell resolution, first order ---
    out["S_A"] = float((ghat * logp).sum() / n)
    # sigma-free rank version, averaged over cells
    rg = rankdata(gn, axis=1); rp = rankdata(-d2, axis=1)
    rg = rg - rg.mean(1, keepdims=True); rp = rp - rp.mean(1, keepdims=True)
    num = (rg * rp).sum(1)
    den = np.sqrt((rg * rg).sum(1) * (rp * rp).sum(1))
    out["S_Arank"] = float(np.mean(num / np.maximum(den, 1e-300)))
    # --- repair (B): triples, j retained ---
    ti, tj, ts = tri
    out["S_B"] = spearman(gn[ti, ts] * gn[tj, ts], p[ti, ts] * p[tj, ts])
    # --- diagnostics ---
    out["row_entropy"] = float(np.mean(-(ghat * np.log(ghat + 1e-300)).sum(1)))
    return out


rows = []
t0 = time.time()
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"])
    n = len(cidx)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    X_sc = d["X_sc"]

    # ---------------- anchor (stated regime: clonal CNV) ----------------
    crng = np.random.default_rng(770000 + seed)
    nb_ = d["cnv_clone"].shape[1]
    base_cell = np.asarray(d["cnv_clone"], np.float64)[cidx]
    Y_spot0 = np.asarray(d["cnv_spot"], np.float64)
    e_cell = crng.normal(0, 1.0, size=(n, nb_))
    e_spot = crng.normal(0, 1.0, size=Y_spot0.shape)
    PS, ACC_A = {}, {}
    for mlt in NOISE_MULT:
        yc = base_cell + CNV_NOISE * mlt * e_cell
        Ys = Y_spot0 + 0.15 * np.sqrt(max(mlt ** 2 - 1, 0)) * e_spot
        dd = np.maximum((yc ** 2).sum(1)[:, None] + (Ys ** 2).sum(1)[None, :] - 2.0 * yc @ Ys.T, 0.0)
        s2 = float(np.median(dd)) / 2.0
        zz = -dd / (2.0 * s2); zz -= zz.max(1, keepdims=True)
        pp_ = np.exp(zz); pp_ /= pp_.sum(1, keepdims=True)
        PS[mlt] = (pp_, np.log(pp_ + 1e-300), dd, yc, s2)
        ACC_A[mlt] = float((spot_ci[pp_.argmax(1)] == cidx).mean())
    p, logp, d2, y_cell, sigma2 = PS[1]
    # cell-cell anchor kernel for the paired negative control (i)
    dcc = ((y_cell ** 2).sum(1)[:, None] + (y_cell ** 2).sum(1)[None, :] - 2.0 * y_cell @ y_cell.T)
    Ecc = np.exp(-np.maximum(dcc, 0.0) / (4.0 * sigma2))
    anchor_only_acc = ACC_A[1]

    # pair / triple index sets (fixed across alpha and condition)
    prng = np.random.default_rng(880000 + seed)
    iu, ju = np.triu_indices(n, 1)
    wc = cidx[iu] == cidx[ju]
    wc_idx = (iu[wc], ju[wc])
    sub = prng.choice(len(iu), size=min(N_PAIR_SUB, len(iu)), replace=False)
    all_idx = (iu[sub], ju[sub])
    m = len(spot_ci)
    tri = (prng.integers(0, n, N_TRIPLE), prng.integers(0, n, N_TRIPLE),
           prng.integers(0, m, N_TRIPLE))

    Cf, C_T0, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(d["X_st"]),
                                              d["C_tree"], d["C_space"])
    # shuffled-tree control: permute leaf labels of the patristic matrix
    srng = np.random.default_rng(990000 + seed)
    perm = srng.permutation(n)
    C_T_shuf = jnp.asarray(np.asarray(d["C_tree"])[np.ix_(perm, perm)])

    for cond, C_T in (("real", C_T0), ("shuffled", C_T_shuf)):
        omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
        sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

        def cost(av, g):
            return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                                Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                                equalize="measured")
        REF = float(jnp.median(jnp.abs(cost(0.0, g0))))

        for av in GRID:
            g = g0
            for _ in range(J):
                C = cost(av, g)
                C = C * (REF / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))   # q29 scale fix
                g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
            gn = np.asarray(g, np.float64)
            rec = dict(seed=seed, cond=cond, alpha=av, perm_spots=False,
                       acc=float((spot_ci[gn.argmax(1)] == cidx).mean()),
                       anchor_only_acc=anchor_only_acc)
            rec.update(criteria(gn, p, logp, d2, None, wc_idx, all_idx, tri, cidx))
            ghat_ = gn / np.maximum(gn.sum(1, keepdims=True), 1e-300)
            for mlt in NOISE_MULT:
                pp_, lp_, dd_, _, _ = PS[mlt]
                rec[f"S_A_n{mlt}"] = float((ghat_ * lp_).sum() / n)
                rg = rankdata(gn, axis=1); rp = rankdata(-dd_, axis=1)
                rg = rg - rg.mean(1, keepdims=True); rp = rp - rp.mean(1, keepdims=True)
                rec[f"S_Arank_n{mlt}"] = float(np.mean((rg * rp).sum(1) / np.maximum(
                    np.sqrt((rg * rg).sum(1) * (rp * rp).sum(1)), 1e-300)))
                rec[f"anchor_acc_n{mlt}"] = ACC_A[mlt]
            rows.append(rec)

            # ---- control 1: spot-permutation invariance (b uniform => one equal-b group) ----
            if cond == "real":
                pp = np.random.default_rng(int(1e6 + seed * 100 + av * 100)).permutation(gn.shape[1])
                gp = gn[:, pp]
                rec2 = dict(seed=seed, cond=cond, alpha=av, perm_spots=True,
                            acc=float((spot_ci[gp.argmax(1)] == cidx).mean()),
                            anchor_only_acc=anchor_only_acc)
                rec2.update(criteria(gp, p, logp, d2, None, wc_idx, all_idx, tri, cidx))
                rows.append(rec2)
    print(f"  seed {seed} done  ({time.time()-t0:.0f}s)", flush=True)

r = pd.DataFrame(rows)
r.to_csv(HERE / "cand_spot_routed_colocation_rank.csv", index=False)

# ============================== reporting ==============================
CRITS = ["S3_within", "S3_all", "S3_cc_within", "S3_cc_all", "S_A", "S_Arank", "S_B"]
main = r[(~r.perm_spots)].copy()
real = main[main.cond == "real"]
shuf = main[main.cond == "shuffled"]
seeds = sorted(real.seed.unique())
print("\n" + "=" * 100)
print(f"seeds used: {seeds}  (n={len(seeds)})")
print(f"anchor-only map accuracy (argmax_j p[i,j]) : "
      f"{real.groupby('seed').anchor_only_acc.first().mean():.4f}   "
      f"[Q18 crossover check]")

print("\n--- TRUE ACCURACY, real tree, per seed ---")
print("seed  " + "".join(f"{'a='+str(v):>9}" for v in GRID))
for s_ in seeds:
    g_ = real[real.seed == s_].set_index("alpha").sort_index()
    print(f"{s_:>4}  " + "".join(f"{g_.acc[v]:>9.4f}" for v in GRID))

mono = lambda x: bool((np.diff(np.asarray(x)) > 0).all() or (np.diff(np.asarray(x)) < 0).all())
summary = {}
for crit in CRITS:
    print(f"\n--- S = {crit} ---")
    print("seed  " + "".join(f"{'a='+str(v):>9}" for v in GRID) + "   argmaxS  argmaxACC  mono")
    per = []
    for s_ in seeds:
        g_ = real[real.seed == s_].set_index("alpha").sort_index()
        v = g_[crit]
        if not np.isfinite(v.values).all():
            continue
        sel, opt = float(v.idxmax()), float(g_.acc.idxmax())
        mo = mono(v.values)
        print(f"{s_:>4}  " + "".join(f"{v[a]:>9.4f}" for a in GRID) +
              f"   {sel:>7.2f}  {opt:>9.2f}  {str(mo):>5}")
        per.append(dict(seed=s_, sel=sel, opt=opt, mono=mo, acc_sel=float(g_.acc[sel]),
                        acc_a0=float(g_.acc[0.0]), acc_a02=float(g_.acc[0.20]),
                        acc_opt=float(g_.acc.max())))
    P = pd.DataFrame(per)
    # shuffled-tree control
    sper = []
    for s_ in seeds:
        g_ = shuf[shuf.seed == s_].set_index("alpha").sort_index()
        v = g_[crit]
        if np.isfinite(v.values).all():
            sper.append(float(v.idxmax()))
    ag = int((np.abs(P.sel - P.opt) < 1e-9).sum())
    summary[crit] = dict(
        agreement_n=ag, agreement_total=len(P),
        mean_abs_err=float(np.abs(P.sel - P.opt).mean()),
        acc_sel=float(P.acc_sel.mean()), acc_a0=float(P.acc_a0.mean()),
        acc_a02=float(P.acc_a02.mean()), acc_opt=float(P.acc_opt.mean()),
        monotone_seeds=int(P.mono.sum()), sel_list=list(P.sel),
        shuf_sel=sper, shuf_mean=float(np.mean(sper)) if sper else np.nan)
    S_ = summary[crit]
    print(f"  agreement argmax(S)==argmax(acc) : {ag}/{len(P)}")
    print(f"  mean |selected - optimal| alpha  : {S_['mean_abs_err']:.4f}")
    print(f"  monotone in alpha                : {S_['monotone_seeds']}/{len(P)} seeds")
    print(f"  acc @selected {S_['acc_sel']:.4f} | a=0 {S_['acc_a0']:.4f} | "
          f"a=0.2 {S_['acc_a02']:.4f} | oracle {S_['acc_opt']:.4f}")
    print(f"  BEATS a=0: {S_['acc_sel'] > S_['acc_a0']}   BEATS a=0.2: {S_['acc_sel'] > S_['acc_a02']}")
    print(f"  SHUFFLED-TREE selected alpha     : {', '.join(f'{x:.2f}' for x in sper)}"
          f"   (mean {S_['shuf_mean']:.3f} vs real {np.mean(S_['sel_list']):.3f})")

print("\n--- PAIRED win/loss of the selected alpha vs fixed alpha=0.2 (real tree) ---")
for crit in CRITS:
    w = l = t = 0; dif = []
    for s_ in seeds:
        g_ = real[real.seed == s_].set_index("alpha").sort_index()
        v = g_[crit]
        if not np.isfinite(v.values).all():
            continue
        dd_ = float(g_.acc[float(v.idxmax())] - g_.acc[0.20]); dif.append(dd_)
        w += dd_ > 1e-9; l += dd_ < -1e-9; t += abs(dd_) <= 1e-9
    print(f"  {crit:<14} wins {w}  losses {l}  ties {t}   mean delta {np.mean(dif):+.5f}")

print("\n--- ANCHOR-DEGRADATION SWEEP for repair (A) (real tree) ---")
print("  the anchor at mult=1 is oracle-grade; a criterion scored against it is scoring")
print("  against an answer key. This sweep asks whether the repair survives a usable anchor.")
hdr = f"{'mult':>5} {'anchorACC':>10} {'agree':>7} {'|err|':>7} {'mono':>5} {'accSel':>8} {'vs a=0':>8} {'vs a=.2':>8} {'shufSel':>8}"
for tag in ("S_A", "S_Arank"):
    print(f"  -- {tag} --")
    print("  " + hdr)
    for mlt in NOISE_MULT:
        col = f"{tag}_n{mlt}"
        sel, opt, accs, mo = [], [], [], 0
        for s_ in seeds:
            g_ = real[real.seed == s_].set_index("alpha").sort_index()
            v = g_[col]
            sel.append(float(v.idxmax())); opt.append(float(g_.acc.idxmax()))
            accs.append(float(g_.acc[float(v.idxmax())])); mo += mono(v.values)
        ssel = []
        for s_ in seeds:
            g_ = shuf[shuf.seed == s_].set_index("alpha").sort_index()
            ssel.append(float(g_[col].idxmax()))
        aa = real.groupby('seed')[f'anchor_acc_n{mlt}'].first().mean()
        print(f"  {mlt:>5} {aa:>10.4f} "
              f"{int(np.sum(np.abs(np.array(sel)-np.array(opt))<1e-9)):>4}/10 "
              f"{np.mean(np.abs(np.array(sel)-np.array(opt))):>7.3f} {mo:>4}/10 "
              f"{np.mean(accs):>8.4f} {np.mean(accs)-real[real.alpha==0.0].acc.mean():>+8.4f} "
              f"{np.mean(accs)-real[real.alpha==0.20].acc.mean():>+8.4f} {np.mean(ssel):>8.3f}")

print("\n--- CONTROL 1: SPOT-PERMUTATION INVARIANCE (real tree) ---")
pv = r[r.cond == "real"]
for av in (0.20, 0.50):
    A_ = pv[(~pv.perm_spots) & (pv.alpha == av)].set_index("seed").sort_index()
    B_ = pv[(pv.perm_spots) & (pv.alpha == av)].set_index("seed").sort_index()
    print(f"  alpha={av}:  acc {A_.acc.mean():.4f} -> {B_.acc.mean():.4f}   "
          f"(random floor ~{1.0/12:.4f})")
    for crit in CRITS:
        dd = float(np.max(np.abs(A_[crit].values - B_[crit].values)))
        print(f"      max|dS| over seeds  {crit:<14} = {dd:.3e}"
              f"{'   <-- INVARIANT (blind to the error)' if dd < 1e-9 else ''}")

print("\n--- CONTROL 3: coupling row entropy vs alpha (real tree, mean over seeds) ---")
ent = real.groupby("alpha").row_entropy.mean()
print("  " + "  ".join(f"a={a:.2f}:{ent[a]:.4f}" for a in GRID))
print(f"  entropy monotone in alpha: {mono(ent.values)}")
print(f"\ntotal wall time {time.time()-t0:.0f}s")
