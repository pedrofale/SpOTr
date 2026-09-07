"""Candidate: dual-perturbation-instability (DPI).

TWO VERSIONS ARE COMPUTED AND BOTH REPORTED.

  V1 = AS SPECIFIED. V(alpha) = V_E(alpha) + V_T(alpha), instability of the hard placement
       under (E) expression data-fission and (T) GW-reference half-sampling. alpha* = argmin V.
       Also the "robust variant": alpha* where V_E crosses V_T.

  V2 = THE REPAIRED VERSION the review asks for under fixes[1] ("REMOVE THE KNOB by testing
       instead of optimising"). For each alpha the observed tree-side subsample dispersion
       V_T(alpha; p) is compared against a PARAMETRIC-BOOTSTRAP NULL in which the tree is
       correctly specified by construction: draw cell positions z_i ~ Cat(gamma_i) from the
       fitted map at that alpha, build C_tree^null = pairwise ||z_i - z_j||, quantile-match it
       to the real C_tree's off-diagonal distance spectrum (so only CONCORDANCE differs, not
       the spectrum), and re-run the SAME reference sub-sampling at the SAME p. p cancels
       between observed and null. alpha* = the largest alpha at which V_T^obs does not exceed
       every one of the R null replicates (rank test at level 1/(R+1)).

Also implemented from the review's `fixes`:
  [2] COMMON RANDOM NUMBERS: the same B sub-sample masks and the same B expression-noise
      draws are reused at every alpha, in every arm (real / shuffled / null).
  [3] the mean normalised top-2 gap of gamma is reported next to V at every alpha, so
      degeneracy-driven and evidence-driven instability can be told apart.
  [4] the per-clade arm is DROPPED (global arm only), as instructed.

CONTROLS RUN (review test_spec + task obstacle 2):
  K1  leaf-shuffled C_tree (rows+cols permuted -> exact same distance spectrum, zero
      cell-identity concordance), same criterion, same masks.
  K2  knob sensitivity: p = 0.5 vs p = 0.85 reference sub-sampling.
  K3  endpoint identities: V_T(alpha=0) and V_E(alpha=1) to solver tolerance.

SCALE is held fixed at every alpha by the q29_scalefixed.py one-liner.
"""
import sys, time, json
from pathlib import Path
from functools import partial
import numpy as np, pandas as pd, anndata as ad, jax, jax.numpy as jnp, ot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.spotr import _dc_rect

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
B      = int(sys.argv[1]) if len(sys.argv) > 1 else 6     # perturbation replicates (CRN)
R      = int(sys.argv[2]) if len(sys.argv) > 2 else 5     # null replicates
SEEDS  = None if len(sys.argv) < 4 else [int(x) for x in sys.argv[3].split(",")]
TAU_E  = 1.0        # expression data-fission noise multiple (knob; affects V1 only)
P_MAIN = 0.5
P_ALT  = 0.85

# ---------------------------------------------------------------- solver ----
@jax.jit
def solve(alpha, Cf, C_T, C_S, a, b, omega, sF, sL, REF, s_mask):
    """One FGW solve. s_mask in {0,1}^n selects which cells voice a tree opinion
    (all cells are still placed).  s_mask = ones -> the unperturbed solver."""
    F_hat = _dc_rect(Cf, a, b) / sF
    CT2, CS2 = C_T * C_T, C_S * C_S
    a_s = a * s_mask
    Z = jnp.maximum(a_s.sum(), 1e-12)
    a_sub = a_s / Z
    cnt_k = jnp.maximum(omega.sum(axis=0), 1e-8)
    row_t = (CT2 @ a_sub)[:, None]
    col_t = (CS2 @ b)[None, :]

    def step(_, g):
        g_sub = (g * s_mask[:, None]) / Z
        L = row_t + col_t - 2.0 * (C_T @ g_sub @ C_S.T)
        L_hat = _dc_rect(L, a, b) / sL
        C0 = (1.0 - alpha) * F_hat + alpha * L_hat
        m_i = jnp.median(jnp.abs(C0), axis=1)
        m_k = jax.lax.stop_gradient(jnp.clip((omega.T @ m_i) / cnt_k, 1e-6, 1e6))
        s_k = jnp.clip(jax.lax.stop_gradient(jnp.median(m_k)) / m_k, 0.5, 2.0)
        C = C0 * (omega @ s_k)[:, None]
        C = C * (REF / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))   # q29 scale fix
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        return g
    g = jax.lax.fori_loop(0, J, step, a[:, None] * b[None, :])
    return g

@jax.jit
def ref_median(Cf, C_T, C_S, a, b, omega, sF, sL):
    """median|C| of the alpha=0 cost at gamma0 -- the fixed scale target."""
    F_hat = _dc_rect(Cf, a, b) / sF
    C0 = F_hat
    cnt_k = jnp.maximum(omega.sum(axis=0), 1e-8)
    m_i = jnp.median(jnp.abs(C0), axis=1)
    m_k = jnp.clip((omega.T @ m_i) / cnt_k, 1e-6, 1e6)
    s_k = jnp.clip(jnp.median(m_k) / m_k, 0.5, 2.0)
    return jnp.median(jnp.abs(C0 * (omega @ s_k)[:, None]))

# ------------------------------------------------------------- statistics ---
def instability(place, D, w=None):
    """place: [B, n] hard spot indices. D: [m, m] spot distances.
    Returns (V, V_gapweighted): mean pairwise displacement of each cell's placement
    across replicates, unweighted and weighted by the cell's top-2 gap (review fix 3,
    so entropic tie-flipping among near-equal spots does not dominate)."""
    Bn = place.shape[0]
    tot = np.zeros(place.shape[1]); npair = 0
    for x in range(Bn):
        for y in range(x + 1, Bn):
            tot += D[place[x], place[y]]; npair += 1
    per = tot / max(npair, 1)
    V = float(per.mean())
    Vw = float((per * w).sum() / max(w.sum(), 1e-12)) if w is not None else np.nan
    return V, Vw

def top2gap_cell(g):
    p = np.sort(g, axis=1)
    return (p[:, -1] - p[:, -2]) / np.maximum(p[:, -1] + p[:, -2], 1e-30)

def qmatch(Draw, sorted_ref, rng):
    """quantile-match Draw's off-diagonal entries onto sorted_ref (ties broken at random)."""
    n = Draw.shape[0]
    iu = np.triu_indices(n, 1)
    v = Draw[iu] + rng.normal(0, 1e-6, iu[0].size)   # random tie-breaking
    out = np.zeros(v.size, np.float32)
    out[np.argsort(v)] = sorted_ref
    M = np.zeros((n, n), np.float32)
    M[iu] = out; M = M + M.T
    return M

# ------------------------------------------------------------------- main ---
rows = []
t0 = time.time()
all_seeds = [s for s in range(14) if (HERE / f"sim_ctrl_s{s}.npz").exists()
             and bool(np.load(HERE / f"sim_ctrl_s{s}.npz", allow_pickle=True)["in_band"][0])]
if SEEDS is not None:
    all_seeds = [s for s in all_seeds if s in SEEDS]
print(f"seeds: {all_seeds}   B={B} R={R}", flush=True)

for seed in all_seeds:
    d = np.load(HERE / f"sim_ctrl_s{seed}.npz", allow_pickle=True)
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"]); n = len(cidx)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    X_sc = np.asarray(d["X_sc"], np.float32); X_st = np.asarray(d["X_st"], np.float32)
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    REF = ref_median(Cf, C_T, C_S, a, b, omega, sF, sL)
    ones = jnp.ones(n)

    xy = np.asarray(d["xy_st"], np.float64)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    D_rand = float(D.mean())            # uniform b => sum_jl b_j b_l ||x_j-x_l|| = mean(D)

    rng = np.random.default_rng(1000 + seed)
    # ---- COMMON RANDOM NUMBERS: fixed across every alpha and every arm ----
    masks_p5  = jnp.asarray(np.stack([(rng.random(n) < P_MAIN).astype(np.float32) for _ in range(B)]))
    masks_p85 = jnp.asarray(np.stack([(rng.random(n) < P_ALT ).astype(np.float32) for _ in range(B)]))
    # expression data fission: per-gene within-clade residual sd (honest, no ground truth)
    res = X_sc - np.stack([X_sc[cidx == k].mean(0) for k in range(K)])[cidx]
    sd_g = res.std(0) + 1e-8
    Cf_Efis, Cf_Egb = [], []
    for _ in range(B):
        ea = rng.normal(0, 1, X_sc.shape).astype(np.float32) * sd_g * TAU_E
        eb = rng.normal(0, 1, X_st.shape).astype(np.float32) * sd_g * TAU_E
        cf = ot.dist(X_sc + ea, X_st + eb); Cf_Efis.append(jnp.asarray(cf / cf.max()))
        gsel = rng.integers(0, X_sc.shape[1], X_sc.shape[1])   # gene bootstrap (no free scale)
        cf = ot.dist(X_sc[:, gsel], X_st[:, gsel]); Cf_Egb.append(jnp.asarray(cf / cf.max()))
    perm = rng.permutation(n)
    C_T_shuf = jnp.asarray(np.asarray(C_T)[perm][:, perm])
    sorted_ref = np.sort(np.asarray(C_T)[np.triu_indices(n, 1)])

    for av in GRID:
        al = jnp.float32(av)
        rec = dict(seed=seed, alpha=av)
        for arm, CT in (("real", C_T), ("shuf", C_T_shuf)):
            g_base = solve(al, Cf, CT, C_S, a, b, omega, sF, sL, REF, ones)
            gb = np.asarray(g_base)
            wgap = top2gap_cell(gb)
            rec[f"acc_{arm}"]  = float((spot_ci[gb.argmax(1)] == cidx).mean())
            rec[f"gap_{arm}"]  = float(wgap.mean())
            # ---- (T) reference half-sampling, observed ----
            for tag, msk in (("p5", masks_p5), ("p85", masks_p85)):
                pl = np.stack([np.asarray(solve(al, Cf, CT, C_S, a, b, omega, sF, sL, REF, msk[t])).argmax(1)
                               for t in range(B)])
                v, vw = instability(pl, D, wgap)
                rec[f"VT_{arm}_{tag}"], rec[f"VTw_{arm}_{tag}"] = v / D_rand, vw / D_rand
            # ---- (E) expression perturbation, observed (real arm only) ----
            if arm == "real":
                for tag, CfE in (("fis", Cf_Efis), ("gb", Cf_Egb)):
                    pl = np.stack([np.asarray(solve(al, CfE[t], CT, C_S, a, b, omega, sF, sL, REF, ones)).argmax(1)
                                   for t in range(B)])
                    v, vw = instability(pl, D, wgap)
                    rec[f"VE_{tag}"], rec[f"VEw_{tag}"] = v / D_rand, vw / D_rand
                rec["VE"] = rec["VE_gb"]
            # ---- NULL: tree correctly specified by construction ----
            pr = gb / np.maximum(gb.sum(1, keepdims=True), 1e-30)
            cum = np.cumsum(pr, 1)
            vnull = []
            for r in range(R):
                u = rng.random((n, 1))
                jsel = (u > cum).sum(1).clip(0, len(b) - 1)
                z = xy[jsel]
                Dn = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)).astype(np.float32)
                CTn = jnp.asarray(qmatch(Dn, sorted_ref, rng))
                pl = np.stack([np.asarray(solve(al, Cf, CTn, C_S, a, b, omega, sF, sL, REF, masks_p5[t])).argmax(1)
                               for t in range(B)])
                vnull.append(instability(pl, D, wgap)[0] / D_rand)
            rec[f"VTnull_{arm}_mean"] = float(np.mean(vnull))
            rec[f"VTnull_{arm}_max"]  = float(np.max(vnull))
            rec[f"VTnull_{arm}_sd"]   = float(np.std(vnull, ddof=1)) if R > 1 else 0.0
        rows.append(rec)
        print(f"  seed {seed} alpha {av:.2f}  VEgb={rec['VE_gb']:.4f} VEfis={rec['VE_fis']:.4f} VT={rec['VT_real_p5']:.4f} "
              f"null={rec['VTnull_real_mean']:.4f}/{rec['VTnull_real_max']:.4f} "
              f"VTshuf={rec['VT_shuf_p5']:.4f} nullshuf={rec['VTnull_shuf_max']:.4f} "
              f"acc={rec['acc_real']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- K3: endpoint identity check, alpha=1 (E cannot move the coupling) ----
    if seed == all_seeds[0]:
        pl = np.stack([np.asarray(solve(jnp.float32(1.0), Cf_Egb[t], C_T, C_S, a, b, omega, sF, sL, REF, ones)).argmax(1)
                       for t in range(B)])
        print(f"  K3: V_E(alpha=1.0) = {instability(pl, D)[0]/D_rand:.6g}  (identity claim: exactly 0)", flush=True)

r = pd.DataFrame(rows)
r.to_csv(HERE / "cand_dual_perturbation_instability.csv", index=False)
print(f"\nwrote CSV, {time.time()-t0:.0f}s total\n" + "=" * 100)

# =============================== ANALYSIS =====================================
def sel_v1(g, ecol="VE_gb", tcol="VT_real_p5"):   # ORIGINAL spec: argmin V_E + V_T
    v = g[ecol].values + g[tcol].values
    return float(g.alpha.values[int(np.argmin(v))]), v

def sel_v1_cross(g):                 # ORIGINAL "robust variant": V_E == V_T crossing
    dv = g.VE_gb.values - g.VT_real_p5.values
    idx = np.where(np.sign(dv[:-1]) != np.sign(dv[1:]))[0]
    if len(idx) == 0: return float(g.alpha.values[int(np.argmin(np.abs(dv)))])
    i = idx[0]
    a0, a1 = g.alpha.values[i], g.alpha.values[i + 1]
    t = abs(dv[i]) / max(abs(dv[i]) + abs(dv[i + 1]), 1e-12)
    return float(a0 + t * (a1 - a0))

def sel_v2(g, arm):                  # REPAIRED: largest alpha not rejecting the null
    obs = g[f"VT_{arm}_p5"].values; nul = g[f"VTnull_{arm}_max"].values
    ok = obs <= nul + 1e-12
    sel = 0.0
    for i, a_ in enumerate(g.alpha.values):
        if ok[i]: sel = float(a_)
        else: break
    return sel

def mono(x):
    dx = np.diff(np.asarray(x)); return bool((dx > 0).all() or (dx < 0).all())

per = []
for s_, g in r.groupby("seed"):
    g = g.sort_values("alpha").reset_index(drop=True)
    v1, vcurve = sel_v1(g)
    v1f, _ = sel_v1(g, "VE_fis")
    v1w, vwcurve = sel_v1(g, "VEw_gb", "VTw_real_p5")
    per.append(dict(seed=s_,
        v1=v1, v1f=v1f, v1w=v1w, v1x=sel_v1_cross(g),
        v2=sel_v2(g, "real"), v2s=sel_v2(g, "shuf"),
        v2_p85=None,
        argmax_acc=float(g.alpha[g.acc_real.idxmax()]),
        argmax_acc_shuf=float(g.alpha[g.acc_shuf.idxmax()]),
        acc_v1=float(g.acc_real[g.alpha == v1].iloc[0]),
        acc_v1f=float(g.acc_real[g.alpha == v1f].iloc[0]),
        acc_v1w=float(g.acc_real[g.alpha == v1w].iloc[0]),
        acc_v2=float(g.acc_real[g.alpha == sel_v2(g, "real")].iloc[0]),
        acc_a0=float(g.acc_real[g.alpha == 0.0].iloc[0]),
        acc_a02=float(g.acc_real[g.alpha == 0.20].iloc[0]),
        acc_opt=float(g.acc_real.max()),
        mono_V=mono(vcurve), mono_Vw=mono(vwcurve), mono_VT=mono(g.VT_real_p5.values),
        mono_VE=mono(g.VE_gb.values),
        VT0=float(g.VT_real_p5[g.alpha == 0.0].iloc[0])))
    # K2 knob sensitivity: same rule with p=0.85
    obs = g["VT_real_p85"].values; nul = g["VTnull_real_max"].values
    sel = 0.0
    for i, a_ in enumerate(g.alpha.values):
        if obs[i] <= nul[i] + 1e-12: sel = float(a_)
        else: break
    per[-1]["v2_p85"] = sel
p = pd.DataFrame(per)
p.to_csv(HERE / "cand_dual_perturbation_instability_per_seed.csv", index=False)

print("PER-SEED CURVES  (V = V_E + V_T  |  VT_obs / VTnull_max  |  accuracy)")
hdr = "".join(f"{'a='+str(v):>18}" for v in GRID)
print(f"{'seed':>5}{hdr}")
for s_, g in r.groupby("seed"):
    g = g.sort_values("alpha").set_index("alpha")
    print(f"{s_:>5}" + "".join(f"{g.VE[v]+g.VT_real_p5[v]:>6.3f}{g.VT_real_p5[v]:>6.3f}/{g.VTnull_real_max[v]:.3f}" for v in GRID))
    print(f"{'':>5}" + "".join(f"{'acc':>12}{g.acc_real[v]:>6.3f}" for v in GRID))
print("=" * 100)

def block(name, selcol, acccol):
    sel = p[selcol].values; opt = p.argmax_acc.values
    ag = int((np.abs(sel - opt) < 1e-9).sum())
    print(f"\n--- {name} ---")
    print(f"  selected alpha per seed : {', '.join(f'{v:.2f}' for v in sel)}")
    print(f"  argmax accuracy per seed: {', '.join(f'{v:.2f}' for v in opt)}")
    print(f"  agreement               : {ag}/{len(p)}")
    print(f"  mean |selected - optimal|: {np.abs(sel-opt).mean():.3f}")
    print(f"  mean acc @selected {p[acccol].mean():.4f} | alpha=0 {p.acc_a0.mean():.4f} | "
          f"fixed 0.2 {p.acc_a02.mean():.4f} | oracle {p.acc_opt.mean():.4f}")
    print(f"  beats alpha=0: {p[acccol].mean() > p.acc_a0.mean()}   "
          f"beats fixed 0.2: {p[acccol].mean() > p.acc_a02.mean()}")
    return ag

ag1  = block("V1  AS SPECIFIED, E = gene bootstrap (argmin V_E+V_T)", "v1", "acc_v1")
ag1f = block("V1b AS SPECIFIED, E = Gaussian data fission tau=1 (E-side knob control)", "v1f", "acc_v1f")
ag1w = block("V1c AS SPECIFIED + review fix 3 (top-2-gap-weighted V)", "v1w", "acc_v1w")
print(f"  robust-variant crossing V_E=V_T per seed: {', '.join(f'{v:.3f}' for v in p.v1x)}")
ag2 = block("V2  REPAIRED (largest alpha not rejecting the correctly-specified-tree null)", "v2", "acc_v2")

print("\n--- MONOTONICITY ---")
print(f"  V = V_E+V_T monotone in alpha       : {int(p.mono_V.sum())}/{len(p)} seeds")
print(f"  V gap-weighted monotone in alpha    : {int(p.mono_Vw.sum())}/{len(p)} seeds")
print(f"  V_T alone monotone in alpha         : {int(p.mono_VT.sum())}/{len(p)} seeds")
print(f"  V_E alone monotone in alpha         : {int(p.mono_VE.sum())}/{len(p)} seeds")

print("\n--- DIAGNOSTIC: mean V_T curve, real vs leaf-shuffled tree (p=0.5) ---")
mt = r.groupby("alpha")[["VT_real_p5", "VT_shuf_p5", "VE_gb", "VE_fis",
                         "VTnull_real_mean", "VTnull_shuf_mean"]].mean()
print(f"{'':<20}" + "".join(f"{'a='+str(v):>10}" for v in GRID))
for c in mt.columns:
    print(f"{c:<20}" + "".join(f"{mt[c][v]:>10.4f}" for v in GRID))
print(f"{'argmax VT real':<20} {mt.VT_real_p5.idxmax():.2f}   argmax VT shuffled {mt.VT_shuf_p5.idxmax():.2f}")

print("\n--- K3  ENDPOINT IDENTITIES ---")
print(f"  V_T(alpha=0) per seed: {', '.join(f'{v:.2e}' for v in p.VT0)}   (identity claim: exactly 0)")

print("\n--- K1  SHUFFLED-TREE CONTROL (rows+cols of C_tree permuted; spectrum preserved) ---")
print(f"  V2 selected alpha, REAL tree     : {', '.join(f'{v:.2f}' for v in p.v2)}   mean {p.v2.mean():.3f}")
print(f"  V2 selected alpha, SHUFFLED tree : {', '.join(f'{v:.2f}' for v in p.v2s)}  mean {p.v2s.mean():.3f}")
print(f"  argmax accuracy on shuffled tree : {', '.join(f'{v:.2f}' for v in p.argmax_acc_shuf)}")
same = int((np.abs(p.v2 - p.v2s) < 1e-9).sum())
print(f"  identical selection real vs shuffled: {same}/{len(p)} seeds")
# V1 under the shuffled tree
v1s = []
for s_, g in r.groupby("seed"):
    g = g.sort_values("alpha").reset_index(drop=True)
    v = g.VE.values + g.VT_shuf_p5.values
    v1s.append(float(g.alpha.values[int(np.argmin(v))]))
print(f"  V1 argmin, REAL tree             : {', '.join(f'{v:.2f}' for v in p.v1)}")
print(f"  V1 argmin, SHUFFLED tree         : {', '.join(f'{v:.2f}' for v in v1s)}")
print(f"  V1 identical real vs shuffled    : {int((np.abs(p.v1.values-np.array(v1s))<1e-9).sum())}/{len(p)}")

print("\n--- K2  KNOB SENSITIVITY (p = 0.5 vs p = 0.85) ---")
print(f"  V2 selected alpha, p=0.50 : {', '.join(f'{v:.2f}' for v in p.v2)}")
print(f"  V2 selected alpha, p=0.85 : {', '.join(f'{v:.2f}' for v in p.v2_p85)}")
print(f"  mean |difference|         : {np.abs(p.v2-p.v2_p85).mean():.3f}")
v1_p85 = []
for s_, g in r.groupby("seed"):
    g = g.sort_values("alpha").reset_index(drop=True)
    v = g.VE.values + g.VT_real_p85.values
    v1_p85.append(float(g.alpha.values[int(np.argmin(v))]))
print(f"  V1 argmin, p=0.50 : {', '.join(f'{v:.2f}' for v in p.v1)}")
print(f"  V1 argmin, p=0.85 : {', '.join(f'{v:.2f}' for v in v1_p85)}")
print(f"  V1 mean |difference| : {np.abs(p.v1.values-np.array(v1_p85)).mean():.3f}")

print("\n--- top-2 gap of gamma (degeneracy diagnostic, review fix 3) ---")
gg = r.groupby("alpha").gap_real.mean()
print("  " + "  ".join(f"a={v:.2f}: {gg[v]:.4f}" for v in GRID))

json.dump(dict(agreement_v1=ag1, agreement_v2=ag2, n=len(p),
               v2_real=list(p.v2), v2_shuf=list(p.v2s), v2_p85=list(p.v2_p85),
               v1_real=list(p.v1), v1_shuf=v1s,
               acc_v1=float(p.acc_v1.mean()), acc_v2=float(p.acc_v2.mean()),
               acc_a0=float(p.acc_a0.mean()), acc_a02=float(p.acc_a02.mean()),
               acc_opt=float(p.acc_opt.mean()),
               mae_v1=float(np.abs(p.v1-p.argmax_acc).mean()),
               mae_v2=float(np.abs(p.v2-p.argmax_acc).mean()),
               mono_V=int(p.mono_V.sum()), mono_VT=int(p.mono_VT.sum())),
          open(HERE / "cand_dual_perturbation_instability_summary.json", "w"), indent=1)
print("\ndone", flush=True)
