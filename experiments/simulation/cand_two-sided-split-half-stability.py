"""CANDIDATE 5 -- two-sided split-half stability, REPAIRED per the adversarial review.

S5(alpha) = sqrt( A_expr(alpha) * A_tree(alpha) )
  A_expr = agreement between couplings from two disjoint GENE halves (full tree)
  A_tree = agreement between couplings from two independently perturbed TREES (full genes)

Repairs applied (review "fixes"):
 (a) NO FREE KNOB. The tree perturbation magnitude is not chosen by the experimenter: it is
     bisected per seed so that median|DC(L_hat^1)-DC(L_hat^2)| equals median|DC(F_hat^1)-
     DC(F_hat^2)| measured from the gene split -- equal magnitude in the cost's own units.
     The perturbation is TOPOLOGICAL (clade-level distances jittered, tree REFIT by NJ, so the
     topology can change), not terminal-branch jitter, because the review shows additive
     jitter is annihilated by the double-centering.
 (b) DE-BIASED AGREEMENT + FROZEN SCALE + FROZEN ENTROPY.
     - every cost matrix, at every alpha and every split, is renormalised to a single constant
       median|C| = REF (q29 fix) so the endpoint pins are exact and eps means the same thing;
     - eps is retuned per alpha so the coupling's MEAN ROW ENTROPY is held at its alpha=0
       value, killing the blur drift that biases the cosine;
     - the cosine agreement is additionally blur-corrected analytically:
         agree_i = exp(-D_i^2 / (2(h^2+2w_i^2))), and self_i = q_i'K_s q_i = h^2/(h^2+2w_i^2),
       so the blur-free agreement at reference width h is agree_i ** (1/sqrt(self_i*self_i')).
       A hard-argmax spatial agreement (blur-free by construction) is reported as a third read.
 (c) not repairable -- "stability measures decisiveness not correctness" is exactly what the
     experiment tests, so it is left to the data.

Controls run:
 * SHUFFLED-TREE control (mandatory obstacle 2): the identical pipeline with every tree
   (real and perturbed) hit by one fixed leaf permutation -- distance spectrum preserved
   exactly, cell identity destroyed.
 * KNOB TEST (review test_spec iii): tree perturbation re-run at 0.5x and 2x its calibrated
   magnitude; if argmax S5 moves, S5 reports sigma_e/sigma_t and not alpha.
 * ZERO-SOLVE PRE-TEST (review test_spec): mean entropic assignment margin and mean row
   entropy vs alpha, from the baseline costs.

Reduced from spec: R=4 random splits instead of R=5, and the knob test on 4 seeds, to fit the
25-minute compute cap. Nothing else changed.
"""
import sys, time, json, os
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax, jax.numpy as jnp
import ot as pot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from simulate_controlled import neighbor_joining

EPS0, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
GRID_EXT = GRID + [1.0]          # 1.0 only as a mechanism diagnostic (A_tree(1)=c1)
R = 4
KNOB_SEEDS = 4
EPS_LO, EPS_HI, EPS_STEPS = 0.005, 0.20, 6


# ---------------------------------------------------------------- tree perturbation
def clade_blocks(C_tree, lab, K):
    """Recover the clade-level distance matrix and the within-clade fine structure."""
    D = np.zeros((K, K))
    for k in range(K):
        rk = np.where(lab == k)[0]
        for l in range(k + 1, K):
            rl = np.where(lab == l)[0]
            D[k, l] = D[l, k] = C_tree[np.ix_(rk, rl)].mean()
    W = {k: C_tree[np.ix_(np.where(lab == k)[0], np.where(lab == k)[0])].copy() for k in range(K)}
    return D, W


def offmean(M):
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)].mean()


def perturb_tree(D, W, lab, K, sigma, rng):
    """Jitter clade-level distances multiplicatively and REFIT the tree by NJ (topology can
    change), then re-expand to cells keeping the within-clade fine structure."""
    E = rng.normal(0.0, sigma, size=(K, K)); E = 0.5 * (E + E.T)
    M = D * np.exp(E); np.fill_diagonal(M, 0.0)
    P = neighbor_joining(M)
    P = P * (offmean(D) / max(offmean(P), 1e-12))       # match the distance spectrum scale
    Ct = P[np.ix_(lab, lab)].astype(np.float32)
    for k in range(K):
        rk = np.where(lab == k)[0]
        Ct[np.ix_(rk, rk)] = W[k]
    np.fill_diagonal(Ct, 0.0)
    return Ct


# ---------------------------------------------------------------- agreement
def agreements(G1, G2, Ks, C_space_np, h):
    Q1 = np.asarray(G1); Q1 = Q1 / np.maximum(Q1.sum(1, keepdims=True), 1e-30)
    Q2 = np.asarray(G2); Q2 = Q2 / np.maximum(Q2.sum(1, keepdims=True), 1e-30)
    cr = (Q1 * (Q2 @ Ks)).sum(1)
    s1 = np.maximum((Q1 * (Q1 @ Ks)).sum(1), 1e-30)
    s2 = np.maximum((Q2 * (Q2 @ Ks)).sum(1), 1e-30)
    ai = np.clip(cr / np.sqrt(s1 * s2), 1e-12, 1.0)
    raw = float(ai.mean())
    deb = float(np.mean(ai ** (1.0 / np.sqrt(s1 * s2))))     # blur-corrected to width h
    i1, i2 = Q1.argmax(1), Q2.argmax(1)
    hard = float(np.mean(np.exp(-C_space_np[i1, i2] ** 2 / (2 * h * h))))
    return raw, deb, hard


# ---------------------------------------------------------------- main
def run():
    rows, diag_rows = [], []
    t0 = time.time()
    seeds = []
    for seed in range(14):
        f = HERE / f"sim_ctrl_s{seed}.npz"
        if not f.exists():
            continue
        if not bool(np.load(f, allow_pickle=True)["in_band"][0]):
            continue
        seeds.append(seed)
    import os
    _lim = os.environ.get("CAND_SEEDS")
    if _lim: seeds = seeds[:int(_lim)]
    print(f"in-band seeds: {seeds}", flush=True)

    for seed in seeds:
        d = np.load(HERE / f"sim_ctrl_s{seed}.npz", allow_pickle=True)
        cidx, spot_ci = d["cell_clade"], d["spot_clade"]
        K = len(d["clades"])
        lab = np.asarray(cidx)
        A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
        omega = jnp.asarray(A)
        X_sc = np.asarray(d["X_sc"], np.float32); X_st = np.asarray(d["X_st"], np.float32)
        C_tree_real = np.asarray(d["C_tree"], np.float32)
        C_space_np = np.asarray(d["C_space"], np.float32)
        n_genes = X_sc.shape[1]

        Cf_full, _, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                                    C_tree_real, C_space_np)
        g0 = a[:, None] * b[None, :]
        an, bn = np.asarray(a), np.asarray(b)

        # spatial kernel
        Coff = C_space_np + np.eye(C_space_np.shape[0], dtype=np.float32) * 1e9
        h = float(np.median(Coff.min(1)))
        Ks = np.exp(-C_space_np ** 2 / (2 * h * h)).astype(np.float64)

        # ---- gene splits (shared by both modes) ----
        rngs = np.random.default_rng(1000 + seed)
        Cf_halves = []
        for r in range(R):
            perm = rngs.permutation(n_genes)
            g1, g2 = perm[: n_genes // 2], perm[n_genes // 2:]
            hs = []
            for gg in (g1, g2):
                Cfh = np.asarray(pot.dist(X_sc[:, gg], X_st[:, gg]), np.float32)
                Cfh = Cfh / Cfh.max()
                hs.append(jnp.asarray(Cfh))
            Cf_halves.append(tuple(hs))

        D_clade, Wblocks = clade_blocks(C_tree_real, lab, K)
        perm_shuf = np.random.default_rng(7000 + seed).permutation(len(lab))

        for mode in ("real", "shuffled"):
            def app(Ct):
                return Ct[np.ix_(perm_shuf, perm_shuf)] if mode == "shuffled" else Ct
            C_T = jnp.asarray(app(C_tree_real))
            sF, sL = S.make_cladexfgw_scales(Cf_full, C_T, C_S, a, b, g0, eps=EPS0)

            def dc(M):
                return np.asarray(S._dc_rect(jnp.asarray(M), a, b))

            def build(av, Cf, Ctree, gam):
                return build_perclade_alpha_cost_v2(
                    jnp.full((K,), av, dtype=jnp.float32), Cf, Ctree, C_S, a, b, gam,
                    omega, sF, sL, equalize="measured")

            REF = float(jnp.median(jnp.abs(build(0.0, Cf_full, C_T, g0))))

            def solve(av, Cf, Ctree, eps):
                g = g0
                for _ in range(J):
                    C = build(av, Cf, Ctree, g)
                    C = C * (REF / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))
                    g, _ = S.sinkhorn_unrolled_safe(C, a, b, eps, T)
                return g, C

            # ---- calibrate the tree perturbation magnitude to the gene-split magnitude ----
            feat_mag = float(np.mean([
                np.median(np.abs(dc(np.asarray(c1)) - dc(np.asarray(c2)))) / float(sF)
                for c1, c2 in Cf_halves]))

            def tree_mag(sigma, seed_off=0):
                rr = np.random.default_rng(50000 + 100 * seed + seed_off)
                vals = []
                for r in range(R):
                    T1 = app(perturb_tree(D_clade, Wblocks, lab, K, sigma, rr))
                    T2 = app(perturb_tree(D_clade, Wblocks, lab, K, sigma, rr))
                    L1 = np.asarray(S.compute_Lgw(jnp.asarray(T1), C_S, a, b, g0))
                    L2 = np.asarray(S.compute_Lgw(jnp.asarray(T2), C_S, a, b, g0))
                    vals.append(np.median(np.abs(dc(L1) - dc(L2))) / float(sL))
                return float(np.mean(vals))

            def calibrate(target):
                lo, hi = np.log(1e-3), np.log(5.0)
                best = None
                for _ in range(11):
                    mid = 0.5 * (lo + hi)
                    m = tree_mag(np.exp(mid))
                    if best is None or abs(np.log(m / target)) < abs(np.log(best[1] / target)):
                        best = (np.exp(mid), m)
                    if m > target:
                        hi = mid
                    else:
                        lo = mid
                return best

            sigma_cal, got = calibrate(feat_mag)
            sigmas = {"1x": sigma_cal}
            if mode == "real" and seed in seeds[:KNOB_SEEDS]:
                sigmas["0.5x"] = calibrate(0.5 * feat_mag)[0]
                sigmas["2x"] = calibrate(2.0 * feat_mag)[0]
            print(f"  seed {seed} [{mode}] feat_mag={feat_mag:.4f} sigma={sigma_cal:.4f} "
                  f"(tree_mag={got:.4f}) sigmas={ {k: round(v,4) for k,v in sigmas.items()} }",
                  flush=True)

            # perturbed tree pairs per magnitude
            tree_pairs = {}
            for tag, sg in sigmas.items():
                rr = np.random.default_rng(90000 + 100 * seed + hash(tag) % 97)
                tree_pairs[tag] = [(jnp.asarray(app(perturb_tree(D_clade, Wblocks, lab, K, sg, rr))),
                                    jnp.asarray(app(perturb_tree(D_clade, Wblocks, lab, K, sg, rr))))
                                   for _ in range(R)]

            # ---- alpha=0 reference entropy (target for eps tuning) ----
            g_ref, C_ref = solve(0.0, Cf_full, C_T, EPS0)

            def rowH(g):
                q = np.asarray(g); q = q / np.maximum(q.sum(1, keepdims=True), 1e-30)
                return float(np.mean(-(q * np.log(np.maximum(q, 1e-30))).sum(1)))
            H_target = rowH(g_ref)

            for av in GRID_EXT:
                # canonical fixed-eps baseline (comparable with q10/q29)
                g_can, C_can = solve(av, Cf_full, C_T, EPS0)
                acc_can = float((spot_ci[np.asarray(g_can).argmax(1)] == cidx).mean())
                # zero-solve pre-test diagnostics on the canonical cost
                Cn = np.sort(np.asarray(C_can), axis=1)
                margin = float(np.mean((Cn[:, 1] - Cn[:, 0]) / EPS0))
                H_can = rowH(g_can)

                # eps tuned so mean row entropy == H_target (kills blur drift)
                if av == 0.0:
                    eps_a = EPS0
                else:
                    lo, hi = np.log(EPS_LO), np.log(EPS_HI)
                    for _ in range(EPS_STEPS):
                        mid = 0.5 * (lo + hi)
                        gh, _ = solve(av, Cf_full, C_T, np.exp(mid))
                        if rowH(gh) > H_target:
                            hi = mid
                        else:
                            lo = mid
                    eps_a = float(np.exp(0.5 * (lo + hi)))
                g_tun, _ = solve(av, Cf_full, C_T, eps_a)
                acc_tun = float((spot_ci[np.asarray(g_tun).argmax(1)] == cidx).mean())
                H_tun = rowH(g_tun)

                # ---- A_expr: gene halves, full tree ----
                ae = []
                for (c1, c2) in Cf_halves:
                    G1, _ = solve(av, c1, C_T, eps_a)
                    G2, _ = solve(av, c2, C_T, eps_a)
                    ae.append(agreements(G1, G2, Ks, C_space_np, h))
                ae = np.array(ae).mean(0)

                # ---- A_tree: full genes, perturbed trees (per magnitude) ----
                at = {}
                for tag, pairs in tree_pairs.items():
                    vv = []
                    for (T1, T2) in pairs:
                        G1, _ = solve(av, Cf_full, T1, eps_a)
                        G2, _ = solve(av, Cf_full, T2, eps_a)
                        vv.append(agreements(G1, G2, Ks, C_space_np, h))
                    at[tag] = np.array(vv).mean(0)

                rec = dict(seed=seed, mode=mode, alpha=av, acc=acc_can, acc_tuned=acc_tun,
                           eps=eps_a, margin=margin, H_can=H_can, H_tun=H_tun,
                           feat_mag=feat_mag, sigma=sigma_cal,
                           Aexpr_raw=ae[0], Aexpr_deb=ae[1], Aexpr_hard=ae[2])
                for tag, v in at.items():
                    rec[f"Atree_raw_{tag}"] = v[0]
                    rec[f"Atree_deb_{tag}"] = v[1]
                    rec[f"Atree_hard_{tag}"] = v[2]
                rows.append(rec)
            print(f"  seed {seed} [{mode}] done  t={time.time()-t0:.0f}s", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(HERE / os.environ.get("CAND_CSV", "cand_two-sided-split-half-stability.csv"), index=False)
    return r


if __name__ == "__main__":
    run()
