"""The per-group tree weight, learned (plan A1), 2026-10-13.

101_aware_retest.py showed that switching the tree term off (or damping it) for a mixing group -- migrating clades,
or a region where every cell mixes -- improves the map when the truth says which group mixes.  Here the weights are
parameters of the learner:
  clade weights   w_k > 0 multiply the tree term of the within-clade pairs of clade k (cross-clade pairs: 1)
  spot weights    v_r > 0 for each region r of the spots; a pair of spots (j, l) carries v_r(j) v_r(l)
so the tree term is  sum W_ik w_ik (Phi_ik - D_jl)^2 v_j v_l T_ij T_kl.  The product form keeps it separable: its
gradient is  2 [ (W Phi^2 (T v))_i - 2 (W Phi T (D v))_ij + (W T (D^2 v))_ij ] v_j  -- three matrix products, as
before.  log w and log v carry a N(0, sd^2) prior; the overall scale is absorbed by the tree term's normalisation,
so only relative weights (and the global tree weight alpha) matter.  With every weight 1 the learner is
70_spotr.learn (checked by `python 103_aware_learn.py check`).  The exact solve inside the learner runs its products on
the GPU and stops after 150 Frank-Wolfe iterations per step (warm-started from the previous step's map); the first
sweep, with the CPU solver and 300 iterations, stalled for an hour on growth-loss runs that raised the tree weight.
Regions: k-means on the principal components of spot expression (spot_regions); the learner never sees the niche.
"""
import sys, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


m70 = mod("m70", "70_spotr.py")


def spot_regions(X, k=4, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    A = X["Xst"] - X["Xst"].mean(0)
    Z = PCA(n_components=min(20, A.shape[1] - 1), random_state=seed).fit_transform(A)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def tree_grad(W, WP, WP2, T, D, D2, v):
    """A(T) such that the tree term is <A(T), T> and its gradient 2 A(T); numpy or torch"""
    Dv, D2v = D * v[:, None], D2 * v[:, None]
    return ((WP2 @ (T @ v))[:, None] - 2.0 * (WP @ T @ Dv) + W @ T @ D2v) * v[None, :]


def solve_exact_w(M, W, WP, D, D2, p, q, a, v=None, T0=None, iters=300, tol=1e-9):
    import ot
    v = np.ones(len(q)) if v is None else v
    WP2 = WP * WP / np.where(W > 0, W, 1.0)
    T = np.outer(p, q) if T0 is None else T0.copy()
    for _ in range(iters):
        G = (1 - a) * M + a * 2.0 * tree_grad(W, WP, WP2, T, D, D2, v)
        S = ot.emd(p, q, G, numItermax=10 ** 7)
        Dl = S - T
        slope = float((G * Dl).sum())
        if slope > -tol:
            break
        quad = a * float((tree_grad(W, WP, WP2, Dl, D, D2, v) * Dl).sum())
        g = 1.0 if quad <= 0 else min(1.0, -slope / (2 * quad))
        T = T + g * Dl
    return T


def solve_exact_w_gpu(M, W, WP, D, D2, p, q, a, v=None, T0=None, iters=150, tol=1e-9, device="cuda"):
    """solve_exact_w with the matrix products on the GPU (the EMD stays on the CPU); W, WP as numpy arrays"""
    import torch, ot
    dev = torch.device(device if torch.cuda.is_available() else "cpu"); tt = lambda A: torch.as_tensor(A, dtype=torch.float64, device=dev)
    v = np.ones(len(q)) if v is None else v
    Wt, WPt = tt(W), tt(WP); WP2t = tt(WP * WP / np.where(W > 0, W, 1.0)); Dt, D2t, vt = tt(D), tt(D2), tt(v)
    A = lambda T: tree_grad(Wt, WPt, WP2t, tt(T), Dt, D2t, vt).cpu().numpy()
    T = np.outer(p, q) if T0 is None else T0.copy()
    for _ in range(iters):
        G = (1 - a) * M + a * 2.0 * A(T)
        S = ot.emd(p, q, G, numItermax=10 ** 7)
        Dl = S - T
        slope = float((G * Dl).sum())
        if slope > -tol:
            break
        quad = a * float((A(Dl) * Dl).sum())
        g = 1.0 if quad <= 0 else min(1.0, -slope / (2 * quad))
        T = T + g * Dl
    return T


def learn_two_stage(X, clade_w=False, regions=None, outer="anchor", prior_L=1.0, prior_w=0.5, steps=30, **kw):
    """stage 1: the single-L learner (tree weight and one L); stage 2: from its chosen map, the group weights alone,
    tree weight and L held.  Stage 2's step 0 is stage 1's map, so the validation half can always keep it.  Returns
    (log1, best1, log2, best2)."""
    log1, best1 = learn_w(X, outer=outer, prior_L=prior_L, steps=steps, **kw)
    log2, best2 = learn_w(X, clade_w=clade_w, regions=regions, outer=outer, prior_L=prior_L, prior_w=prior_w, steps=steps,
                          a0=best1["a"], L_init=best1["L"][0], freeze_aL=True, **kw)
    return log1, best1, log2, best2


def tv(T, X, idx=None):
    """composition error per spot (total variation between predicted and true clone proportions), mean over idx"""
    Y = np.eye(X["K"])[X["cl"]]; P = T.T @ Y; P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    Q = X["Nspot"] / X["Nspot"].sum(1, keepdims=True); t = 0.5 * np.abs(P - Q).sum(1)
    return float(t.mean() if idx is None else t[idx].mean())


def learn_w(X, L_groups=None, clade_w=False, regions=None, outer="anchor", a0=0.165, steps=30, lr=0.08,
            prior_L=1.0, prior_w=1.0, eps_rel=0.05, device="cuda", fix_w=None, fix_v=None, L_init=None, freeze_aL=False):
    """70_spotr.learn with per-clade tree weights (clade_w) and per-region spot weights (regions: one label per spot).
    L_groups: per-cell group of the within-clade L (default one shared L).  fix_w / fix_v: hold the weights at given
    values (for checks and oracles).  L_init: start of the within-clade L (one value or one per group; default L0);
    freeze_aL: hold the tree weight (at a0) and L (at L_init) and learn only the group weights -- the second stage of
    learn_two_stage.  Returns (log, best)."""
    import torch
    torch.set_num_threads(1)
    dev = torch.device(device if torch.cuda.is_available() else "cpu"); DT = torch.float64
    t = lambda A: torch.as_tensor(np.asarray(A, float), dtype=DT, device=dev)
    n, m = X["M"].shape; p = np.ones(n) / n; q = X["q"]; D = X["D"]; D2 = D ** 2; cl = X["cl"]; K = X["K"]
    Mt, Dt, D2t, pt, qt = t(X["M"]), t(D), t(D2), t(p), t(q)
    NCt = t(X["NC"]) / (2 * np.pi * X["rho"])
    groups = np.zeros(n, int) if L_groups is None else np.asarray(L_groups)
    G_ = int(groups.max()) + 1
    same_np = cl[:, None] == cl[None, :]
    pg = torch.as_tensor(np.where(same_np, groups[:, None], G_), device=dev)
    same = torch.as_tensor(same_np, device=dev); clt = torch.as_tensor(cl, device=dev)
    off = t(1.0 - np.eye(n)); Y = t(np.eye(K)[cl]); Nsp = t(X["Nspot"]); prior = Y.mean(0)
    R = 0 if regions is None else int(np.max(regions)) + 1
    regt = None if regions is None else torch.as_tensor(np.asarray(regions), device=dev)
    if outer == "growth":
        Jf, Jv = X["growth_fit"], X["growth_val"]; fit_t, edge_t = t(X["fitness"]), t(X["edge_spot"])
        gsign = float(X.get("growth_sign", -1.0)); fit_loss = lambda T, J: m70.growth_np(T, X, J)
    else:
        Jf = X["J"][: len(X["J"]) // 2]; Jv = X["J"][len(X["J"]) // 2:]; fit_loss = lambda T, J: m70.anchor_nll(T, X, J)
    Jft = torch.as_tensor(Jf, device=dev)
    u = torch.tensor(float(np.log(a0 / (1 - a0))), dtype=DT, device=dev, requires_grad=not freeze_aL)
    Lst = np.full(G_, float(X["L0"])) if L_init is None else np.broadcast_to(np.asarray(L_init, float), (G_,)).copy()
    logL = torch.tensor(np.log(Lst), dtype=DT, device=dev, requires_grad=not freeze_aL)
    logL0 = torch.full((G_,), float(np.log(X["L0"])), dtype=DT, device=dev)
    logLs = torch.tensor(float(np.log(X.get("Lx", X["L0"]))), dtype=DT, device=dev); c = float(X["c0"])   # as 70_spotr.learn
    lw = torch.zeros(K, dtype=DT, device=dev, requires_grad=clade_w and fix_w is None)
    if fix_w is not None:
        lw = torch.as_tensor(np.log(np.asarray(fix_w, float)), dtype=DT, device=dev)
    lv = torch.zeros(max(R, 1), dtype=DT, device=dev, requires_grad=R > 0 and fix_v is None)
    if fix_v is not None:
        lv = torch.as_tensor(np.log(np.asarray(fix_v, float)), dtype=DT, device=dev)
    params = ([] if freeze_aL else [u, logL]) + ([lw] if lw.requires_grad else []) + ([lv] if lv.requires_grad else [])
    opt = torch.optim.Adam(params, lr=lr)
    epsM = eps_rel * float(m70.dc_std(X["M"]))

    def weights():
        Lp = torch.exp(torch.cat([logL, logLs.reshape(1)]))[pg]
        R2 = Lp ** 2 + c * c * NCt
        W = 2.0 / ((4 - np.pi) * R2) * off
        Wc = torch.where(same, torch.exp(lw)[clt][:, None].expand(n, n), torch.ones_like(W))
        W = W * Wc; WP = W * torch.sqrt(np.pi / 2 * R2); WP2 = WP * torch.sqrt(np.pi / 2 * R2)
        v = torch.exp(lv)[regt] if R > 0 else torch.ones(m, dtype=DT, device=dev)
        return W, WP, WP2, v

    def sinkhorn(C, iters=300, diff_iters=30):
        lp, lq = pt.log(), qt.log()
        f = torch.zeros(n, dtype=DT, device=dev); g = torch.zeros(m, dtype=DT, device=dev)
        with torch.no_grad():
            Cd = C.detach()
            for _ in range(iters):
                f = epsM * lp - epsM * torch.logsumexp((g[None, :] - Cd) / epsM, dim=1)
                g = epsM * lq - epsM * torch.logsumexp((f[:, None] - Cd) / epsM, dim=0)
        for _ in range(diff_iters):
            f = epsM * lp - epsM * torch.logsumexp((g[None, :] - C) / epsM, dim=1)
            g = epsM * lq - epsM * torch.logsumexp((f[:, None] - C) / epsM, dim=0)
        return torch.exp((f[:, None] + g[None, :] - C) / epsM)

    def nll(T, J):
        if outer == "growth":
            e = (T @ edge_t) / T.sum(1).clamp_min(1e-300)
            fv, ev = fit_t[J] - fit_t[J].mean(), e[J] - e[J].mean()
            return -gsign * (fv * ev).sum() / torch.sqrt((fv * fv).sum() * (ev * ev).sum())
        pred = (T.T @ Y)[J]; nj = Nsp[J].sum(1, keepdim=True)
        pi = (pred * (nj / pred.sum(1, keepdim=True).clamp_min(1e-300)) + prior) / (nj + 1.0)
        return -(Nsp[J] * torch.log(pi.clamp_min(1e-300))).sum() / Nsp[J].sum()

    Tstar, log, best = None, [], None
    for it in range(steps + 1):
        a = torch.sigmoid(u)
        W, WP, WP2, v = weights()
        T0 = torch.outer(pt, qt)
        g0 = 2.0 * tree_grad(W, WP, WP2, T0, Dt, D2t, v)
        s = float(m70.dc_std(X["M"])) / (g0 - g0.mean(1, keepdim=True) - g0.mean(0, keepdim=True) + g0.mean()).std()
        Wn, WPn, vn, an, sn = W.detach().cpu().numpy(), WP.detach().cpu().numpy(), v.detach().cpu().numpy(), float(a), float(s)
        Tstar = solve_exact_w_gpu(X["M"], Wn * sn, WPn * sn, D, D2, p, q, an, v=vn, T0=Tstar, device=device)
        rec = dict(step=it, a=an, err=m70.err(Tstar, X), nll_fit=fit_loss(Tstar, Jf), nll_val=fit_loss(Tstar, Jv),
                   L=np.exp(logL.detach().cpu().numpy()).tolist(), w=np.exp(lw.detach().cpu().numpy()).tolist(),
                   v=np.exp(lv.detach().cpu().numpy()).tolist() if R > 0 else [])
        log.append(rec)
        if best is None or rec["nll_val"] < best["nll_val"]:
            best = dict(rec, T=Tstar.copy())
        if it == steps:
            break
        opt.zero_grad()
        Tt = t(Tstar)
        Glin = (1 - a) * Mt + a * s * 2.0 * tree_grad(W, WP, WP2, Tt, Dt, D2t, v)
        S = sinkhorn(Glin)
        loss = nll(S, Jft) + ((logL - logL0) ** 2).sum() / (2 * prior_L ** 2) / len(Jf)
        if lw.requires_grad:
            loss = loss + (lw ** 2).sum() / (2 * prior_w ** 2) / len(Jf)
        if lv.requires_grad:
            loss = loss + (lv ** 2).sum() / (2 * prior_w ** 2) / len(Jf)
        loss.backward(); opt.step()
    return log, best


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "check":
    import json, time
    m84 = mod("m84", "84_growth_sim.py")
    Dj = json.load(open(HERE / "spotr_out" / "growth_sweep.json"))
    X = m84.problem(m84.simulate(1, m_hi=0.02, niche_mig=30.0, niche_expr=1.0, pos_w=0.3)); X["L0"], X["c0"] = Dj["L0"], Dj["c0"]
    n = len(X["cl"]); p = np.ones(n) / n
    W, WP = m70.moments_np(X, np.full((n, n), X["L0"]), X["c0"]); sc = m70.scale(X, W, WP)
    T1 = m70.solve_exact(X["M"], W * sc, WP * sc, X["D"], X["D"] ** 2, p, X["q"], 0.2)
    T2 = solve_exact_w(X["M"], W * sc, WP * sc, X["D"], X["D"] ** 2, p, X["q"], 0.2)
    print("solver, all weights 1: max |T - T_70| =", np.abs(T1 - T2).max())
    t0 = time.time(); la, ba = m70.learn(X, np.zeros(n, int), learn_a=True, a0=0.165, steps=5, prior_sd=1.0)
    lb, bb = learn_w(X, steps=5, prior_L=1.0)
    print("learner, all weights 1: alpha", [round(r["a"], 5) for r in la], "vs", [round(r["a"], 5) for r in lb])
    print("                        L    ", [round(r["L"][0], 3) for r in la], "vs", [round(r["L"][0], 3) for r in lb])
    reg = spot_regions(X, 4); z = X["spot_in_niche"]
    print("regions: share of each cluster inside the niche", [round(float(z[reg == r].mean()), 2) for r in range(4)],
          "sizes", np.bincount(reg).tolist(), f"{time.time() - t0:.0f}s")
