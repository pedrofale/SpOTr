"""SpOTr learner (2026-10-13): per-clade law parameters learned by gradient against the anchor loss, through the
paper's own scheme (paper.tex, 'Optimizing alpha and theta via subclone deconvolution'): gamma = Sinkhorn of the
cost linearised at the current mapping gamma0.  Here gamma0 is the EXACT FGW solution at the current parameters
(Frank-Wolfe with EMD steps, warm-started from the previous step), so the forward map is exact and the gradient --
taken through one entropic best response to the linearised cost -- does not vanish at convergence (the defect of
60_/61_, which unrolled Frank-Wolfe itself).

Cost (law form, per pair i,k; clade-aware: pairs inside clade k use L_k, all other pairs the shared L):
    r_ik^2 = L_g(i,k)^2 + c^2 n_ik / (2 pi rho);  Phi = sqrt(pi/2) r;  W = 2 / ((4 - pi) r^2);  W_ii = 0
    F(T) = (1 - a) <M, T> + a * s(theta) * sum W_ik (Phi_ik - d_jl)^2 T_ij T_kl
s(theta) re-normalises the tree term so that a sets its strength and L only its shape.
Learned: u = logit a, log L_g (per group), with a prior log L_g ~ N(log L_shared, prior_sd^2).
Outer loss: multinomial NLL of the clade counts at the FIT anchor spots under the composition of the best response.
The VALIDATION anchor spots pick the step (exact map).  No cell position is used anywhere in learning.

Modes:
  planted2   planted control, two groups (A undisturbed, B dispersed), tree weight fixed -- check against the grid
  plantedK   planted control, one L per clade (18), tree weight learned
  real       PEtracer, 3 sections x 2 seeds: FGW and law baselines on the exact weight path chosen by the anchor,
             then learned weight only, and learned weight + per-clade L
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent


def mod(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


LAWS = {"M3-4": (305.0, 1.36), "M3-1": (225.0, 1.62), "M3-2": (220.0, 1.66)}
WGRID = [0.0, 0.03, 0.056, 0.105, 0.197, 0.37, 0.693, 1.299, 2.433, 4.56, 8.544, 16.01, 30.0]


# ---------------------------------------------------------------- data
def data_real(sec, seed, cov=0.2):
    m60 = mod("m60", "60_anchor_learn.py"); P = m60.prep(sec, seed)
    J = np.random.default_rng(1000 * seed + int(cov * 100)).choice(P["m"], max(6, int(round(cov * P["m"]))), replace=False)
    L0, c0 = LAWS[sec]
    return dict(M=P["M"], D=P["D"], q=P["q"], NC=P["NC"], rho=P["rho"], cl=P["cl"], K=P["K"], Nspot=P["Nspot"],
                spots=P["xy_st"], xy_true=P["xy_true"], AGE=P["AGE"], J=J, L0=L0, c0=c0, inB=None, tag=f"{sec}_s{seed}")


def data_planted(sigma, gdraw, cov=0.2):
    m66 = mod("m66", "66_planted_control.py"); B = m66.build(sigma, gdraw)
    J = B["anchors"][cov][0]
    z = np.load(HERE.parent / "petracer" / "petracerFAIR_M3-4.npz", allow_pickle=True)
    n_all = len(np.asarray(z["cell_names"])); use = np.random.default_rng(0).permutation(n_all)[n_all // 2:]
    AGE = np.asarray(z["C_tree"], float)[np.ix_(use, use)] / 2.0
    return dict(M=B["M"], D=B["D"], q=B["q"], NC=B["NC"], rho=B["rho"], cl=B["cl"], K=B["K"], Nspot=B["Nspot"],
                spots=B["centres"], xy_true=B["xy_true"], AGE=AGE, J=J, L0=305.0, c0=1.36, inB=B["inB"],
                tag=f"planted_s{int(sigma)}_g{gdraw}")


# ---------------------------------------------------------------- exact solver with warm start
def dc_std(X):
    return (X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()).std()


def solve_exact(M, W, WPhi, D, D2, p, q, a, T0=None, iters=300, tol=1e-9):
    import ot
    T = np.outer(p, q) if T0 is None else T0.copy()
    for _ in range(iters):
        G = (1 - a) * M + a * 2.0 * (W @ T @ D2 - 2.0 * WPhi @ T @ D)
        S = ot.emd(p, q, G, numItermax=10 ** 7)
        Dl = S - T
        slope = float((G * Dl).sum())
        if slope > -tol:
            break
        quad = a * (float(((W @ Dl @ D2) * Dl).sum()) - 2.0 * float(((WPhi @ Dl @ D) * Dl).sum()))
        g = 1.0 if quad <= 0 else min(1.0, -slope / (2 * quad))
        T = T + g * Dl
    return T


def moments_np(X, Lg, c):
    """Lg: L per pair (n x n)"""
    R2 = Lg ** 2 + c * c * X["NC"] / (2 * np.pi * X["rho"])
    W = 2.0 / ((4 - np.pi) * R2); np.fill_diagonal(W, 0.0)
    return W, W * np.sqrt(np.pi / 2 * R2)


def scale(X, W, WPhi):
    n = X["M"].shape[0]; p = np.ones(n) / n; T0 = np.outer(p, X["q"])
    g0 = 2.0 * (W @ T0 @ X["D"] ** 2 - 2.0 * WPhi @ T0 @ X["D"])
    return dc_std(X["M"]) / dc_std(g0)


def anchor_nll(T, X, J):
    Y = np.eye(X["K"])[X["cl"]]; pred = (T.T @ Y)[J]; N = X["Nspot"][J]; nj = N.sum(1, keepdims=True)
    pi = (pred * (nj / np.maximum(pred.sum(1, keepdims=True), 1e-300)) + Y.mean(0)) / (nj + 1.0)
    return float(-(N * np.log(np.maximum(pi, 1e-300))).sum() / N.sum())


def err(T, X):
    return float(np.median(np.linalg.norm(X["spots"][T.argmax(1)] - X["xy_true"], axis=1)))


def agree(T, X):
    Y = np.eye(X["K"])[X["cl"]]; has = X["Nspot"].sum(1) > 0
    return float(((T.T @ Y).argmax(1)[has] == X["Nspot"].argmax(1)[has]).mean())


# ---------------------------------------------------------------- baselines on the exact weight path
def path_point(args):
    X, form, w = args
    n = X["M"].shape[0]; p = np.ones(n) / n; D = X["D"]
    if form == "fgw":
        W = np.ones((n, n)); np.fill_diagonal(W, 0.0); WPhi = W * 2.0 * X["AGE"]
    elif form in ("oracle", "oracle-rel"):                  # ceiling: each clade's L measured from the TRUE positions
        meas = measured_L(X, X["cl"], X["c0"])
        Lk = np.array([meas.get(k, np.nan) for k in range(X["K"])]); ok = np.isfinite(Lk)
        if form == "oracle-rel":                            # keep only the pattern across clades, at the level of L0
            Lk = Lk * X["L0"] / np.exp(np.log(Lk[ok]).mean())
        Lk[~ok] = X["L0"]
        same = X["cl"][:, None] == X["cl"][None, :]
        W, WPhi = moments_np(X, np.where(same, Lk[X["cl"]][:, None], X["L0"]), X["c0"])
    else:
        W, WPhi = moments_np(X, np.full((n, n), X["L0"]), X["c0"])
    sc = scale(X, W, WPhi)
    T = solve_exact(X["M"], W * sc, WPhi * sc, D, D ** 2, p, X["q"], w / (1 + w))
    Jf, Jv = X["J"][: len(X["J"]) // 2], X["J"][len(X["J"]) // 2:]
    return dict(form=form, w=w, err=err(T, X), agree=agree(T, X), nll_fit=anchor_nll(T, X, Jf), nll_val=anchor_nll(T, X, Jv),
                nll_all=anchor_nll(T, X, X["J"]))


# ---------------------------------------------------------------- learner
def learn(X, groups, learn_a=True, a0=0.165, steps=30, lr=0.08, prior_sd=0.5, eps_rel=0.05, device="cuda", log_every=1,
          pair_groups=None, L_init=None):
    import torch
    torch.set_num_threads(1)
    dev = torch.device(device if torch.cuda.is_available() else "cpu"); DT = torch.float64
    t = lambda A: torch.as_tensor(np.asarray(A, float), dtype=DT, device=dev)
    n, m = X["M"].shape; p = np.ones(n) / n; q = X["q"]; D = X["D"]; D2 = D ** 2
    Mt, Dt, D2t = t(X["M"]), t(D), t(D2); pt, qt = t(p), t(q)
    NCt = t(X["NC"]) / (2 * np.pi * X["rho"])
    G_ = int(groups.max()) + 1
    if pair_groups is None:                                     # pairs inside one clade use that cell's group
        same = (X["cl"][:, None] == X["cl"][None, :])
    else:                                                       # control: an arbitrary grouping replaces the clades
        same = (pair_groups[:, None] == pair_groups[None, :])
    pg = np.where(same, groups[:, None], G_)                    # pair -> group index (G_ = shared); symmetric
    pgt = torch.as_tensor(pg, device=dev)
    off = t(1.0 - np.eye(n))
    Y = t(np.eye(X["K"])[X["cl"]]); Nsp = t(X["Nspot"]); prior = Y.mean(0)
    Jf = X["J"][: len(X["J"]) // 2]; Jv = X["J"][len(X["J"]) // 2:]
    Jft = torch.as_tensor(Jf, device=dev)
    u = torch.tensor(float(np.log(a0 / (1 - a0))), dtype=DT, device=dev, requires_grad=learn_a)
    L_init = np.full(G_, float(X["L0"])) if L_init is None else np.asarray(L_init, float)   # start = prior centre
    logL = torch.tensor(np.log(L_init), dtype=DT, device=dev, requires_grad=True)
    logL0 = torch.tensor(np.log(L_init), dtype=DT, device=dev)                       # prior centre
    logLs = torch.tensor(float(np.log(X.get("Lx", X["L0"]))), dtype=DT, device=dev); c = float(X["c0"])   # pairs outside any group
    params = ([u] if learn_a else []) + [logL]
    opt = torch.optim.Adam(params, lr=lr)
    epsM = eps_rel * float(dc_std(X["M"]))

    def moments(logL):
        Lp = torch.exp(torch.cat([logL, logLs.reshape(1)]))[pgt]
        R2 = Lp ** 2 + c * c * NCt
        W = 2.0 / ((4 - np.pi) * R2) * off
        return W, W * torch.sqrt(np.pi / 2 * R2)

    def sinkhorn(C, iters=300, diff_iters=30):
        """log-domain Sinkhorn; converge without gradients, then differentiate the last diff_iters from the
        converged duals (approximates the implicit gradient at a fraction of the memory)."""
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
        pred = (T.T @ Y)[J]; nj = Nsp[J].sum(1, keepdim=True)
        pi = (pred * (nj / pred.sum(1, keepdim=True).clamp_min(1e-300)) + prior) / (nj + 1.0)
        return -(Nsp[J] * torch.log(pi.clamp_min(1e-300))).sum() / Nsp[J].sum()

    Tstar = None; log = []; best = None
    for it in range(steps + 1):
        a = torch.sigmoid(u)
        W, WPhi = moments(logL)
        T0 = torch.outer(pt, qt)
        g0 = 2.0 * (W @ T0 @ D2t - 2.0 * WPhi @ T0 @ Dt)
        s = float(dc_std(X["M"])) / (g0 - g0.mean(1, keepdim=True) - g0.mean(0, keepdim=True) + g0.mean()).std()
        Wn, WPn, an, sn = W.detach().cpu().numpy(), WPhi.detach().cpu().numpy(), float(a), float(s)
        Tstar = solve_exact(X["M"], Wn * sn, WPn * sn, D, D2, p, q, an, T0=Tstar)
        rec = dict(step=it, a=an, err=err(Tstar, X), agree=agree(Tstar, X), nll_fit=anchor_nll(Tstar, X, Jf),
                   nll_val=anchor_nll(Tstar, X, Jv), L=np.exp(logL.detach().cpu().numpy()).tolist())
        log.append(rec)
        if best is None or rec["nll_val"] < best["nll_val"]:
            best = dict(rec, T=Tstar.copy())
        if it == steps:
            break
        # gradient through one entropic best response to the cost linearised at the exact solution
        opt.zero_grad()
        Tt = t(Tstar)
        Glin = (1 - a) * Mt + a * s * 2.0 * (W @ Tt @ D2t - 2.0 * WPhi @ Tt @ Dt)
        S = sinkhorn(Glin)
        loss = nll(S, Jft) + ((logL - logL0) ** 2).sum() / (2 * prior_sd ** 2) / len(Jf)
        loss.backward(); opt.step()
    return log, best


def anchor_L(X, min_cells=5):
    """each clade's mixing length read off the FIT anchor spots alone: mean squared separation of pairs of that
    clade's anchor cells (pairs inside one spot at d = 0), minus the law's packing term from the dissociated cells'
    tree.  Clades the fit anchors barely see keep the shared L0.  No validation spot and no true position enters."""
    from scipy.spatial.distance import cdist
    Jf = X["J"][: len(X["J"]) // 2]; D2 = cdist(X["spots"][Jf], X["spots"][Jf]) ** 2
    out = np.full(X["K"], float(X["L0"]))
    for k in range(X["K"]):
        w = X["Nspot"][Jf, k]; idx = np.where(X["cl"] == k)[0]
        W2 = np.outer(w, w) - np.diag(w)
        if w.sum() < min_cells or len(idx) < 2 or W2.sum() <= 0:
            continue
        iu = np.triu_indices(len(idx), 1)
        pk = float((X["NC"][np.ix_(idx, idx)][iu] / (2 * np.pi * X["rho"])).mean())
        v = (W2 * D2).sum() / W2.sum() / 2.0 - X["c0"] ** 2 * pk
        out[k] = float(np.sqrt(max(v, (0.25 * X["L0"]) ** 2)))
    return out


def anchor_law(X, fit_c=False, c_fixed=1.0, L_floor=50.0):
    """the growth law calibrated on the FIT anchor spots and the dissociated cells' tree alone, nothing transferred.
    One point per pair of clades (k, k'): y = mean d^2/2 over pairs of their anchor cells (pairs inside one spot at
    d = 0), x = mean n/(2 pi rho) over dissociated pairs with one cell in each.  y = L^2 + c^2 x, fitted by
    non-negative least squares weighted by sqrt(anchor pairs); c either fitted or fixed (c = 1: a clone packed as a
    uniform disc)."""
    from scipy.optimize import nnls
    from scipy.spatial.distance import cdist
    Jf = X["J"][: len(X["J"]) // 2]; D2 = cdist(X["spots"][Jf], X["spots"][Jf]) ** 2
    Nf = X["Nspot"][Jf]; cl = X["cl"]; P = X["NC"] / (2 * np.pi * X["rho"])
    xs, ys, ws = [], [], []
    for k in range(X["K"]):
        ik = np.where(cl == k)[0]
        if Nf[:, k].sum() < 3 or len(ik) < 2:
            continue
        for k2 in range(k, X["K"]):
            i2 = np.where(cl == k2)[0]
            if Nf[:, k2].sum() < 3:
                continue
            W2 = np.outer(Nf[:, k], Nf[:, k2]) - (np.diag(Nf[:, k]) if k == k2 else 0.0)
            if W2.sum() <= 0:
                continue
            sub = P[np.ix_(ik, i2)]
            xs.append(sub[np.triu_indices(len(ik), 1)].mean() if k == k2 else sub.mean())
            ys.append((W2 * D2).sum() / W2.sum() / 2.0); ws.append(np.sqrt(W2.sum()))
    xs, ys, ws = map(np.array, (xs, ys, ws))
    if fit_c:
        coef, _ = nnls(np.c_[np.ones_like(xs), xs] * ws[:, None], ys * ws); L2, c = coef[0], np.sqrt(coef[1])
    else:
        c = c_fixed; L2 = max(np.sum(ws * (ys - c * c * xs)) / ws.sum(), 0.0)
    return float(max(np.sqrt(L2), L_floor)), float(c)


def measured_L(X, groups, c):
    out = {}
    for g in np.unique(groups):
        num = den = 0.0
        for k in np.unique(X["cl"][groups == g]):
            idx = np.where(X["cl"] == k)[0]
            if len(idx) < 30:
                continue
            iu = np.triu_indices(len(idx), 1)
            d2 = ((X["xy_true"][idx][:, None] - X["xy_true"][idx][None]) ** 2).sum(-1)[iu]
            pk = X["NC"][np.ix_(idx, idx)][iu] / (2 * np.pi * X["rho"])
            num += float(np.sum(d2 / 2 - c * c * pk)); den += len(iu[0])
        out[int(g)] = float(np.sqrt(max(num / den, 1.0))) if den else float("nan")
    return out


# ---------------------------------------------------------------- jobs
def job_planted2(args):
    sigma, gdraw, eps_rel = args
    X = data_planted(sigma, gdraw)
    groups = X["inB"][X["cl"]].astype(int)                  # 0 = A, 1 = B
    log, best = learn(X, groups, learn_a=False, a0=0.197 / 1.197, eps_rel=eps_rel)
    meas = measured_L(X, groups, X["c0"])
    return dict(sigma=sigma, gdraw=gdraw, eps_rel=eps_rel, LA=best["L"][0], LB=best["L"][1], err=best["err"],
                best_step=best["step"], measA=meas[0], measB=meas[1], log=[{k: v for k, v in r.items()} for r in log])


def job_plantedK(args):
    sigma, gdraw, learn_a = args
    X = data_planted(sigma, gdraw)
    log, best = learn(X, X["cl"].copy(), learn_a=learn_a, a0=0.197 / 1.197)
    meas = measured_L(X, X["cl"], X["c0"])
    return dict(sigma=sigma, gdraw=gdraw, learn_a=learn_a, L=best["L"], inB=X["inB"].tolist(), err=best["err"],
                agree=best["agree"], a=best["a"], best_step=best["step"], meas=meas,
                sizes=np.bincount(X["cl"], minlength=X["K"]).tolist(), log=log)


def shuffle_tree(X, seed):
    """control: permute which tree position (ages, clone sizes, clade) each cell carries, keeping its expression"""
    perm = np.random.default_rng(4000 + seed).permutation(len(X["cl"]))
    Y = dict(X); Y["AGE"] = X["AGE"][np.ix_(perm, perm)]; Y["NC"] = X["NC"][np.ix_(perm, perm)]; Y["cl"] = X["cl"][perm]
    return Y


def job_real(args):
    out = _job_real(args)
    out["cov"] = args[3] if len(args) > 3 else 0.2
    return out


def _job_real(args):
    sec, seed, arm = args[:3]
    cov = args[3] if len(args) > 3 else 0.2                 # share of spots in the anchor (half fit, half validation)
    X = data_real(sec, seed, cov)
    if arm == "path-law-shuffled":
        rows = [path_point((shuffle_tree(X, seed), "law", w)) for w in WGRID]
        for r in rows:                                      # placement is scored on the cells' TRUE positions anyway
            r["form"] = "law-shuffled"
        return dict(sec=sec, seed=seed, arm=arm, path=rows)
    if arm in ("path-law-anc1", "path-law-ancfit"):        # law calibrated on the anchor alone, nothing transferred
        X["L0"], X["c0"] = anchor_law(X, fit_c=(arm == "path-law-ancfit"))
        rows = [path_point((X, "law", w)) for w in WGRID]
        return dict(sec=sec, seed=seed, arm=arm, path=rows, L0=X["L0"], c0=X["c0"])
    if arm == "per-clade-anc1":                             # per-clade learning from the anchor-calibrated law
        X["L0"], X["c0"] = anchor_law(X, fit_c=False)
        log, best = learn(X, X["cl"].copy(), learn_a=True, a0=0.197 / 1.197)
        return dict(sec=sec, seed=seed, arm=arm, err=best["err"], agree=best["agree"], a=best["a"], L=best["L"],
                    best_step=best["step"], meas=measured_L(X, X["cl"], X["c0"]), L0=X["L0"], c0=X["c0"],
                    sizes=np.bincount(X["cl"], minlength=X["K"]).tolist(), log=log)
    if arm.startswith("path-"):
        rows = [path_point((X, arm[5:], w)) for w in WGRID]
        return dict(sec=sec, seed=seed, arm=arm, path=rows)
    if arm == "per-random":                                 # control: same learning, random groups of clade sizes
        groups = np.random.default_rng(200 + seed).permutation(X["cl"])     # random cells, clade-sized groups
        log, best = learn(X, groups, learn_a=True, a0=0.197 / 1.197, pair_groups=groups)
        return dict(sec=sec, seed=seed, arm=arm, err=best["err"], agree=best["agree"], a=best["a"], L=best["L"],
                    best_step=best["step"], meas={}, sizes=np.bincount(groups).tolist(), log=log)
    groups = X["cl"].copy() if arm.startswith("per-clade") else np.zeros(X["K"], int)[X["cl"]]
    L_init = anchor_L(X) if arm == "per-clade-mom" else None          # start and prior centre from the fit anchors
    log, best = learn(X, groups, learn_a=True, a0=0.197 / 1.197, L_init=L_init)
    meas = measured_L(X, X["cl"], X["c0"])
    out = dict(sec=sec, seed=seed, arm=arm, err=best["err"], agree=best["agree"], a=best["a"], L=best["L"],
               best_step=best["step"], meas=meas, sizes=np.bincount(X["cl"], minlength=X["K"]).tolist(), log=log,
               L_init=None if L_init is None else L_init.tolist())
    tag = "" if cov == 0.2 else f"_c{int(round(cov * 100))}"
    np.save(HERE / "spotr_out" / f"T_{sec}_s{seed}_{arm}{tag}.npy", best["T"].astype(np.float32))
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mode = sys.argv[1]; (HERE / "spotr_out").mkdir(exist_ok=True)
    ctx = get_context("spawn"); procs = int(os.environ.get("PROCS", 12))
    t0 = time.time()
    if mode == "planted2":
        jobs = [(s, g, e) for s in [0.0, 300.0, 500.0] for g in [0, 1] for e in [0.05]]
        with ctx.Pool(procs) as pool:
            res = pool.map(job_planted2, jobs)
        for r in res:
            print(f"sigma {r['sigma']:>5.0f} draw {r['gdraw']}: measured A {r['measA']:.0f} B {r['measB']:.0f} | learned A {r['LA']:.0f} B {r['LB']:.0f}"
                  f" (log ratio {np.log(r['LB'] / r['LA']):+.2f}) err {r['err']:.0f} um, best step {r['best_step']}")
    elif mode == "plantedK":
        jobs = [(s, g, la) for s in [0.0, 300.0, 500.0] for g in [0, 1] for la in [False]]
        with ctx.Pool(procs) as pool:
            res = pool.map(job_plantedK, jobs)
    elif mode == "real":
        SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
        ARMS = os.environ.get("ARMS", "path-fgw,path-law,weight,per-clade").split(",")
        COVS = [float(x) for x in os.environ.get("COVS", "0.2").split(",")]
        jobs = [(s, sd, arm, cv) for s in ["M3-4", "M3-1", "M3-2"] for sd in SEEDS for arm in ARMS for cv in COVS]
        with ctx.Pool(procs) as pool:
            res = pool.map(job_real, jobs)
        mode = os.environ.get("OUTNAME", mode)
    json.dump(res, open(HERE / "spotr_out" / f"{mode}.json", "w"), default=float)
    print(f"{mode}: {len(res)} jobs in {time.time() - t0:.0f}s")
