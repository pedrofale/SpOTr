"""ANCHOR-ONLY, STEP 2 (2026-10-12): learn the tree weight and per-clade mixing lengths by GRADIENT through
an entropic FGW mapping, against the anchor loss alone.

Mapping (torch, GPU, float64), structural term = the growth law's moment-matched likelihood (55_):
    r_ik^2 = L_g(i,k)^2 + c^2 n_ik / (2 pi rho)       g(i,k) = clade k if i, k share clade k, else "shared"
    Phi_ik = sqrt(pi/2) r_ik,  W_ik = 2 / ((4 - pi) r_ik^2),  W_ii = 0
    G(T)   = (1 - a) M~ + a s grad S(T),  grad S = 2 (W T D^2 - 2 (W o Phi) T D)
    T <- Sinkhorn_eps(G(T))  (entropic GW by mirror descent, log domain -- no clipped kernel, D1 fixed),
         cell marginal uniform, spot marginal observed nuclei share; balanced (PEtracer construction)
Learned: a = sigmoid(u), log L_g (one per clade + shared), log c.  Gradients through the last N_DIFF outer
iterations (earlier iterates detached).  Prior: log L_k ~ N(log L_shared, 0.5^2).
Loss: multinomial log-likelihood of observed clade counts at the FIT anchor spots (59_'s xent);
the VAL anchor spots choose the stopping step.  No cell position is ever used in learning.
Evaluation: placement error of the entropic map, and of the EXACT solver (55_) at the learned parameters.

Modes:  --check   entropic vs exact FGW on one split (sanity of the solver)
        (default) learning arms on 6 splits x anchor coverages
"""
import os, sys, time, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
from pathlib import Path
import numpy as np, pandas as pd, torch, treedata as td
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
DEV = torch.device("cuda")
DT = torch.float32 if os.environ.get("FP32") else torch.float64
torch.backends.cuda.matmul.allow_tf32 = False
EPS, N_OUTER, N_INNER, N_DIFF = float(os.environ.get('EPS', 0.03)), 40, 60, 3
WARM = int(os.environ.get('WARM', 10))
LAWS = {"M3-4": (305.0, 1.36), "M3-1": (225.0, 1.62), "M3-2": (220.0, 1.66)}   # fitted on the other sections
_OBS = {}


def obs_clade():
    if "c" not in _OBS:
        _OBS["c"] = td.read_h5td("/home/pedro/.treedata/datasets/koblan25_tumor.h5td").obs["clade"].astype(str)
    return _OBS["c"]


def prep(sec, seed):
    o = obs_clade()
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    S = m55.load(sec)
    n = len(S["AGE"]); use = np.random.default_rng(seed).permutation(n)[n // 2:]
    cells = np.asarray(z["cell_names"])[use]
    scells = np.asarray(z["spot_cell_names"]); sof = np.asarray(z["spot_of_cell"]); m = S["M"].shape[1]
    names = sorted(set(o.loc[np.asarray(z["cell_names"])]) | set(o.loc[scells]))
    ci = {c: i for i, c in enumerate(names)}; K = len(names)
    cl = np.array([ci[o.loc[c]] for c in cells])
    Nspot = np.zeros((m, K))
    for c, j in zip(scells, sof):
        Nspot[j, ci[o.loc[c]]] += 1
    AGE = S["AGE"][np.ix_(use, use)]
    return dict(sec=sec, seed=seed, M=S["M"][use], D=S["D"], q=S["q"], AGE=AGE, NC=m55.clone_size(AGE).astype(float),
                rho=len(use) / S["area"], cl=cl, K=K, Nspot=Nspot, xy_st=S["xy_st"], xy_true=S["xy_sc"][use], m=m)


def dc_std(X):
    return (X - X.mean(1, keepdim=True) - X.mean(0, keepdim=True) + X.mean()).std()


class Problem:
    def __init__(self, P, groups):
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=DT, device=DEV)
        self.P = P; self.n, self.m = P["M"].shape
        self.Mraw = t(P["M"]); self.mscale = dc_std(self.Mraw); self.M = self.Mraw / self.mscale
        self.D = t(P["D"]); self.D2 = self.D ** 2
        self.NCt = t(P["NC"]) / (2 * np.pi * P["rho"])
        self.p = torch.full((self.n,), 1.0 / self.n, dtype=DT, device=DEV); self.q = t(P["q"])
        self.logp, self.logq = self.p.log(), self.q.log()
        same = groups[:, None] == groups[None, :]
        self.G = int(groups.max()) + 1                                      # index G = shared
        self.pairgrp = torch.as_tensor(np.where(same, groups[:, None], self.G), device=DEV)
        self.offdiag = 1.0 - torch.eye(self.n, dtype=DT, device=DEV)
        self.Y = torch.nn.functional.one_hot(torch.as_tensor(P["cl"], device=DEV), P["K"]).to(DT)
        self.Nspot = t(P["Nspot"]); self.prior = self.Y.mean(0)
        self.s = None

    def moments(self, logL, logc):
        R2 = torch.exp(2 * logL)[self.pairgrp] + torch.exp(2 * logc) * self.NCt
        W = 2.0 / ((4 - np.pi) * R2) * self.offdiag
        return W, W * torch.sqrt(np.pi / 2 * R2)

    def grad_s(self, T, W, WPhi):
        return 2.0 * (W @ T @ self.D2 - 2.0 * WPhi @ T @ self.D)

    def sinkhorn(self, C, f, g, iters):
        for _ in range(iters):
            f = EPS * self.logp - EPS * torch.logsumexp((g[None, :] - C) / EPS, dim=1)
            g = EPS * self.logq - EPS * torch.logsumexp((f[:, None] - C) / EPS, dim=0)
        return f, g, torch.exp((f[:, None] + g[None, :] - C) / EPS)

    def solve(self, a, W, WPhi, n_diff=N_DIFF):
        if self.s is None:                                                  # scale fixed at the first call
            with torch.no_grad():
                self.s = 1.0 / dc_std(self.grad_s(torch.outer(self.p, self.q), W, WPhi))
        T = torch.outer(self.p, self.q)
        f = torch.zeros(self.n, dtype=DT, device=DEV); g = torch.zeros(self.m, dtype=DT, device=DEV)
        with torch.no_grad():
            Wd, WPd, ad = W.detach(), WPhi.detach(), a.detach()
            for _ in range(N_OUTER - n_diff):
                f, g, T = self.sinkhorn((1 - ad) * self.M + ad * self.s * self.grad_s(T, Wd, WPd), f, g, N_INNER)
        for _ in range(n_diff):
            f, g, T = self.sinkhorn((1 - a) * self.M + a * self.s * self.grad_s(T, W, WPhi), f.detach(), g.detach(), N_INNER)
        return T

    def quad(self, Dl, W, WPhi):
        """quadratic coefficient of the structural objective along a direction Dl (55_'s line search)."""
        return ((W @ Dl @ self.D2) * Dl).sum() - 2.0 * ((WPhi @ Dl @ self.D) * Dl).sum()

    def solve_fw(self, a, W, WPhi, n_diff=N_DIFF, T0=None, n_outer=N_OUTER):
        """Frank-Wolfe with an ENTROPIC linear oracle and exact line search: the exact solver's algorithm with
        Sinkhorn in place of EMD, so it tracks the exact solution as EPS -> 0 and stays differentiable."""
        if self.s is None:
            with torch.no_grad():
                self.s = 1.0 / dc_std(self.grad_s(torch.outer(self.p, self.q), W, WPhi))
        T = torch.outer(self.p, self.q) if T0 is None else T0.detach()
        f = torch.zeros(self.n, dtype=DT, device=DEV); g = torch.zeros(self.m, dtype=DT, device=DEV)

        def step(T, f, g, a, W, WPhi):
            G = (1 - a) * self.M + a * self.s * self.grad_s(T, W, WPhi)
            f, g, S = self.sinkhorn(G, f, g, N_INNER)
            Dl = S - T
            slope = (G * Dl).sum(); qd = a * self.s * self.quad(Dl, W, WPhi)
            gam = torch.where(qd > 0, (-slope / (2 * qd)).clamp(0.0, 1.0), torch.ones_like(qd))
            return T + gam * Dl, f, g
        with torch.no_grad():
            Wd, WPd, ad = W.detach(), WPhi.detach(), a.detach()
            for _ in range(n_outer - n_diff):
                T, f, g = step(T, f, g, ad, Wd, WPd)
        for _ in range(n_diff):
            T, f, g = step(T, f.detach(), g.detach(), a, W, WPhi)
        return T

    def anchor_nll(self, T, J):
        pred = (T.T @ self.Y)[J]
        nj = self.Nspot[J].sum(1, keepdim=True)
        pi = (pred * (nj / pred.sum(1, keepdim=True).clamp_min(1e-300)) + self.prior) / (nj + 1.0)
        return -(self.Nspot[J] * torch.log(pi.clamp_min(1e-300))).sum() / self.Nspot[J].sum()

    def err(self, T):
        am = T.argmax(1).cpu().numpy()
        return float(np.median(np.linalg.norm(self.P["xy_st"][am] - self.P["xy_true"], axis=1)))


def exact_err(prob, a, W, WPhi):
    """55_'s exact Frank-Wolfe at the given parameters (same scale convention)."""
    P = prob.P; p = np.ones(prob.n) / prob.n
    sc = float(prob.mscale * prob.s)                    # (1-a) M + a * mscale * s * gradS  ==  55_'s form
    T = m55.solve(P["M"], W.detach().cpu().numpy() * sc, WPhi.detach().cpu().numpy() * sc,
                  P["D"], P["D"] ** 2, p, P["q"], float(a))
    am = T.argmax(1)
    return float(np.median(np.linalg.norm(P["xy_st"][am] - P["xy_true"], axis=1)))


def learn(prob, Jfit, Jval, law, per_group, steps=60, lr=0.05, prior_sd=0.5):
    L0, c0 = law
    u = torch.tensor(np.log(0.2 / 1.2 / (1 - 0.2 / 1.2)), dtype=DT, device=DEV, requires_grad=True)
    logL = torch.full((prob.G + 1,), np.log(L0), dtype=DT, device=DEV, requires_grad=True)
    logc = torch.tensor(np.log(c0), dtype=DT, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([u, logL, logc], lr=lr)
    best, hist = (np.inf, None), []
    Tprev = None
    for it in range(steps):
        opt.zero_grad()
        L_use = logL if per_group else logL[-1].expand(prob.G + 1)
        W, WPhi = prob.moments(L_use, logc)
        T = prob.solve_fw(torch.sigmoid(u), W, WPhi, T0=Tprev, n_outer=N_OUTER if Tprev is None else WARM)
        Tprev = T.detach()
        loss = prob.anchor_nll(T, Jfit)
        if per_group:
            loss = loss + ((logL[:-1] - logL[-1]) ** 2).sum() / (2 * prior_sd ** 2) / len(Jfit)
        loss.backward(); opt.step()
        with torch.no_grad():
            v = float(prob.anchor_nll(T, Jval)); hist.append((it, float(loss), v))
            if v < best[0]:
                best = (v, dict(u=u.detach().clone(), logL=(logL if per_group else logL[-1].expand(prob.G + 1)).detach().clone(),
                                logc=logc.detach().clone(), it=it))
    return best[1], hist


def check():
    P = prep("M3-1", 0); prob = Problem(P, P["cl"])
    d = pd.read_csv(HERE / "decomposition_check.csv")
    ref = d[(d.sec == "M3-1") & (d.seed == 0) & (d.half == 0) & (d.arm.isin(["FGW", "expression only"]))]
    W = torch.ones((prob.n, prob.n), dtype=DT, device=DEV) * prob.offdiag
    WPhi = W * 2.0 * torch.as_tensor(P["AGE"], dtype=DT, device=DEV)
    for w in [0.0, 0.056, 0.197, 0.693, 2.433]:
        t0 = time.time(); a = torch.tensor(w / (1 + w), dtype=DT, device=DEV)
        T = prob.solve(a, W, WPhi); T2 = prob.solve_fw(a, W, WPhi)
        r = ref[np.isclose(ref.w, w, atol=2e-3)].err
        print(f"w={w:<6} entropic MD {prob.err(T):6.1f}  FW+Sinkhorn {prob.err(T2):6.1f} um   exact {r.iloc[0] if len(r) else float('nan'):6.1f} um   "
              f"row-mass err {float((T.sum(1) - prob.p).abs().max()):.1e}  [{time.time()-t0:.1f}s]", flush=True)


def main():
    rows, lrows = [], []
    for sec in m55.SECS:
        for seed in (0, 1):
            P = prep(sec, seed)
            for cov in (0.2, 0.05):
                rng = np.random.default_rng(1000 * seed + int(cov * 100))
                J = rng.choice(P["m"], max(4, int(round(cov * P["m"]))), replace=False)
                Jfit, Jval = J[: max(2, int(0.75 * len(J)))], J[max(2, int(0.75 * len(J))):]
                for arm, groups, per in [("global", P["cl"], False), ("per-clade", P["cl"], True),
                                         ("per-random-group", np.random.default_rng(seed).permutation(P["cl"]), True)]:
                    t0 = time.time()
                    prob = Problem(P, groups)
                    best, hist = learn(prob, torch.as_tensor(Jfit, device=DEV), torch.as_tensor(Jval, device=DEV),
                                       LAWS[sec], per)
                    with torch.no_grad():
                        W, WPhi = prob.moments(best["logL"], best["logc"]); a = torch.sigmoid(best["u"])
                        T = prob.solve_fw(a, W, WPhi)
                        e_ent = prob.err(T)
                    e_ex = exact_err(prob, a, W, WPhi)
                    rows.append(dict(sec=sec, seed=seed, cov=cov, arm=arm, a=float(a), c=float(best["logc"].exp()),
                                     L_shared=float(best["logL"][-1].exp()), stop_it=best["it"],
                                     err_entropic=e_ent, err_exact=e_ex, secs=time.time() - t0))
                    if per and arm == "per-clade":
                        # measured per-clade L from TRUE positions (evaluation only)
                        Lk = best["logL"][:-1].exp().cpu().numpy(); c = float(best["logc"].exp())
                        for k in range(P["K"]):
                            idx = np.where(P["cl"] == k)[0]
                            if len(idx) < 30:
                                continue
                            iu = np.triu_indices(len(idx), 1)
                            d2 = ((P["xy_true"][idx][:, None] - P["xy_true"][idx][None]) ** 2).sum(-1)[iu]
                            pk = P["NC"][np.ix_(idx, idx)][iu] / (2 * np.pi * P["rho"])
                            Lmeas = np.sqrt(max(np.mean(d2 / 2 - c * c * pk), 0.0))
                            lrows.append(dict(sec=sec, seed=seed, cov=cov, clade=k, n=len(idx), L_learned=Lk[k], L_measured=Lmeas))
                    print(rows[-1], flush=True)
                    pd.DataFrame(rows).to_csv(HERE / "anchor_learn.csv", index=False)
                    pd.DataFrame(lrows).to_csv(HERE / "anchor_learn_clades.csv", index=False)


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
