"""What the outer loss is, whether it helps, and whether per-clade parameters are learned -- on the clone-map
split (PEtracer M3-4, seed 0; anchor = clade counts of 20% of spots, half used to fit, half to validate).

 A. the anchor: observed clade composition of the anchor spots vs the composition the mapping predicts there
 B. along the tree weight (exact solver, law-based tree term): anchor loss vs true placement error; and the
    learning trajectory of the weight-only arm (anchor loss and placement error per step)
 C. per-clade mixing lengths L_k over learning steps (anchor), and learned vs measured L_k
    (anchor, supervised on 30% of cells, and size-matched random groups)
Usage: python 64_outer_loss.py <out_dir>
"""
import os, sys, json, importlib.util
os.environ["FP32"] = "1"
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outer_loss"
SEC, SEED, COV = "M3-4", 0, 0.2
L0, C0 = 305.0, 1.36                       # law transferred to M3-4 from M3-1 and M3-2


def load():
    spec = importlib.util.spec_from_file_location("m60", HERE / "60_anchor_learn.py")
    m60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m60)
    P = m60.prep(SEC, SEED)
    J = np.random.default_rng(1000 * SEED + int(COV * 100)).choice(P["m"], max(6, int(round(COV * P["m"]))), replace=False)
    return m60, P, J


def law_moments(P, logL_pair, c):
    R2 = np.exp(2 * logL_pair) + c * c * P["NC"] / (2 * np.pi * P["rho"])
    W = 2.0 / ((4 - np.pi) * R2); np.fill_diagonal(W, 0.0)
    return W, W * np.sqrt(np.pi / 2 * R2)


def anchor_nll_np(T, Y, Nspot, J):
    pred = (T.T @ Y)[J]; nj = Nspot[J].sum(1, keepdims=True); prior = Y.mean(0)
    pi = (pred * (nj / np.maximum(pred.sum(1, keepdims=True), 1e-300)) + prior) / (nj + 1.0)
    return float(-(Nspot[J] * np.log(np.maximum(pi, 1e-300))).sum() / Nspot[J].sum()), pi


def path_job(w):
    os.environ["OMP_NUM_THREADS"] = "1"
    m60, P, J = load(); m55 = m60.m55
    n = P["M"].shape[0]; p = np.ones(n) / n
    W, WPhi = law_moments(P, np.full((n, n), np.log(L0)), C0)
    T0 = np.outer(p, P["q"]); g0 = 2.0 * (W @ T0 @ P["D"] ** 2 - 2.0 * WPhi @ T0 @ P["D"])
    sc = m55.dc_std(P["M"]) / m55.dc_std(g0)
    T = m55.solve(P["M"], W * sc, WPhi * sc, P["D"], P["D"] ** 2, p, P["q"], w / (1 + w))
    Y = np.eye(P["K"])[P["cl"]]
    loss, _ = anchor_nll_np(T, Y, P["Nspot"], J)
    err = float(np.median(np.linalg.norm(P["xy_st"][T.argmax(1)] - P["xy_true"], axis=1)))
    return dict(w=w, anchor_loss=loss, err=err)


def learn_job(arm):
    import torch
    m60, P, J = load(); DEV, DT = m60.DEV, m60.DT
    groups = np.random.default_rng(100 + SEED).permutation(P["cl"]) if arm == "random groups" else P["cl"]
    prob = m60.Problem(P, groups)
    Jfit = torch.as_tensor(J[: len(J) // 2], device=DEV); Jval = torch.as_tensor(J[len(J) // 2:], device=DEV)
    tr = np.random.default_rng(500 + SEED).random(prob.n) < 0.3; tr_t = torch.as_tensor(tr, device=DEV)
    Ys = torch.as_tensor(P["xy_st"], dtype=DT, device=DEV); Xt = torch.as_tensor(P["xy_true"], dtype=DT, device=DEV)
    D2tr = ((Xt[tr_t][:, None, :] - Ys[None]) ** 2).sum(-1); D2tr = D2tr / D2tr.mean()
    sup = arm == "per-clade, supervised"
    per = arm != "weight only"
    u = torch.tensor(np.log(0.2 / 1.2 / (1 - 0.2 / 1.2)), dtype=DT, device=DEV, requires_grad=True)
    dk = torch.zeros(prob.G, dtype=DT, device=DEV, requires_grad=True)
    logLs = torch.tensor(np.log(L0), dtype=DT, device=DEV); logc = torch.tensor(np.log(C0), dtype=DT, device=DEV)
    opt = torch.optim.Adam([{"params": [u], "lr": 0.05}] + ([{"params": [dk], "lr": 0.03}] if per else []))
    prior_sd, steps = (1.0, 80) if sup else (0.3, 60)
    T = f = g = None; log = []

    def solve(a, W, WPhi, T0, f0, g0, n_outer, n_diff=3):
        prob.s = 1.0 / m60.dc_std(prob.grad_s(torch.outer(prob.p, prob.q), W, WPhi))
        T_ = torch.outer(prob.p, prob.q) if T0 is None else T0
        f_ = torch.zeros(prob.n, dtype=DT, device=DEV) if f0 is None else f0
        g_ = torch.zeros(prob.m, dtype=DT, device=DEV) if g0 is None else g0
        inner = 60 if T0 is None else 15

        def step(T_, f_, g_, a, W, WPhi):
            G = (1 - a) * prob.M + a * prob.s * prob.grad_s(T_, W, WPhi)
            f_, g_, S = prob.sinkhorn(G, f_, g_, inner)
            Dl = S - T_; slope = (G * Dl).sum(); qd = a * prob.s * prob.quad(Dl, W, WPhi)
            gam = torch.where(qd > 0, (-slope / (2 * qd)).clamp(0.0, 1.0), torch.ones_like(qd))
            return T_ + gam * Dl, f_, g_
        with torch.no_grad():
            for _ in range(n_outer - n_diff):
                T_, f_, g_ = step(T_, f_, g_, a.detach(), W.detach(), WPhi.detach())
        for _ in range(n_diff):
            T_, f_, g_ = step(T_, f_.detach(), g_.detach(), a, W, WPhi)
        return T_, f_.detach(), g_.detach()
    for it in range(steps):
        opt.zero_grad()
        logL = torch.cat([logLs + dk, logLs.reshape(1)])
        W, WPhi = prob.moments(logL, logc)
        T, f, g = solve(torch.sigmoid(u), W, WPhi, None if T is None else T.detach(), f, g, 40 if T is None else 10)
        fit = prob.anchor_nll(T, Jfit)
        loss = (((T[tr_t] / prob.p[tr_t][:, None]) * D2tr).sum(1).mean() if sup else fit)
        if per:
            loss = loss + (dk ** 2).sum() / (2 * prior_sd ** 2) / len(Jfit)
        loss.backward(); opt.step()
        with torch.no_grad():
            am = T.argmax(1).cpu().numpy()
            e = np.linalg.norm(P["xy_st"][am] - P["xy_true"], axis=1)
            log.append(dict(arm=arm, step=it, a=float(torch.sigmoid(u)), anchor_fit=float(fit),
                            anchor_val=float(prob.anchor_nll(T, Jval)), err=float(np.median(e)),
                            err_heldout=float(np.median(e[~tr])), L=(np.exp(np.log(L0) + dk.detach().cpu().numpy())).tolist()))
    return log, groups.tolist()


def measured_L(P, groups):
    out = []
    for k in range(int(max(groups)) + 1):
        idx = np.where(groups == k)[0]
        if len(idx) < 30:
            out.append(np.nan); continue
        iu = np.triu_indices(len(idx), 1)
        d2 = ((P["xy_true"][idx][:, None] - P["xy_true"][idx][None]) ** 2).sum(-1)[iu]
        pk = P["NC"][np.ix_(idx, idx)][iu] / (2 * np.pi * P["rho"])
        out.append(float(np.sqrt(max(np.mean(d2 / 2 - C0 * C0 * pk), 1.0))))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = get_context("spawn")
    spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
    m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
    with ctx.Pool(14) as pool:
        path = pool.map(path_job, [float(w) for w in m55.WGRID])
    with ctx.Pool(4) as pool:
        runs = pool.map(learn_job, ["weight only", "per-clade, anchor", "random groups", "per-clade, supervised"])
    pd.DataFrame(path).to_csv(OUT / "path.csv", index=False)
    traj = pd.DataFrame([r for log, _ in runs for r in log])
    traj.to_json(OUT / "trajectories.json", orient="records")
    m60, P, J = load()
    meas = {arm: measured_L(P, np.array(gr)) for (log, gr), arm in zip(runs, ["weight only", "per-clade, anchor", "random groups", "per-clade, supervised"])}
    json.dump(dict(J=J.tolist(), measured=meas, sizes=np.bincount(P["cl"]).tolist()), open(OUT / "meta.json", "w"))
    print(pd.DataFrame(path).round(3).to_string(index=False))
    print(traj.groupby("arm").tail(1)[["arm", "a", "anchor_fit", "anchor_val", "err", "err_heldout"]].round(3).to_string(index=False))
