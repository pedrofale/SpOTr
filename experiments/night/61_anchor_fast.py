"""FAST ITERATION RUNNER for anchor-only learning (2026-10-12).  Same model and loss as 60_anchor_learn.py;
float32, couplings AND Sinkhorn duals warm-started across learning steps, several configurations in parallel
on the GPU, exact-solver evaluation deferred to a CPU pool.

Per-clade variants (60_ showed per-clade learning unstable: high lr on 18 extra parameters, weak prior,
validation on 3-5 spots):
  global          a, L_shared, c
  pc-v1           60_'s setting: all parameters from the law, lr 0.05, prior sd 0.5
  pc-2stage       learn global first, then per-clade deviations: clade lr 0.02, prior sd 0.3
  rand-2stage     the same on size-matched RANDOM groups (the null)
Anchor spots split 50/50 into fit and validation; the validation loss picks the step.
"""
import os, sys, time, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "anchor_fast"
COVS = [float(c) for c in os.environ.get("COVS", "0.2,0.05").split(",")]
ARMS = os.environ.get("ARMS", "global,pc-v1,pc-2stage,rand-2stage").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]


def worker(cfg):
    os.environ["FP32"] = "1"
    import torch
    spec = importlib.util.spec_from_file_location("m60", HERE / "60_anchor_learn.py")
    m60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m60)
    DEV, DT = m60.DEV, m60.DT
    INNER = int(os.environ.get("INNER", 15))
    DYN = os.environ.get("DYN", "1") == "1"

    class Fast(m60.Problem):
        def solve_warm(self, a, W, WPhi, T0, f0, g0, n_outer, n_diff=3):
            if self.s is None or DYN:
                s_new = 1.0 / m60.dc_std(self.grad_s(torch.outer(self.p, self.q), W, WPhi))
                self.s = s_new if DYN else s_new.detach()
            T = torch.outer(self.p, self.q) if T0 is None else T0
            f = torch.zeros(self.n, dtype=DT, device=DEV) if f0 is None else f0
            g = torch.zeros(self.m, dtype=DT, device=DEV) if g0 is None else g0
            inner = 60 if T0 is None else INNER

            def step(T, f, g, a, W, WPhi):
                G = (1 - a) * self.M + a * self.s * self.grad_s(T, W, WPhi)
                f, g, S = self.sinkhorn(G, f, g, inner)
                Dl = S - T
                slope = (G * Dl).sum(); qd = a * self.s * self.quad(Dl, W, WPhi)
                gam = torch.where(qd > 0, (-slope / (2 * qd)).clamp(0.0, 1.0), torch.ones_like(qd))
                return T + gam * Dl, f, g
            with torch.no_grad():
                for _ in range(n_outer - n_diff):
                    T, f, g = step(T, f, g, a.detach(), W.detach(), WPhi.detach())
            for _ in range(n_diff):
                T, f, g = step(T, f.detach(), g.detach(), a, W, WPhi)
            return T, f.detach(), g.detach()

    sec, seed, cov, arm = cfg["sec"], cfg["seed"], cfg["cov"], cfg["arm"]
    P = m60.prep(sec, seed)
    groups = np.random.default_rng(100 + seed).permutation(P["cl"]) if arm.startswith("rand") else P["cl"]
    prob = Fast(P, groups)
    rng = np.random.default_rng(1000 * seed + int(cov * 100))
    J = rng.choice(P["m"], max(6, int(round(cov * P["m"]))), replace=False)
    Jfit = torch.as_tensor(J[: len(J) // 2], device=DEV); Jval = torch.as_tensor(J[len(J) // 2:], device=DEV)
    tr = np.random.default_rng(500 + seed).random(prob.n) < 0.3
    tr_t = torch.as_tensor(tr, device=DEV)
    Ys = torch.as_tensor(P["xy_st"], dtype=DT, device=DEV); Xt = torch.as_tensor(P["xy_true"], dtype=DT, device=DEV)
    D2tr = ((Xt[tr_t][:, None, :] - Ys[None, :, :]) ** 2).sum(-1); D2tr = D2tr / D2tr.mean()

    def sup_loss(T):
        return ((T[tr_t] / prob.p[tr_t][:, None]) * D2tr).sum(1).mean()
    L0, c0 = m60.LAWS[sec]
    t = lambda x: torch.tensor(x, dtype=DT, device=DEV, requires_grad=True)
    u, logLs, logc = t(np.log(0.2 / 1.2 / (1 - 0.2 / 1.2))), t(np.log(L0)), t(np.log(c0))
    dk = torch.zeros(prob.G, dtype=DT, device=DEV, requires_grad=True)     # per-group deviations of log L
    state = dict(T=None, f=None, g=None)

    def run(params, groups_on, steps, lr, prior_sd, lr_dk, last=False, sup=False):
        opt = torch.optim.Adam([{"params": params, "lr": lr}] + ([{"params": [dk], "lr": lr_dk}] if groups_on else []))
        best = (np.inf, None)
        for it in range(steps):
            opt.zero_grad()
            logL = torch.cat([logLs + (dk if groups_on else dk.detach() * 0), logLs.reshape(1)])
            W, WPhi = prob.moments(logL, logc)
            T, state["f"], state["g"] = prob.solve_warm(torch.sigmoid(u), W, WPhi, state["T"], state["f"], state["g"],
                                                        40 if state["T"] is None else 10)
            state["T"] = T.detach()
            loss = sup_loss(T) if sup else prob.anchor_nll(T, Jfit)
            if groups_on:
                loss = loss + (dk ** 2).sum() / (2 * prior_sd ** 2) / len(Jfit)
            loss.backward(); opt.step()
            with torch.no_grad():
                v = float(prob.anchor_nll(T, Jval))
            if v < best[0] or last:
                best = (v, dict(train=float(loss.detach()), u=float(u), logLs=float(logLs), logc=float(logc), dk=dk.detach().cpu().numpy().copy(), it=it))
        return best

    t0 = time.time()
    if arm in ("a-only", "a+shape", "pc-law", "rand-law"):
        params = {"a-only": [u], "a+shape": [u, logLs, logc], "pc-law": [u], "rand-law": [u]}[arm]
        best = run(params, arm in ("pc-law", "rand-law"), 40, 0.05, 0.3, 0.03, last=True)
    elif arm == "pc-sup2":                                   # two-stage: supervised global, then deviations
        b1 = run([u], False, 80, 0.05, None, 0, last=True, sup=True)
        with torch.no_grad():
            u.fill_(b1[1]["u"])
        best = run([u], True, 60, 0.01, 1.0, 0.01, last=True, sup=True)
        best[1]["train_stage1"] = b1[1]["train"]
    elif arm in ("a-sup", "pc-sup", "pc-free"):
        best = run([u], arm != "a-sup", 80, 0.05, 1.0, 0.03, last=True, sup=arm.endswith("sup"))
    elif arm == "global":
        best = run([u, logLs, logc], False, 60, 0.05, None, 0)
    elif arm == "pc-v1":
        best = run([u, logLs, logc], True, 60, 0.05, 0.5, 0.05)
    else:                                                    # two-stage: global, then per-group deviations
        b1 = run([u, logLs, logc], False, 40, 0.05, None, 0)
        with torch.no_grad():
            u.fill_(b1[1]["u"]); logLs.fill_(b1[1]["logLs"]); logc.fill_(b1[1]["logc"])
        best = run([u, logLs, logc], True, 40, 0.01, 0.3, 0.02)
    b = best[1]
    with torch.no_grad():
        logL = torch.cat([torch.tensor(b["logLs"], dtype=DT, device=DEV) + torch.as_tensor(b["dk"], dtype=DT, device=DEV),
                          torch.tensor([b["logLs"]], dtype=DT, device=DEV)])
        W, WPhi = prob.moments(logL, torch.tensor(b["logc"], dtype=DT, device=DEV))
        a = torch.sigmoid(torch.tensor(b["u"], dtype=DT, device=DEV))
        T, _, _ = prob.solve_warm(a, W, WPhi, None, None, None, 40)
        e_ent = prob.err(T)
        am = T.argmax(1).cpu().numpy()
        e_ent_ho = float(np.median(np.linalg.norm(P["xy_st"][am][~tr] - P["xy_true"][~tr], axis=1)))
        sc_final = float(prob.mscale * prob.s)
    tag = f"{sec}_s{seed}_c{int(cov*100)}_{arm}"
    np.savez(OUT / f"{tag}.npz", W=W.cpu().numpy().astype(np.float64), WPhi=WPhi.cpu().numpy().astype(np.float64),
             a=float(a), sc=sc_final, Lk=np.exp(b["logLs"] + b["dk"]), c=np.exp(b["logc"]),
             groups=groups, tr=tr)
    return dict(sec=sec, seed=seed, cov=cov, arm=arm, a=float(a), L_shared=float(np.exp(b["logLs"])),
                c=float(np.exp(b["logc"])), stop_it=b["it"], train=b.get("train"), train_stage1=b.get("train_stage1"), err_entropic=e_ent, err_entropic_heldout=e_ent_ho, secs=time.time() - t0, tag=tag)


def exact_eval(tag):
    os.environ["OMP_NUM_THREADS"] = "1"
    spec = importlib.util.spec_from_file_location("m60", HERE / "60_anchor_learn.py")
    m60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m60)
    z = np.load(OUT / f"{tag}.npz"); sec, s, _, _ = tag.split("_", 3); P = m60.prep(sec, int(s[1:]))
    p = np.ones(P["M"].shape[0]) / P["M"].shape[0]
    T = m60.m55.solve(P["M"], z["W"] * z["sc"], z["WPhi"] * z["sc"], P["D"], P["D"] ** 2, p, P["q"], float(z["a"]))
    am = T.argmax(1); e = np.linalg.norm(P["xy_st"][am] - P["xy_true"], axis=1); tr = z["tr"]
    return tag, (float(np.median(e)), float(np.median(e[~tr])))


if __name__ == "__main__":
    import multiprocessing as mp
    OUT.mkdir(exist_ok=True)
    cfgs = [dict(sec=s, seed=sd, cov=c, arm=a) for s in ["M3-4", "M3-1", "M3-2"] for sd in SEEDS for c in COVS for a in ARMS]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(int(os.environ.get("GPU_PROCS", 6))) as pool:
        rows = pool.map(worker, cfgs)
    print(f"learning: {len(cfgs)} runs in {time.time()-t0:.0f}s", flush=True)
    with ctx.Pool(int(os.environ.get("CPU_PROCS", 24))) as pool:
        ex = dict(pool.map(exact_eval, [r["tag"] for r in rows]))
    df = pd.DataFrame(rows); df["err_exact"] = df.tag.map(lambda t: ex[t][0]); df["err_exact_heldout"] = df.tag.map(lambda t: ex[t][1])
    df.to_csv(HERE / os.environ.get("OUTCSV", "anchor_fast.csv"), index=False)
    print(f"total {time.time()-t0:.0f}s\n")
    print(df.groupby(["cov", "arm"])[["err_entropic", "err_exact", "a", "stop_it"]].mean().round(2).to_string())
