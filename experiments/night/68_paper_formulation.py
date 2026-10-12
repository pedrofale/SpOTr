"""THE PAPER'S FORMULATION ON THE PLANTED CONTROL (2026-10-13), same anchor outer loss as 66_.

paper.tex, eq:cladefgw_cost: per cell n in clade tau(n),
    C_nm = (1 - alpha_tau) C^F_nm + alpha_tau [ (1 - theta_tau) L^x_nm(gamma) + theta_tau L^w_nm(gamma) ]
L^w / L^x: linearised square-loss GW cost restricted by the masks Omega^w (same clade) / Omega^x (different
clade):  L^O(T)_nm = sum_{n'm'} O_nn' (C^T_nn' - C^S_mm')^2 T_n'm',  C^T = patristic / max, C^S = spot distance / max.
With alpha or theta varying by clade this is not the gradient of any objective (method-perclade-exact), so it
is run as the paper runs it: re-linearise, re-solve -- here with exact EMD steps and Frank-Wolfe damping
T <- T + 2/(t+2) (EMD(C(T)) - T), 30 iterations.

Planted conditions, groups, spots, anchors: identical to 66_ (imported).  Per group (A undisturbed, B dispersed):
  phase 1  grid alpha_A x alpha_B, theta = 0.5 for both
  phase 2  grid theta_A x theta_B at the uniform alpha the ANCHOR picks on phase 1's diagonal
Usage: python 68_paper_formulation.py <out_dir>
"""
import os, sys, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "paper_form"
spec = importlib.util.spec_from_file_location("m66", HERE / "66_planted_control.py")
m66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m66)
ALPHAS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95]
THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
_P = {}


def prep(sigma, gdraw):
    key = (sigma, gdraw)
    if key not in _P:
        B = m66.build(sigma, gdraw)
        z = np.load(HERE.parent / "petracer" / f"{'petracerFAIR_' + m66.SEC}.npz", allow_pickle=True)
        n_all = len(np.asarray(z["cell_names"])); use = np.random.default_rng(m66.SEED).permutation(n_all)[n_all // 2:]
        C1 = np.asarray(z["C_tree"], float)[np.ix_(use, use)]; C1 = C1 / C1.max()
        C2 = B["D"] / B["D"].max()
        Ow = (B["cl"][:, None] == B["cl"][None, :]).astype(float); Ox = 1.0 - Ow
        _P[key] = dict(B=B, C1=C1, C2=C2, C2sq=C2 ** 2, Ow=Ow, Ox=Ox, OwC1=Ow * C1, OxC1=Ox * C1,
                       OwC1sq=Ow * C1 ** 2, OxC1sq=Ox * C1 ** 2, grpB=B["inB"][B["cl"]])
    return _P[key]


def lin(P, T, which):
    O, OC1, OC1sq = (P["Ow"], P["OwC1"], P["OwC1sq"]) if which == "w" else (P["Ox"], P["OxC1"], P["OxC1sq"])
    return (OC1sq @ T.sum(1))[:, None] + O @ T @ P["C2sq"].T - 2.0 * (OC1 @ T @ P["C2"].T)


def solve_paper(P, alpha_n, theta_n, iters=30):
    import ot
    B = P["B"]; n = B["n"]; p = np.ones(n) / n; q = B["q"]
    T = np.outer(p, q); M = B["M"]
    a = alpha_n[:, None]; th = theta_n[:, None]
    for t in range(iters):
        C = (1 - a) * M + a * ((1 - th) * lin(P, T, "x") + th * lin(P, T, "w"))
        S = ot.emd(p, q, C, numItermax=10 ** 7)
        T = T + 2.0 / (t + 2.0) * (S - T)
    return T


def score(P, T):
    B = P["B"]; Y = np.eye(B["K"])[B["cl"]]
    row = dict(err=float(np.median(np.linalg.norm(B["centres"][T.argmax(1)] - B["xy_true"], axis=1))))
    for cov, Js in B["anchors"].items():
        for r, J in enumerate(Js):
            row[f"nll_{int(cov*100)}_{r}"] = m66.anchor_nll(T, Y, B["Nspot"], J)
    return row


def job(args):
    phase, sigma, gdraw, xA, xB, alpha_u = args
    P = prep(sigma, gdraw); g = P["grpB"]
    if phase == 1:
        alpha_n = np.where(g, xB, xA); theta_n = np.full(len(g), 0.5)
    else:
        alpha_n = np.full(len(g), alpha_u); theta_n = np.where(g, xB, xA)
    T = solve_paper(P, alpha_n, theta_n)
    return dict(phase=phase, sigma=sigma, gdraw=gdraw, A=xA, B=xB, alpha_u=alpha_u, **score(P, T))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    conds = [(s, g) for s in m66.SIGMAS for g in m66.GROUP_DRAWS]
    ctx = get_context("fork"); procs = int(os.environ.get("PROCS", 28))
    with ctx.Pool(procs) as pool:
        r1 = pool.map(job, [(1, s, g, a, b, None) for s, g in conds for a in ALPHAS for b in ALPHAS], chunksize=2)
    d1 = pd.DataFrame(r1)
    d1["nll_20"] = d1[[c for c in d1.columns if c.startswith("nll_20_")]].mean(1)
    au = {(s, g): float(h[h.A == h.B].sort_values("nll_20").A.iloc[0]) for (s, g), h in d1.groupby(["sigma", "gdraw"])}
    with ctx.Pool(procs) as pool:
        r2 = pool.map(job, [(2, s, g, a, b, au[(s, g)]) for s, g in conds for a in THETAS for b in THETAS], chunksize=2)
    df = pd.concat([d1, pd.DataFrame(r2)], ignore_index=True)
    df.to_csv(OUT / "paper_grid.csv", index=False)
    print("uniform alpha picked by the anchor on the diagonal:", au)
