"""Does a NICHE-dependent mixing length improve the map when mixing is set by place rather than by lineage?
Growth simulator, 2026-10-13.  Fixed before running.

Pedro's hypothesis: some part of the tumour (e.g. infiltrated by T cells) makes every cell in it more motile,
whatever its clade.  Tumours (84_growth_sim.simulate, niche_mig): growth with every lineage static, then every cell
inside a disc holding 30% of the cells (centred half-way out) makes on average NICHE_MIG runs confined to the disc
(0.001 ~ none, 10, 30; true within-clade L inside about 36, 108, 118 um against 69 um outside).  Expression with the
edge program, without or with weak positional programs.  4 tumours per setting.  Law L0, c0 as in 85_.

The niche model: a within-clade pair's L depends on where its two spots are -- L_in if both are in the disc, L_out
if both are outside, sqrt((L_in^2 + L_out^2) / 2) if one each; cross-clade pairs keep L0.  The tree term then splits
into three spot blocks, each separable (solve_blocks); the disc is taken as known, as a region read off the spots
(e.g. a T-cell signature) would be.
Maps, exact, each at its best tree weight among 0.1, 0.165, 0.3, 0.45, 0.65:
  single L   one within-clade L (best of 0.5-3 x L0 and the pooled measured value)
  niche L    L_in, L_out at their true values (within-clade pairs measured on the true positions)
  clade L    each clade's true L: what the per-clade model can do with niche-driven mixing
  expression expression only
Readouts: the clone map (mean r of clone proportions across spots) over all spots, over the niche's spots and over
the rest; placement error of the cells inside and outside the niche.
Usage: python 95_niche_oracle.py [procs]     (writes spotr_out/niche_oracle.json)
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np

HERE = Path(__file__).resolve().parent
NMIG, POS, SEEDS = [0.001, 10.0, 30.0], [0.0, 0.3], [1, 2, 3, 4]
ALPHAS, SCAN = [0.1, 0.165, 0.3, 0.45, 0.65], [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def solve_blocks(M, blocks, p, q, a, T0=None, iters=300, tol=1e-9):
    """Frank-Wolfe on (1-a)<M,T> + a sum_b sum W^b_ik (Phi^b_ik - D_jl)^2 T_ij T_kl over the spot pairs (j,l) of block
    b; blocks = [(W, W*Phi, W*Phi^2, mask, D*mask, D^2*mask)], masks symmetric and summing to one.  With every block
    equal it reduces to 70_spotr.solve_exact."""
    import ot
    T = np.outer(p, q) if T0 is None else T0.copy()
    A = lambda T: sum(WP2 @ T @ Mk - 2.0 * (WP @ T @ DMk) + W @ T @ D2Mk for W, WP, WP2, Mk, DMk, D2Mk in blocks)
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


def niche_L(X, c0):
    """true within-clade L inside and outside the niche, from the true positions (major clades)"""
    cl, Km, inR = X["cl"], X["Kmaj"], X["cell_in_niche"]
    P = X["NC"] / (2 * np.pi * X["rho"]); iu = np.triu_indices(len(cl), 1)
    d2 = (((X["xy_true"][:, None] - X["xy_true"][None]) ** 2).sum(-1) / 2.0)[iu]
    same = (cl[iu[0]] == cl[iu[1]]) & (cl[iu[0]] < Km)
    L = lambda m: float(np.sqrt(max(np.mean(d2[m] - c0 ** 2 * P[iu][m]), 1.0)))
    return L(same & inR[iu[0]] & inR[iu[1]]), L(same & ~inR[iu[0]] & ~inR[iu[1]])


def readouts(T, X):
    inR, zR = X["cell_in_niche"], X["spot_in_niche"]
    d = np.linalg.norm(X["spots"][T.argmax(1)] - X["xy_true"], axis=1)
    Y = np.eye(X["K"])[X["cl"]]; P = T.T @ Y; P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    Q = X["Nspot"] / X["Nspot"].sum(1, keepdims=True)
    def cmap(rows):
        rs = [np.corrcoef(P[rows, k], Q[rows, k])[0, 1] for k in range(X["Kmaj"]) if P[rows, k].std() > 0 and Q[rows, k].std() > 0]
        return float(np.mean(rs)) if rs else float("nan")
    return dict(err=float(np.median(d)), err_in=float(np.median(d[inR])), err_out=float(np.median(d[~inR])),
                r=cmap(np.arange(len(zR))), r_in=cmap(np.where(zR)[0]), r_out=cmap(np.where(~zR)[0]))


def job(args):
    nmig, pos_w, seed, L0, c0 = args
    import warnings; warnings.filterwarnings("ignore")
    import ot
    m70 = mod("m70", "70_spotr.py"); m84 = mod("m84", "84_growth_sim.py")
    X = m84.problem(m84.simulate(seed, m_hi=0.02, niche_mig=nmig, pos_w=pos_w)); X["L0"], X["c0"] = L0, c0
    n, cl, Km = len(X["cl"]), X["cl"], X["Kmaj"]
    Lin, Lout = niche_L(X, c0)
    meas = m70.measured_L(X, cl, c0); Lt = np.array([meas.get(k, np.nan) for k in range(Km)])
    pooled = m70.measured_L(X, np.zeros(n, int), c0)[0]
    p = np.ones(n) / n; D = X["D"]; same = cl[:, None] == cl[None, :]
    W0, WP0 = m70.moments_np(X, np.full((n, n), L0), c0); sc = m70.scale(X, W0, WP0)
    out = dict(niche_mig=nmig, pos_w=pos_w, seed=seed, Kmaj=Km, L_in=Lin, L_out=Lout, L_clade=Lt.tolist(),
               share_in=float(X["cell_in_niche"].mean()), maps={})
    Te = ot.emd(p, X["q"], X["M"], numItermax=10 ** 7); out["expression"] = readouts(Te, X)

    def path(W, WP, blocks=None):
        T, recs = None, []
        for a in ALPHAS:
            T = (m70.solve_exact(X["M"], W, WP, D, D ** 2, p, X["q"], a, T0=T) if blocks is None
                 else solve_blocks(X["M"], blocks, p, X["q"], a, T0=T))
            recs.append(readouts(T, X))
        return recs
    for s in SCAN + ["pooled"]:
        Ls = pooled if s == "pooled" else s * L0
        W, WP = m70.moments_np(X, np.where(same, Ls, L0), c0); out["maps"][f"one L {s}"] = path(W * sc, WP * sc)
    W, WP = m70.moments_np(X, np.where(same, np.r_[np.where(np.isfinite(Lt), Lt, L0), L0][cl][:, None], L0), c0)
    out["maps"]["clade L"] = path(W * sc, WP * sc)
    z = X["spot_in_niche"].astype(float); masks = [np.outer(z, z), np.outer(1 - z, 1 - z), np.outer(z, 1 - z) + np.outer(1 - z, z)]
    blocks = []
    for Lb, Mk in zip([Lin, Lout, np.sqrt((Lin ** 2 + Lout ** 2) / 2)], masks):
        W, WP = m70.moments_np(X, np.where(same, Lb, L0), c0); W, WP = W * sc, WP * sc
        blocks.append((W, WP, WP * WP / np.where(W > 0, W, 1.0), Mk, D * Mk, D ** 2 * Mk))
    out["maps"]["niche L"] = path(None, None, blocks)
    return out


def best(r, names, key):
    vals = [m[key] for nm in names for m in r["maps"][nm]]
    return min(vals) if key.startswith("err") else max(vals)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    D = json.load(open(HERE / "spotr_out" / "growth_sweep.json")); L0, c0 = D["L0"], D["c0"]
    if os.environ.get("CHECK"):                              # the block solver reduces to solve_exact
        m70 = mod("m70", "70_spotr.py"); m84 = mod("m84", "84_growth_sim.py")
        X = m84.problem(m84.simulate(1, m_hi=0.02, niche_mig=10.0)); X["c0"] = c0; n = len(X["cl"]); p = np.ones(n) / n
        W, WP = m70.moments_np(X, np.full((n, n), L0), c0); sc = m70.scale(X, W, WP); W, WP = W * sc, WP * sc
        z = X["spot_in_niche"].astype(float); D = X["D"]
        masks = [np.outer(z, z), np.outer(1 - z, 1 - z), np.outer(z, 1 - z) + np.outer(1 - z, z)]
        T1 = m70.solve_exact(X["M"], W, WP, D, D ** 2, p, X["q"], 0.3)
        T2 = solve_blocks(X["M"], [(W, WP, WP * WP / np.where(W > 0, W, 1.0), Mk, D * Mk, D ** 2 * Mk) for Mk in masks], p, X["q"], 0.3)
        print("max |T_exact - T_blocks| =", np.abs(T1 - T2).max(), " (mass per cell", 1 / n, ")"); sys.exit()
    t0 = time.time()
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(job, [(nm, pw, sd, L0, c0) for pw in POS for nm in NMIG for sd in SEEDS])
    json.dump(dict(L0=L0, c0=c0, alphas=ALPHAS, res=res), open(HERE / "spotr_out" / "niche_oracle.json", "w"), indent=1, default=float)
    single = [v for v in res[0]["maps"] if v.startswith("one L")]
    for key in ["r", "r_in", "r_out", "err_in", "err_out"]:
        print(f"== {key}")
        for pw in POS:
            for nm in NMIG:
                rr = [r for r in res if r["pos_w"] == pw and r["niche_mig"] == nm]
                d = np.array([best(r, ["niche L"], key) - best(r, single, key) for r in rr])
                win = (d > 0).sum() if not key.startswith("err") else (d < 0).sum()
                print(f"  pos {pw:g} niche runs {nm:6g} (L in {np.mean([r['L_in'] for r in rr]):.0f}, out {np.mean([r['L_out'] for r in rr]):.0f}): "
                      f"expression {np.mean([r['expression'][key] for r in rr]):.3f}  single L {np.mean([best(r, single, key) for r in rr]):.3f}  "
                      f"clade L {np.mean([best(r, ['clade L'], key) for r in rr]):.3f}  niche L {np.mean([best(r, ['niche L'], key) for r in rr]):.3f}"
                      f"  (niche - single {d.mean():+.3f}, better in {win}/{len(d)})")
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
