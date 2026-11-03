"""Does one global tree penalty distort the map when one clade expands and diffuses while the others stay localized?
Growth simulator, 2026-10-13.  Pedro's hypothesis: "If Clade A undergoes rapid expansion and high migration (becoming
highly diffuse in space), while Clade B remains highly localized, a single global tree-distance penalty matrix will
distort the mapping."  Fixed before running.

Tumours (84_growth_sim.simulate, fast=1): at founding the most exposed frontier lineage divides ADV times faster
(ADV 2: about half the tumour; 3: about three quarters), and after growth only it makes on average FAST_MIG runs
per cell (0, 10, 30, 100); every other lineage static.  Expression with the edge program, without (pos_w 0) or with
weak positional programs (0.3).  4 tumours per setting.  Law L0, c0 from the reference tumour (85_growth_sweep.json).

Maps, exact, each at its best tree weight among 0.1, 0.165, 0.3, 0.45, 0.65:
  fgw         plain FGW on the tree distance: the literal single global penalty
  single L    the law with one within-clade L for every clade: the best of 0.5-3 x L0 and the pooled measured value
  two-level   clade A at its true L, every other clade at their pooled true L
  true        each clade at its true L (measured on the true positions with the law's slope)
  expression  expression only
Readouts: placement error (median um) over all cells, over A's cells and over the localized clades' cells; the clone
map (mean Pearson r of clone proportions across spots) over all major clades and over the localized ones.
Usage: python 92_expanding_clade.py [procs]     (writes spotr_out/expanding_clade.json)
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np

HERE = Path(__file__).resolve().parent
ADV, FMIG, POS, SEEDS = [2.0, 3.0], [0.0, 10.0, 30.0, 100.0], [0.0, 0.3], [1, 2, 3, 4]
ALPHAS, SCAN = [0.1, 0.165, 0.3, 0.45, 0.65], [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def readouts(T, X, A):
    d = np.linalg.norm(X["spots"][T.argmax(1)] - X["xy_true"], axis=1); inA = np.isin(X["cl"], A)
    loc = np.isin(X["cl"], np.setdiff1d(np.arange(X["Kmaj"]), A))
    Y = np.eye(X["K"])[X["cl"]]; P = T.T @ Y; P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    Q = X["Nspot"] / X["Nspot"].sum(1, keepdims=True)
    r = {k: np.corrcoef(P[:, k], Q[:, k])[0, 1] for k in range(X["Kmaj"]) if P[:, k].std() > 0 and Q[:, k].std() > 0}
    return dict(err=float(np.median(d)), err_A=float(np.median(d[inA])), err_loc=float(np.median(d[loc])),
                r=float(np.mean(list(r.values()))), r_loc=float(np.mean([v for k, v in r.items() if k not in A])))


def job(args):
    adv, fmig, pos_w, seed, L0, c0 = args
    import warnings; warnings.filterwarnings("ignore")
    import ot
    m70 = mod("m70", "70_spotr.py"); m84 = mod("m84", "84_growth_sim.py")
    X = m84.problem(m84.simulate(seed, m_hi=0.02, fast=1, adv=adv, fast_mig=fmig, pos_w=pos_w)); X["L0"], X["c0"] = L0, c0
    n, cl, Km, A = len(X["cl"]), X["cl"], X["Kmaj"], X["fast_clades"]
    meas = m70.measured_L(X, cl, c0); Lt = np.array([meas.get(k, np.nan) for k in range(Km)])
    others = np.setdiff1d(np.arange(Km), A)
    pooled = m70.measured_L(X, np.zeros(n, int), c0)[0]
    pooled_loc = m70.measured_L(X, np.where(np.isin(cl, others), 0, 1), c0)[0]
    variants = {f"one L x{s:g}": np.full(Km + 1, s * L0) for s in SCAN}
    variants["one L pooled"] = np.full(Km + 1, pooled)
    two = np.full(Km + 1, pooled_loc); two[A] = Lt[A]; variants["two-level"] = two
    variants["true"] = np.r_[np.where(np.isfinite(Lt), Lt, L0), L0]
    p = np.ones(n) / n; D = X["D"]; same = cl[:, None] == cl[None, :]
    W0, WP0 = m70.moments_np(X, np.full((n, n), L0), c0); sc = m70.scale(X, W0, WP0)
    Te = ot.emd(p, X["q"], X["M"], numItermax=10 ** 7)
    out = dict(adv=adv, fast_mig=fmig, pos_w=pos_w, seed=seed, Kmaj=Km, A=A.tolist(), share_A=float(np.isin(cl, A).mean()),
               L_true=Lt.tolist(), expression=readouts(Te, X, A), maps={})
    Wf = np.ones((n, n)); np.fill_diagonal(Wf, 0.0); WPf = Wf * 2.0 * X["AGE"]; scf = m70.scale(X, Wf, WPf)
    T = None; out["maps"]["fgw"] = []
    for a in ALPHAS:
        T = m70.solve_exact(X["M"], Wf * scf, WPf * scf, D, D ** 2, p, X["q"], a, T0=T)
        out["maps"]["fgw"].append(readouts(T, X, A))
    for name, v in variants.items():
        Wv, WPv = m70.moments_np(X, np.where(same, v[cl][:, None], L0), c0); T = None; out["maps"][name] = []
        for a in ALPHAS:
            T = m70.solve_exact(X["M"], Wv * sc, WPv * sc, D, D ** 2, p, X["q"], a, T0=T)
            out["maps"][name].append(readouts(T, X, A))
    return out


def best(r, names, key):
    """best over tree weights (and over the names given), by placement error or by clone map"""
    vals = [m[key] for nm in names for m in r["maps"][nm]]
    return min(vals) if key.startswith("err") else max(vals)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    D = json.load(open(HERE / "spotr_out" / "growth_sweep.json")); L0, c0 = D["L0"], D["c0"]
    t0 = time.time()
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(job, [(ad, fm, pw, sd, L0, c0) for ad in ADV for pw in POS for fm in FMIG for sd in SEEDS])
    json.dump(dict(L0=L0, c0=c0, alphas=ALPHAS, res=res), open(HERE / "spotr_out" / "expanding_clade.json", "w"),
              indent=1, default=float)
    single = [v for v in res[0]["maps"] if v.startswith("one L")]
    for key in ["err", "err_loc", "err_A", "r", "r_loc"]:
        print(f"== {key}")
        for ad in ADV:
            for pw in POS:
                for fm in FMIG:
                    rr = [r for r in res if r["adv"] == ad and r["pos_w"] == pw and r["fast_mig"] == fm]
                    f = lambda names: np.mean([best(r, names, key) for r in rr])
                    d = np.array([best(r, ["true"], key) - best(r, single, key) for r in rr])
                    print(f"  adv {ad:g} pos {pw:g} runs {fm:5.0f} (A {np.mean([r['share_A'] for r in rr]):.0%}): expression "
                          f"{np.mean([r['expression'][key] for r in rr]):.3f}  fgw {f(['fgw']):.3f}  single L {f(single):.3f}  "
                          f"two-level {f(['two-level']):.3f}  true {f(['true']):.3f}  (true - single {d.mean():+.3f})")
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
