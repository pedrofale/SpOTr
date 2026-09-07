"""FIXING CROSS-SECTION TRANSFER: move the DEPENDENCE, not the joint.

Section-to-section transfer of the raw fitted transforms is a wash (03_transfer.py), and the
diagnosis is that the sections differ in composition -- 49% / 28% / 30% of pairs sit at the
saturation ceiling, so the same tumour's sections sample very different parts of the phylogeny.
The correspondence analysis is computed on the JOINT histogram, which mixes the dependence
structure with those marginals.

But both marginals are OBSERVABLE on the target without any positions of the cells being mapped:
the tree-distance marginal comes from the query cells' own tree, the spatial one from the target's
spot coordinates.  So transfer the source's dependence and rake it onto the target's marginals
(iterative proportional fitting), then decompose:

    P_src  --IPF-->  P~  with row sums a_target, column sums b_target  -->  phi, psi

which is the same statement as "the copula transfers, the marginals do not".

Three arms per ordered pair, all with the weight swept over one common grid on the target:
    FGW                 no transfer at all
    plain transfer      the source's phi, psi applied directly           (03_transfer.py)
    raked transfer      the source's dependence on the target's marginals
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import GenFGW, fit_kernel, standardise, apply_lookup, tree_bins, space_bins

SECS = ["M3-4", "M3-1", "M3-2"]
CENTRE = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
MULT = np.array([0.0, 0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0, 13.0])
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def ipf(P, a, b, n_iter=200, tol=1e-12):
    """Rake P to row sums a and column sums b, preserving its odds ratios (the copula)."""
    Q = P.copy() + 1e-12
    for _ in range(n_iter):
        r = Q.sum(1); Q *= np.where(r > 0, a / np.maximum(r, 1e-300), 0)[:, None]
        c = Q.sum(0); Q *= np.where(c > 0, b / np.maximum(c, 1e-300), 0)[None, :]
        if np.abs(Q.sum(1) - a).max() < tol:
            break
    return Q / Q.sum()


def decompose(P, te, se, ncomp):
    a = P.sum(1); b = P.sum(0)
    kr = a > 0; kc = b > 0
    Q = P[np.ix_(kr, kc)] / np.sqrt(np.outer(a[kr], b[kc]))
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    tc = 0.5 * (te[:-1] + te[1:]); sc = 0.5 * (se[:-1] + se[1:])
    rho, phi, psi = [], [], []
    for r in range(1, min(ncomp + 1, len(S))):
        f = np.zeros(len(a)); g = np.zeros(len(b))
        f[kr] = U[:, r] / np.sqrt(a[kr]); g[kc] = Vt[r] / np.sqrt(b[kc])
        if np.polyfit(tc[kr], f[kr], 1)[0] < 0:
            f, g = -f, -g
        rho.append(float(S[r])); phi.append(f); psi.append(g)
    return dict(rho=np.array(rho), phi=np.array(phi), psi=np.array(psi), tc=tc, sc=sc)


def _init(target, arm, source, R):
    T_ = load(target)
    n = len(T_["xy_sc"])
    if arm == "fgw":
        Phi = [standardise(T_["CT"])]; Psi = [standardise(T_["CS"])]; rho = np.array([1.0])
    else:
        S_ = load(source)
        # ONE common bin grid, defined on the union so both sections index the same cells
        tt = np.r_[S_["CT"][np.triu_indices(len(S_["xy_sc"]), 1)], T_["CT"][np.triu_indices(n, 1)]]
        ss = np.r_[pdist(S_["xy_sc"]), T_["CS"][np.triu_indices(len(T_["CS"]), 1)]]
        te = tree_bins(tt); se = space_bins(ss)
        Psrc, _, _ = np.histogram2d(S_["CT"][np.triu_indices(len(S_["xy_sc"]), 1)],
                                    pdist(S_["xy_sc"]), bins=[te, se])
        Psrc = Psrc / Psrc.sum()
        if arm == "raked":
            # target marginals: its query cells' tree distances, and its own spot distances --
            # both available with no knowledge of where the query cells belong
            a, _ = np.histogram(T_["CT"][np.triu_indices(n, 1)], bins=te); a = a / a.sum()
            b, _ = np.histogram(T_["CS"][np.triu_indices(len(T_["CS"]), 1)], bins=se)
            b = b / b.sum()
            Psrc = ipf(Psrc, a, b)
        K = decompose(Psrc, te, se, R)
        Phi = [standardise(apply_lookup(T_["CT"], K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(T_["CS"], K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=T_["M"], Phi=Phi, Psi=Psi, rho=rho, xy_sc=T_["xy_sc"], xy_st=T_["xy_st"],
                   n=n, m=T_["M"].shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    if w == 0:
        T = ot.emd(p, q, S["M"], numItermax=1_000_000)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, q, 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    rows = []
    jobs = [(b, "fgw", None, 1, "FGW (no transfer)") for b in SECS]
    for b in SECS:
        for a in SECS:
            if a == b:
                continue
            jobs.append((b, "plain", a, 2, f"plain transfer from {a}"))
            jobs.append((b, "raked", a, 2, f"raked transfer from {a}"))
    for target, arm, source, R, name in jobs:
        with Pool(7, initializer=_init, initargs=(target, arm, source, R)) as pool:
            for w, e in pool.imap_unordered(_solve, list(CENTRE[target] * MULT)):
                rows.append(dict(target=target, arm=name, kind=arm, source=str(source), w=w, err=e))
        g = pd.DataFrame(rows).query("target == @target and arm == @name")
        print(f"target {target}   {name:<26} best {g.err.min():7.1f}   "
              f"median over w>0 {g[g.w>0].err.median():7.1f}", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "copula_transfer.csv", index=False)
    print("done")
