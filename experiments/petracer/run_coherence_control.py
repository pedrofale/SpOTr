"""alpha_tau is not fitted -- it is whatever it takes to make clade tau cohere.

The target is a MODELLING ASSUMPTION, not a measurement: "most of a clone's cells should sit
together". Concretely, 90% of a clade's mapped cells should lie within `target` x the tissue
radius of that clade's centre. No ground truth anywhere.

Then run a feedback loop:
    solve  ->  measure each clade's q90  ->  raise alpha for clades that are too spread out,
    lower it for clades that are tighter than needed  ->  re-solve

One constraint per clade, one alpha per clade, so the system is determined -- unlike a single
global coherence number, which pins down only one parameter and left the three resolution
weights underdetermined (+55 to +83 um worse than optimal).

The converged alpha_tau ARE the biological readout: a clade that needed a lot of tree weight is
one whose expression alone does not localise it.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.perclade_exact as pe

NITER, GAIN, DAMP = 12, 1.2, 0.6
TARGETS = [0.30, 0.45, 0.60, 0.75]
_S = {}


def prep(sec):
    d = np.load(HERE / f"petracer_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    keep = [c for c in np.unique(cc) if (cc == c).sum() >= 30]
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float),
                cc=cc, keep=keep, n=n)


def _init(sec): _S["d"] = prep(sec)


def q90_per_clade(pos, cc, keep):
    R = np.median(np.linalg.norm(pos - np.median(pos, 0), axis=1))
    return {c: float(np.quantile(np.linalg.norm(pos[cc == c] - np.median(pos[cc == c], 0),
                                                axis=1), 0.90) / R) for c in keep}


def _run(job):
    sec, target = job
    S = _S["d"]
    a_c = {c: 0.5 for c in S["keep"]}
    out = []
    for it in range(NITER):
        a = np.array([a_c.get(c, 0.5) for c in S["cc"]])
        T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], a,
                               max_iter=200, tol=1e-9)
        pos = S["xy_st"][np.asarray(T).argmax(1)]
        err = float(np.median(np.linalg.norm(pos - S["xy_sc"], axis=1)))   # evaluation only
        q = q90_per_clade(pos, S["cc"], S["keep"])
        out.append(dict(section=sec, target=target, it=it, err=err,
                        mean_a=float(np.mean(list(a_c.values()))),
                        sd_a=float(np.std(list(a_c.values()))),
                        mean_q=float(np.mean(list(q.values())))))
        for c in S["keep"]:                       # raise alpha where the clade is too spread out
            a_c[c] = float(np.clip((1 - DAMP) * a_c[c]
                                   + DAMP * np.clip(a_c[c] + GAIN * (q[c] - target), 0.0, 0.95),
                                   0.0, 0.95))
    return out, {c: a_c[c] for c in S["keep"]}


if __name__ == "__main__":
    SEC = "M3-4"
    rows, finals = [], {}
    with Pool(4, initializer=_init, initargs=(SEC,)) as pool:
        for out, af in pool.imap_unordered(_run, [(SEC, t) for t in TARGETS]):
            rows += out; finals[out[0]["target"]] = af
    R = pd.DataFrame(rows); R.to_csv(HERE / "coherence_control.csv", index=False)
    print(f"=== {SEC} ===  alpha_tau driven to hit a per-clade coherence target\n")
    for t in TARGETS:
        g = R[R.target == t].sort_values("it")
        print(f"target q90 = {t}")
        print("   iter  " + " ".join(f"{i:>6}" for i in g.it))
        print("   err   " + " ".join(f"{v:>6.0f}" for v in g.err))
        print("   <a>   " + " ".join(f"{v:>6.2f}" for v in g.mean_a))
        print("   sd(a) " + " ".join(f"{v:>6.2f}" for v in g.sd_a))
        af = finals[t]
        srt = sorted(af, key=lambda c: af[c])
        print(f"   converged alpha per clade: "
              + " ".join(f"{c}:{af[c]:.2f}" for c in srt[:4]) + " ... "
              + " ".join(f"{c}:{af[c]:.2f}" for c in srt[-4:]))
        print()
