"""alpha_tau is whatever it takes to make clade tau cohere -- with an ISOLATED actuator.

The first version used the unmasked pair-mean cost, where the pair (i,k) is weighted by
(a_i+a_k)/2, so raising a_tau to tighten clade tau also strengthened every constraint between
tau and every other clade. A feedback loop whose actuators are cross-coupled fights itself, and
the error traces oscillated (488, 521, 508, 481, 520...).

Here u_tau multiplies the WITHIN-clade mask only, so it controls clade tau internal cohesion and
nothing else, while a shared cross-clade weight c handles between-clade placement. One isolated
actuator per constraint.

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
CROSS = 0.25                     # shared cross-clade weight, handles BETWEEN-clade placement
UMAX = 0.95                      # ceiling on alpha = u + CROSS, so (1-alpha) stays positive
TARGETS = [0.30, 0.45, 0.60, 0.75]
_S = {}


def prep(sec):
    d = np.load(HERE / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    keep = [c for c in np.unique(cc) if (cc == c).sum() >= 30]
    n, m = M.shape
    _, clab = np.unique(cc, return_inverse=True)
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float),
                cc=cc, keep=keep, n=n, clab=clab)


def _init(sec): _S["d"] = prep(sec)


def q90_per_clade(pos, cc, keep):
    R = np.median(np.linalg.norm(pos - np.median(pos, 0), axis=1))
    return {c: float(np.quantile(np.linalg.norm(pos[cc == c] - np.median(pos[cc == c], 0),
                                                axis=1), 0.90) / R) for c in keep}


def _run(job):
    sec, target = job
    S = _S["d"]
    a_c = {c: 0.4 for c in S["keep"]}
    out = []
    for it in range(NITER):
        u = np.array([a_c.get(c, 0.5) for c in S["cc"]])
        # alpha = u + c, theta = u/(u+c): u_tau weights WITHIN-clade pairs (the isolated
        # actuator), c weights cross-clade pairs and is shared. This is the well-posed
        # (u, c) family -- valid gradient for any per-clade u.
        al = u + CROSS
        th = u / np.maximum(al, 1e-12)
        T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], al,
                               theta=th, labels=S["clab"], max_iter=200, tol=1e-9)
        pos = S["xy_st"][np.asarray(T).argmax(1)]
        err = float(np.median(np.linalg.norm(pos - S["xy_sc"], axis=1)))   # evaluation only
        q = q90_per_clade(pos, S["cc"], S["keep"])
        out.append(dict(section=sec, target=target, it=it, err=err,
                        mean_a=float(np.mean(list(a_c.values()))),
                        sd_a=float(np.std(list(a_c.values()))),
                        mean_q=float(np.mean(list(q.values())))))
        for c in S["keep"]:                       # raise alpha where the clade is too spread out
            # u must satisfy u + CROSS <= UMAX, or alpha = u + c exceeds 1 and the expression
            # term picks up a NEGATIVE weight -- which is what drove the first run to 1080 um.
            hi = UMAX - CROSS
            a_c[c] = float(np.clip((1 - DAMP) * a_c[c]
                                   + DAMP * np.clip(a_c[c] + GAIN * (q[c] - target), 0.0, hi),
                                   0.0, hi))
    return out, {c: a_c[c] for c in S["keep"]}


if __name__ == "__main__":
    SEC = "M3-4"
    rows, finals = [], {}
    with Pool(4, initializer=_init, initargs=(SEC,)) as pool:
        for out, af in pool.imap_unordered(_run, [(SEC, t) for t in TARGETS]):
            rows += out; finals[out[0]["target"]] = af
    R = pd.DataFrame(rows); R.to_csv(HERE / "coherence_control_fair.csv", index=False)
    print(f"=== {SEC} ===  u_tau (within-clade weight) driven to hit a per-clade "
      f"coherence target; cross-clade weight fixed at {CROSS}\n")
    for t in TARGETS:
        g = R[R.target == t].sort_values("it")
        print(f"target q90 = {t}")
        print("   iter  " + " ".join(f"{i:>6}" for i in g.it))
        print("   err   " + " ".join(f"{v:>6.0f}" for v in g.err))
        print("   <a>   " + " ".join(f"{v:>6.2f}" for v in g.mean_a))
        print("   sd(a) " + " ".join(f"{v:>6.2f}" for v in g.sd_a))
        print("   q90   " + " ".join(f"{v:>6.2f}" for v in g.mean_q))
        af = finals[t]
        srt = sorted(af, key=lambda c: af[c])
        print(f"   converged alpha per clade: "
              + " ".join(f"{c}:{af[c]:.2f}" for c in srt[:4]) + " ... "
              + " ".join(f"{c}:{af[c]:.2f}" for c in srt[-4:]))
        print()
