"""SpOTr on one simulated problem, as spotr.tex's Methods define it (rewritten 2026-11-02 for the current engine, spotr.c2l_gw.fit; the earlier
engine with learned spreads is in git history).  Run in the spotr environment, GPU.

Inputs as experiments/night/450_ builds them on PEtracer: every dissociated cell labelled (each microenvironment type, the tumour cells without a
lineage readout as one label "untraced", each clone of the tree given to SpOTr -- true or reconstructed -- with the clones below clone_min
"pooled"); one signature per label from the cell2location package's regression (experiments/night/c2l_signatures.py, conda env c2l); the
microenvironment and untraced signatures as the deconvolution's types (untraced "free"), each traced cell's clone signature as its own profile;
the Methods' settings (450_.METHODS: totals fixed, residual cosine expression cost, beta 1/2, linear tree targets, capacities from cell counts).
Arms: clone labels (structure "compact") and the tree (structure "dm" on the tree's distances).
  one alpha (config spotr_alpha; Pedro, 2026-11-02: "produce all the results on a fixed alpha, so that we can share the paper"): each arm at it;
  a grid (config spotr_alphas, the overnight run, rule spotr_grid): each arm over the grid, alpha 0 shared (no structural term), each arm's alpha
  at the elbow of 450_.elbow (the alpha minimising the sum of the expression and structural costs, each normalised between alpha 0 and the
  largest alpha).
The cell2location signatures are cached in the world-seed's folder (c2l_signatures.npz) and shared by both.

Usage: python fit_spotr.py <params.json> <problem.npz> <trees.npz> <alpha | a0,a1,...> <out.npz>
Writes out.npz: phi_tree, phi_labels (at the fixed alpha or each at its elbow); on a grid also phi_expr (alpha 0) and phi_<arm>_<alpha> for every
fit; clone_used; and out.json.
"""
import os, sys, json, time, subprocess
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

ALPHAS = [0.0, 0.1, 0.2, 0.4, 0.95]                                                                   # set from the command line
METHODS = dict(totals="fixed", cell_cost="resid", cell_metric="cosine", beta=0.5, tree_target="linear", capacity="share")   # share frozen 2026-11-02 (HTAN, PEtracer, simulations)
SIG = Path(__file__).resolve().parents[3] / "experiments" / "night" / "c2l_signatures.py"; C2L = Path.home() / "miniconda3" / "envs" / "c2l" / "bin" / "python"


def signatures(Xc, labs, out):
    f_out = str(Path(out).parent / "c2l_signatures.npz"); f_in = f_out.replace(".npz", "_in.npz")
    if not Path(f_out).exists():
        np.savez_compressed(f_in, X=np.asarray(Xc, np.float32), labels=np.asarray(labs).astype(str))
        subprocess.run([str(C2L), str(SIG), f_in, f_out], check=True, env=dict(os.environ, PYTHONNOUSERSITE="1"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(f_in)
    z = np.load(f_out, allow_pickle=True); return dict(zip([str(l) for l in z["labels"]], np.asarray(z["signatures"], float)))


def elbow(costs, arm):
    E = np.array([costs[(arm, a)]["expression"] for a in ALPHAS]); S = np.array([costs[(arm, a)]["structure"] for a in ALPHAS])
    e = (E - E[0]) / max(E[-1] - E[0], 1e-30); s = (S - S[-1]) / max(S[0] - S[-1], 1e-30); return float(ALPHAS[int(np.argmin(e + s))])


def build(pfile, prob, trees, out):
    """the fit's inputs for one world-seed: the positional arguments, the keyword arguments common to every arm, the tree's distances, the clones"""
    from scipy.spatial.distance import pdist, squareform
    p = json.load(open(pfile)); q, t = np.load(prob), np.load(trees)
    rec = p["tree"] == "reconstructed"; d = np.asarray(t["D_rec"] if rec else t["D_true"], float); clone = (t["clone_rec_tree"] if rec else t["clone_true_tree"]).astype(int)
    Xc = q["Xc"].astype(float); kind, me, tr = q["kind"], q["me_type"], q["traced"].astype(bool); tum = kind == 0
    labs = np.array([f"me{e}" for e in me], dtype=object); labs[tum & ~tr] = "untraced"; labs[tr] = [f"c{c}" if c >= 0 else "pooled" for c in clone]
    S = signatures(Xc, labs, out); types = sorted(set(me[~tum].tolist()))
    xy = q["spots"].astype(float); args = (q["Y"].astype(float), np.stack([S[f"me{e}"] for e in types]), S["untraced"][None, :], np.array([tr.sum() / max(tum.sum(), 1)]),
                                           np.stack([S[l] for l in labs[tr]]), np.zeros(int(tr.sum()), int), squareform(pdist(xy)), xy)
    common = dict(verbose=False, clone=clone, mode="clones", X_cells=Xc[tr], untraced="free", **METHODS)
    return dict(p=p, q=q, args=args, common=common, d=d, clone=clone)


def main(pfile, prob, trees, alphas, out):
    global ALPHAS
    ALPHAS = [float(a) for a in str(alphas).split(",")]
    from scipy.spatial.distance import pdist, squareform
    import spotr.c2l_gw as M
    t0 = time.time(); I = build(pfile, prob, trees, out); p, args, common, d, clone = I["p"], I["args"], I["common"], I["d"], I["clone"]
    phis, costs = {}, {}
    if len(ALPHAS) == 1:                                                                               # one fixed alpha
        a = ALPHAS[0]; info = {}
        for arm, kw in (("tree", dict(structure="dm", tree_d=d)), ("labels", dict(structure="compact"))):
            ra = M.fit(*args, **common, alpha_tree=a, **kw); phis[arm] = ra["phi"]; info[arm] = ra["costs"]
        np.savez_compressed(out, **{f"phi_{k}": np.asarray(v, np.float16) for k, v in phis.items()}, clone_used=clone)
        json.dump(dict(record={}, alpha=a, costs=info, seconds=time.time() - t0, tree=p["tree"]), open(out.replace(".npz", ".json"), "w"), default=float)
        print(f"SpOTr: alpha {a}, tree and labels in {time.time() - t0:.0f}s"); return
    r = M.fit(*args, **common, structure="dm", alpha_tree=0.0, tree_d=d); phis["expr"] = r["phi"]; costs[("tree", 0.0)] = r["costs"]
    costs[("labels", 0.0)] = M.fit(*args, **common, structure="compact", alpha_tree=0.0)["costs"]    # the same map, its compaction cost (each arm's
                                                                                                     # structural cost is its own term's)
    for arm, kw in (("tree", dict(structure="dm", tree_d=d)), ("labels", dict(structure="compact"))):
        for a in ALPHAS[1:]:
            ra = M.fit(*args, **common, alpha_tree=a, **kw); phis[f"{arm}_{a}"] = ra["phi"]; costs[(arm, a)] = ra["costs"]
    el = {arm: elbow(costs, arm) for arm in ("tree", "labels")}
    for arm in ("tree", "labels"):
        phis[arm] = phis["expr"] if el[arm] == 0.0 else phis[f"{arm}_{el[arm]}"]
    np.savez_compressed(out, **{f"phi_{k}": np.asarray(v, np.float16) for k, v in phis.items()}, clone_used=clone)
    json.dump(dict(record={}, elbows=el, costs={f"{k[0]} {k[1]}": v for k, v in costs.items()}, alphas=ALPHAS, seconds=time.time() - t0, tree=p["tree"]),
              open(out.replace(".npz", ".json"), "w"), default=float)
    print(f"SpOTr: elbows {el}, {len(phis)} maps in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:6])
