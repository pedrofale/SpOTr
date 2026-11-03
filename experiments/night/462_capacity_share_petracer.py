"""The capacity rule on PEtracer (Pedro, 2026-11-02: "(1) Test it"): capacities from each spot's number of cancer cells ("count", the Methods)
against its cancer share ("share", which took B2 out of A's territory on HT260C1, sections 181 and 184).

Every half-visium section (M3-4, M3-1, M3-2), the tree at alpha 0.2, every other setting as 450_'s quick check (untraced free).  The count fits
are 450_'s (PEtracer_<section>_visium_tree_0.2.npz); here the share fits.  Scores: the per-clade means of PCC, SSIM, RMSE and JS (as fig:petracer)
and, for the undercount of tumour cells without a lineage readout seen on these sections (section 186), each spot's fitted untraced and traced
tumour cells against the truth, and the number of tumour spots under the figure's two-level rule against the truth's.

Usage: python 462_capacity_share_petracer.py [tied]   (`tied`: untraced tied instead of free, capacities from counts -- the undercount check; benchmark_env; CPU is enough: JAX_PLATFORMS=cpu; writes spotr_out/writeup_check/capacity_share_petracer.json)
"""
import os, sys, json, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; OUT = O / "writeup_check"
sys.path.insert(0, str(ROOT / "src"))
import spotr.c2l_gw as M


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def problem(sec):
    """450_.petracer's inputs for half-visium:<sec>"""
    m354, m276, m379 = mod("m354", "354_c2l_cgw_anova.py"), mod("m276", "276_petracer.py"), mod("m379", "379_c2l_reference.py"); m379.OUT = O / "petracer_comparison"
    name = f"half-visium:{sec}"; st = name.replace(":", "_"); p = m354.problem("petracer", name); B = m276.build(name, 0); X = B["X"]; Xc = np.asarray(B["Xc"], float)
    tum = X["kind"] == 0; tr = tum & X["traced"]; K = X["K"]; cl = X["cl"]; me = np.asarray(X["me_type"]); labs = np.array([f"me{e}" for e in me], dtype=object); labs[tum & ~X["traced"]] = "untraced"
    labs[tr] = [f"c{c}" if c < K - 1 else "pooled" for c in cl[tr]]; S = m379.signatures(Xc, labs, f"{st}_c2l"); types = sorted(set(me[~tum].tolist()))
    args = (p["Y"], np.stack([S[f"me{e}"] for e in types]), S["untraced"][None, :], p["fT"], np.stack([S[l] for l in labs[tr]]), p["kc"], p["D"], p["xy"])
    return dict(p=p, X=X, args=args, d=np.asarray(p["d"], float), clone=p["clone"], Xt=Xc[tr], E=len(types), K=K, tr=tr)


def summary(phi, Ab, P, f289, m451):
    X = P["X"]; K, E = P["K"], P["E"]; onehot = np.eye(K)[X["cl"][P["tr"]]]
    pc = m451.per_clone(f289, phi.T @ onehot, X["Ncl"], f289.MIN_CELLS); out = {k: round(float(np.mean(v)), 3) for k, v in pc.items()}
    if Ab.shape[1] == E + 2:
        un_fit, tr_fit = Ab[:, E], Ab[:, E + 1]
    else:                                                                                     # tied: one tumour abundance, split by the traced fraction
        fT = float(np.ravel(P["args"][3])[0]); un_fit, tr_fit = Ab[:, E] * (1 - fT), Ab[:, E] * fT
    un_true = np.maximum(X["Ntum"] - X["Ncl"].sum(1), 0); tr_true = X["Ncl"].sum(1); me_true = X["Nme"].sum(1); me_fit = Ab[:, :E].sum(1)
    out.update(untraced_per_spot=round(float(un_fit.mean()), 2), untraced_true=round(float(un_true.mean()), 2), traced_per_spot=round(float(tr_fit.mean()), 2), traced_true=round(float(tr_true.mean()), 2),
               tumour_spots=int(((un_fit + tr_fit) > me_fit).sum()), tumour_spots_true=int(((un_true + tr_true) > me_true).sum()))
    return out


def main():
    import warnings; warnings.filterwarnings("ignore")
    m450, f289, m451 = mod("m450", "450_writeup_check.py"), mod("f289", "289_field_metrics.py"), mod("m451", "451_fig_petracer_paper.py")
    fj = OUT / "capacity_share_petracer.json"; res = json.load(open(fj)) if fj.exists() else {}
    TIED = "tied" in sys.argv[1:]                                                                # `tied`: untraced tied to the traced fraction, capacities from counts
    for sec in ("M3-4", "M3-1", "M3-2"):
        P = problem(sec); fs = OUT / f"PEtracer_{sec}_visium_tree_0.2_{'tied' if TIED else 'share'}.npz"
        if not fs.exists():
            kw = dict(untraced="tied", capacity="count") if TIED else dict(untraced="free", capacity="share")
            r = M.fit(*P["args"], verbose=False, clone=P["clone"], mode="clones", X_cells=P["Xt"], **kw, **m450.METHODS, structure="dm", alpha_tree=0.2, tree_d=P["d"])
            np.savez_compressed(fs, phi=np.asarray(r["phi"], np.float16), abund=np.asarray(M.abund(r["th"]), np.float32))
        for rule, f in (("count", OUT / f"PEtracer_{sec}_visium_tree_0.2.npz"), ("tied" if TIED else "share", fs)):
            z = np.load(f); res[f"{sec} {rule}"] = summary(z["phi"].astype(float), z["abund"].astype(float), P, f289, m451)
            print(sec, rule, json.dumps(res[f"{sec} {rule}"]), flush=True)
        json.dump(res, open(fj, "w"), indent=1)


if __name__ == "__main__":
    main()
