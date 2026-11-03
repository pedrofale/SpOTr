"""Choosing SpOTr's settings on a tuning world that the benchmark does not score (Pedro, 2026-11-02: "pick the SpOTr parameters that make it
unequivocally better than the other methods on the average scenario").

The calibrated benchmark's base world (every axis at its middle level) with seed 3 -- the benchmark scores seeds 1 and 2 -- built by the workflow
with every competitor's map.  SpOTr with the tree over alpha {0, 0.1, 0.2, 0.3, 0.4} x beta {0.25, 0.5, 0.75}, every other setting as the
benchmark (scripts/fit_spotr.build: share capacities, untraced free, cosine cost, linear targets).  Criterion: the mean per-clone PCC (the main
metric of Fig. 2, 451_.per_clone, clones with at least 20 cells); the four pooled metrics (289_field_metrics.scores) reported beside it.
The chosen setting is then fixed for every scenario and for the real data.  Writes spotr_out/tuning/tuning.json and the maps.

Usage: python 469_tune_spotr.py   (benchmark_env, GPU)
"""
import os, sys, json, importlib.util, hashlib
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; WF = ROOT / "workflows" / "cassiopeia_benchmark"; OUT = HERE / "spotr_out" / "tuning"
sys.path.insert(0, str(ROOT / "src"))
ALPHAS, BETAS = [0.0, 0.1, 0.2, 0.3, 0.4], [0.25, 0.5, 0.75]


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    import warnings, yaml; warnings.filterwarnings("ignore")
    import spotr.c2l_gw as M
    fs = mod("fs", WF / "scripts" / "fit_spotr.py"); f289 = mod("f289", HERE / "289_field_metrics.py"); m451 = mod("m451", HERE / "451_fig_petracer_paper.py")
    m455 = mod("m455", HERE / "455_fig_sim_benchmark.py")
    c = yaml.safe_load(open(WF / "config.yaml")); p = dict(c["base"]); w = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
    d = WF / c["out"] / "worlds" / w / "seed3"; OUT = HERE / "spotr_out" / "tuning" / w; OUT.mkdir(parents=True, exist_ok=True)   # one folder per tuning world
    fj = OUT / "tuning.json"; res = json.load(open(fj)) if fj.exists() else {}
    I = fs.build(str(d.parent / "params.json"), str(d / "problem.npz"), str(d / "trees.npz"), str(d / "spotr.npz")); q = I["q"]; K = int(q["K"]); tr = q["traced"].astype(bool)
    oh = np.eye(K)[q["clone"][tr]]
    def score(Pm):
        pc = m451.per_clone(f289, Pm, q["Ncl"].astype(float), f289.MIN_CELLS)["PCC"]; s_ = f289.scores(Pm, q["Ncl"].astype(float))
        return dict(clone_PCC=float(np.mean(pc)), n_clones=len(pc),                          # unrounded: the criterion (ties at 3 decimals, 2026-11-03)
                     **{k: round(float(s_[k]), 3) for k in ("PCC", "SSIM", "RMSE", "JS")})
    for b in BETAS:
        for a in ALPHAS:
            key = f"SpOTr alpha {a} beta {b}"
            if key in res:
                continue
            fn = OUT / f"a{a}_b{b}.npz"
            if not fn.exists():
                common = dict(I["common"]); common["beta"] = b
                np.savez_compressed(fn, phi=np.asarray(M.fit(*I["args"], **common, structure="dm", alpha_tree=a, tree_d=I["d"])["phi"], np.float16))
            res[key] = score(np.load(fn)["phi"].astype(float).T @ oh); json.dump(res, open(fj, "w"), indent=1); print(key, res[key], flush=True)
    ms, _ = m455.masses(d)
    for nm, Pm in ms.items():
        if not nm.startswith("SpOTr"):
            res[nm] = score(Pm); print(nm, res[nm], flush=True)
    json.dump(res, open(fj, "w"), indent=1)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["clone_PCC"])[:8]:
        print(f"{v['clone_PCC']:.3f}  {k}")


if __name__ == "__main__":
    main()
