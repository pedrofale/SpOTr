"""The capacity rule on the simulations (Pedro, 2026-11-02: "(1) Test it"): capacities from each spot's number of cancer cells ("count", the
Methods and the benchmark's spotr.npz) against its cancer share ("share").

Every world-seed of the Cassiopeia workflow's current run (Visium spots), SpOTr with the tree and with clone labels at alpha 0.2 through
scripts/fit_spotr.build, capacity "share" -> spotr_share.npz beside each spotr.npz (phi_tree, phi_labels).  Scores as Fig. 2 (289_field_metrics.scores
on the clone masses) and the finest sub-clade error (score.py), count against share, per world-seed; summary in
spotr_out/writeup_check/capacity_share_sims.json.

Usage: python 465_capacity_share_sims.py   (benchmark_env, GPU)
"""
import os, sys, json, importlib.util, hashlib
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; WF = ROOT / "workflows" / "cassiopeia_benchmark"; OUT = HERE / "spotr_out" / "writeup_check"
sys.path.insert(0, str(ROOT / "src"))


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    import warnings, yaml; warnings.filterwarnings("ignore")
    import spotr.c2l_gw as M
    fs = mod("fs", WF / "scripts" / "fit_spotr.py"); f289 = mod("f289", HERE / "289_field_metrics.py")
    c = yaml.safe_load(open(WF / "config.yaml")); out = WF / c["out"] / "worlds"; rows = []
    for st, spec in c["studies"].items():
        (knob, values), = spec["vary"].items()
        for v in values:
            p = dict(c["base"]); p.update(spec.get("fix", {})); p[knob] = v; w = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
            for sd in sorted((out / w).glob("seed*")):
                if not (sd / "spotr.npz").exists():
                    continue
                fn = sd / "spotr_share.npz"; I = fs.build(str(out / w / "params.json"), str(sd / "problem.npz"), str(sd / "trees.npz"), str(sd / "spotr.npz"))
                if not fn.exists():
                    ph = {a: M.fit(*I["args"], **I["common"], alpha_tree=0.2, capacity="share", **kw)["phi"]
                          for a, kw in (("tree", dict(structure="dm", tree_d=I["d"])), ("labels", dict(structure="compact")))}
                    np.savez_compressed(fn, **{f"phi_{a}": np.asarray(v_, np.float16) for a, v_ in ph.items()})
                q, t = I["q"], np.load(sd / "trees.npz"); K = int(q["K"]); tr = q["traced"].astype(bool); ct = q["clone"][tr]; oh = np.eye(K)[ct]
                lev = M.nested_levels(t["D_true"], np.where(ct < K - 1, ct, -1), K - 1)
                for rule, f in (("count", sd / "spotr.npz"), ("share", fn)):
                    z = np.load(f)
                    for a in ("tree", "labels"):
                        phi = z[f"phi_{a}"].astype(float); s_ = f289.scores(phi.T @ oh, q["Ncl"].astype(float))
                        rows.append(dict(study=st, value=v, world=w, seed=sd.name, rule=rule, arm=a, sub=M.subclade_errors(phi, lev, q["xy_true"][tr], q["spots"])["subclade_tv_l3"],
                                         **{k: s_[k] for k in ("PCC", "SSIM", "RMSE", "JS")}))
                print(st, v, sd.name, "done", flush=True)
    json.dump(rows, open(OUT / "capacity_share_sims.json", "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows); print(df.groupby(["arm", "rule"])[["PCC", "SSIM", "RMSE", "JS", "sub"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
