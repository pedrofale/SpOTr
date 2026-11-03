"""SpOTr with cell2location's reference profiles (Pedro, 2026-10-31: "our approach for reference profiles should be whatever cell2location's
is"): every reference profile -- each traced cell's (its subclone's signature), the untraced tumour cells', each microenvironment type's -- is
cell2location's RegressionModel signature for that label (c2l_signatures.py), instead of means of the dissociated cells (traced cells: 15-neighbour
means within the clone).  On the gene-count testbed (376_).

FIXED BEFORE RUNNING (2026-10-31):
  E236  with cell2location's reference profiles, SpOTr's per-spot subclone error at 3,000 genes within 0.02 of its error at 300 genes (no tree
        and lineage, each seed).
Read: against the measured profiles and the factor-model profiles (378_, 20 factors).

Usage: python 379_c2l_reference.py   (benchmark_env, GPU; cell2location in the c2l environment; writes spotr_out/c2l_reference/testbed.json)
"""
import os, sys, json, subprocess, importlib.util
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; WF = ROOT / "workflows" / "cassiopeia_benchmark"; sys.path.insert(0, str(ROOT / "src")); OUT = HERE / "spotr_out" / "c2l_reference"
import spotr.c2l_gw as M


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def signatures(X, labels, tag):
    f_in, f_out = OUT / f"{tag}_in.npz", OUT / f"{tag}_sig.npz"
    if not f_out.exists():
        np.savez_compressed(f_in, X=np.asarray(X, np.float32), labels=np.asarray(labels).astype(str))
        subprocess.run([str(Path.home() / "miniconda3/envs/c2l/bin/python"), str(HERE / "c2l_signatures.py"), str(f_in), str(f_out)], check=True, env=dict(os.environ, PYTHONNOUSERSITE="1"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    z = np.load(f_out, allow_pickle=True); return dict(zip([str(l) for l in z["labels"]], np.asarray(z["signatures"], float)))


def main():
    import warnings; warnings.filterwarnings("ignore")
    fs, sc = mod("fs", WF / "scripts" / "fit_spotr.py"), mod("sc", WF / "scripts" / "score.py"); out = []; OUT.mkdir(parents=True, exist_ok=True)
    fac = {(r["genes"], r["seed"]): r for r in json.load(open(HERE / "spotr_out" / "genes_lowrank20.json"))}
    for d in sorted((WF / "results_genes" / "worlds").iterdir()):
        p = json.load(open(d / "params.json"))
        for seed in (1, 2):
            sd = d / f"seed{seed}"; q = np.load(sd / "problem.npz"); I = fs.inputs(p, sd / "problem.npz", sd / "trees.npz"); tr, kind, me = q["traced"], q["kind"], q["me_type"]
            K = int(q["K"]); onehot = np.eye(K)[q["clone"][tr]]; clone = I["clone"]
            labs = np.array([f"me{e}" for e in me], dtype=object); labs[(kind == 0) & ~tr] = "untraced"; labs[tr] = [f"c{c}" if c >= 0 else "pooled" for c in clone]
            S = signatures(q["Xc"].astype(float), labs, f"{d.name}_{seed}")
            Xh = np.stack([S[l] for l in labs[tr]]); Gme = np.stack([S[f"me{e}"] for e in range(3)]); gu = S["untraced"][None, :]
            args = (I["Y"], Gme, gu, I["fT"], Xh, I["kc"], I["D"], I["xy"]); lev = M.nested_levels(I["d"], clone, I["Kc"])
            old = json.load(open(sd / "scores.json")); row = dict(genes=int(p["genes"]), seed=seed)
            for arm, kw in [("notree", {}), ("lineage", dict(clone=clone, mode="lineage", lev=lev, CT=M.tree_ct("lineage", clone, I["Kc"], I["d"], I["Dk"], M.ct_target(I["D"]))))]:
                r = M.fit(*args, verbose=False, **kw); row[arm] = float(sc.tv(np.asarray(r["phi"], float).T @ onehot, q["Ncl"].astype(float)))
                row[arm + "_measured"] = old[arm]["clone_tv"]; row[arm + "_factor20"] = fac[(row["genes"], seed)][arm]
            out.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(out, open(OUT / "testbed.json", "w"), indent=1)


if __name__ == "__main__":
    main()
