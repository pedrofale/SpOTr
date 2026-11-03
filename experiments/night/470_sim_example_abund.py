"""SpOTr's deconvolution on the example tumor of Fig. 2, for its inferred-spots panel (Pedro, 2026-11-03: "a panel with 2 subpanels, showing
SpOTr's result and scDOT next to it").

The benchmark's spotr.npz keeps only the maps of the traced cells; the two-level spot colours (non-tumor against tumor first, then the majority
clone) need SpOTr's own estimate of each spot's non-tumor cells, i.e. the deconvolution's abundances.  This refits SpOTr on the example world
(the base world, seed 1) exactly as the benchmark's rule spotr does (scripts/fit_spotr.build, the tree, alpha 0.2), checks that the map equals
the benchmark's phi_tree, and writes <world>/seed1/spotr_example.npz: phi (traced cells x spots) and abund (spots x [non-tumor types, untraced,
traced], cell2location's w; spotr.c2l_gw.abund).

Usage: python 470_sim_example_abund.py   (benchmark_env, GPU)
"""
import os, sys, json, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; WF = ROOT / "workflows" / "cassiopeia_benchmark"; sys.path.insert(0, str(ROOT / "src"))


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    import warnings, yaml; warnings.filterwarnings("ignore")
    import spotr.c2l_gw as M
    fs = mod("fs", WF / "scripts" / "fit_spotr.py"); m455 = mod("m455", HERE / "455_fig_sim_benchmark.py")
    d = m455.example_world().parent; alpha = float(yaml.safe_load(open(WF / "config.yaml"))["spotr_alpha"])
    I = fs.build(str(d.parent / "params.json"), str(d / "problem.npz"), str(d / "trees.npz"), str(d / "spotr.npz"))
    r = M.fit(*I["args"], **I["common"], structure="dm", alpha_tree=alpha, tree_d=I["d"])
    phi = np.asarray(r["phi"], float); ref = np.load(d / "spotr.npz")["phi_tree"].astype(float)
    dev = float(np.abs(phi - ref).max()); print(f"{d}: max |phi - benchmark phi_tree| = {dev:.2e} (float16 storage)")
    np.savez_compressed(d / "spotr_example.npz", phi=phi.astype(np.float16), abund=np.asarray(M.abund(r["th"]), np.float32), alpha=alpha, max_dev=dev)


if __name__ == "__main__":
    main()
