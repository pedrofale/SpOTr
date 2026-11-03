"""DestVI and scDOT, their packages, on PEtracer M3-4 with Visium spots (Pedro, 2026-11-02: "proceed with adding all the missing results except
the very slow ones").

The problem as 276_ builds half-visium:M3-4; every dissociated cell labelled as for cell2location (each clade, the tumour cells without a lineage
readout as one label, each microenvironment type); benchmark/packages.py runs DestVI (scvi-tools) and scDOT (upstream, unmodified, its own map
read out).  Writes spotr_out/paper_figures/petracer_half-visium_M3-4_{destvi,scdot}.npz with `comp`, each spot's amount of every label in the
order of 451_.label_keys: DestVI's proportions; scDOT's mapped cells per label.

Usage: python 458_petracer_packages.py [section=M3-4] [destvi] [scdot]   (benchmark_env; DestVI on the GPU, scDOT on the CPU)
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / "spotr_out" / "paper_figures"
SEC = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("section=")), "M3-4"); NAME, TAG = f"half-visium:{SEC}", f"half-visium_{SEC}"
ONLY = [a for a in sys.argv[1:] if a in ("destvi", "scdot")] or ["destvi", "scdot"]
sys.path.insert(0, str(ROOT / "benchmark")); import packages


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    import warnings; warnings.filterwarnings("ignore")
    m451, m276 = mod("m451", "451_fig_petracer_paper.py"), mod("m276", "276_petracer.py"); B = m276.build(NAME, 0); X = B["X"]
    keys = m451.label_keys(X); labs = m451.labels_of(B, X); Xc, Y = np.asarray(B["Xc"], float), np.asarray(B["Y"], float); OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / f"petracer_{TAG}_destvi.npz"
    if "destvi" in ONLY and not fn.exists():
        t0 = time.time(); w, names = packages.destvi(Xc, labs, Y, seed=0)
        comp = np.stack([w[:, names.index(k)] if k in names else np.zeros(len(w)) for k in keys], 1); np.savez_compressed(fn, comp=comp.astype(np.float32), keys=np.array(keys))
        print(f"DestVI: {len(names)} labels, {time.time() - t0:.0f}s", flush=True)
    fn = OUT / f"petracer_{TAG}_scdot.npz"
    if "scdot" in ONLY and not fn.exists():
        t0 = time.time(); P = packages.scdot(Xc, labs, Y, seed=0); comp = P.T @ m451.label_onehot(B, X)
        np.savez_compressed(fn, comp=comp.astype(np.float32), keys=np.array(keys)); print(f"scDOT: map {P.shape}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
