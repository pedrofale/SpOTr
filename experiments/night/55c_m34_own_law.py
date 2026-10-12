"""DIAGNOSTIC for 55_: is the M3-4 loss a TRANSFER failure?  Re-run M3-4's law arms with M3-4's OWN
(L, c), fitted on its own ground truth.  Not an honest arm -- it uses the target's positions -- it only
separates 'the law is wrong for M3-4' from 'the transferred mixing length is wrong for M3-4'."""
import importlib.util, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)

def run(job):
    m55.ONLY = {"decomposed, law", "law, continuous"}
    rows, meta = m55.run(job)
    return [r for r in rows if r["arm"] in ("decomposed, law", "law, continuous")], meta

if __name__ == "__main__":
    own = m55.fit_law([m55.law_points("M3-4")])
    print(f"M3-4 own law: L = {own[0]:.0f} um, c = {own[1]:.2f}", flush=True)
    m55_src = (HERE / "55_decomposition_check.py").read_text()
    # skip FGW / ground-truth arms: they do not depend on the law
    jobs = [("M3-4", sd, hf, own) for sd in (0, 1, 2) for hf in (0, 1)]
    rows = []
    with Pool(6) as pool:
        for r, _ in pool.imap_unordered(run, jobs):
            rows += r
    pd.DataFrame(rows).to_csv(HERE / "decomposition_check_m34_own_law.csv", index=False)
    print("written", flush=True)
