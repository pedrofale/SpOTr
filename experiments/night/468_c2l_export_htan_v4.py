"""cell2location's input for HT260C1 with the v4 clone calls (Pedro, 2026-11-02: "Yes" to running cell2location overnight; Fig. 4 and its supplement
used cell2location on the earlier v2 calls).

As 355_'s export for HTAN, but with every cancer nucleus labelled by its v4 clone (A, B1, B2; v4 has no unassigned cancer nuclei) and every other
nucleus by its cell type; the fit genes and spots of the v4 problem.  Writes spotr_out/c2l_package/htan_HT260C1_v4_in.npz, which
`355_c2l_package.py fit` (env c2l) then fits like the others (30,000 epochs).

Usage: python 468_c2l_export_htan_v4.py   (benchmark_env, CPU)
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / "spotr_out" / "c2l_package"; SEC = "HT260C1-Th1K1Fc2U1Z1Bs1"


def main():
    z = np.load(ROOT / "data" / "htan" / f"{SEC}_v4.npz", allow_pickle=True); ct = np.asarray(z["cell_type"]).astype(str); tr = z["traced"].astype(bool)
    labs = ct.astype(object); ti = np.where(tr)[0]
    labs[ti] = np.where(z["sn_clone"].astype(str)[ti] == "A", "A", z["b_sub"].astype(str)[ti]); labs = labs.astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "htan_HT260C1_v4_in.npz", Xc=np.asarray(z["Xc_fit"], np.float32), Y=np.asarray(z["Ys_fit"], np.float32), labels=labs,
                        clones=np.array(["A", "B1", "B2"]), nhat=10.0, genes=np.asarray(z["fit_genes"]).astype(str))
    u, c = np.unique(labs, return_counts=True); print("exported", dict(zip(u.tolist(), c.tolist())))


if __name__ == "__main__":
    main()
