"""Score SpOTr and the cached DestVI/scDOT baselines on identical footing."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import score, fmt  # noqa: E402

CACHE = HERE.parent.parent / "notebooks" / "bcsa3_benchmark_cache"


def main(section="B1"):
    d = np.load(HERE / f"bcsa3_{section}_inputs.npz", allow_pickle=True)
    clones, truth, soft = d["clones"], d["spot_clone_numbat"], d["spot_soft"]
    spots = [str(s) for s in d["spots"]]
    cols = [str(c) for c in clones]
    rows = []

    for name, f in [("DestVI", f"destvi_{section}.csv"), ("scDOT", f"scdot_{section}.csv")]:
        p = CACHE / f
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0).reindex(index=spots, columns=cols).fillna(0.0)
        rows.append(score(df.to_numpy(), clones, truth, soft, name))

    # CNV-only control: per-spot NNLS onto the Numbat clone CNV profiles
    A = (d["clone_cnvs"] / 2.0).T
    P = np.zeros((d["R_st"].shape[0], A.shape[1]))
    for i in range(P.shape[0]):
        P[i], _ = nnls(A, d["R_st"][i])
    rows.append(score(P, clones, truth, soft, "NNLS on CNV (control)"))

    for f in sorted(HERE.glob("prop_*.npy")):
        rows.append(score(np.load(f), clones, truth, soft, f"SpOTr [{f.stem[5:]}]"))

    rows.append(score(np.tile(np.bincount(truth, minlength=len(clones) + 1)[1:], (len(spots), 1)),
                      clones, truth, soft, "predict majority clone"))
    print(f"BCSA3 / {section} — {len(spots)} spots, {len(clones)} clones, "
          f"majority baseline {rows[0]['majority']:.3f}\n")
    print(fmt(rows))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["B1"]))
