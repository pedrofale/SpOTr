"""52_ continued: thin the barcoded reference spots and find where (if anywhere) FGW passes the vote.
Held-out half and metric unchanged; the reference half is subsampled to a fraction of ALL spots.
FGW uses no reference labels, so its held-out score does not change with density."""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
HERE = Path(__file__).resolve().parent

src = (HERE / "52_gonogo_etracer_knn.py").read_text().split("rows = []")[0]
g = {"__file__": str(HERE / "52_gonogo_etracer_knn.py")}; exec(src, g)
rows = []
for tag in ["LE", "EE"]:
    d = np.load(HERE / f"etracer_spots_{tag}.npz", allow_pickle=True)
    comp = d["spot_comp"].astype(float); xy = d["xy_st"].astype(float); clones = d["clones"].astype(str); m = len(xy)
    for seed in (0, 1, 2):
        perm = np.random.default_rng(seed).permutation(m); ref, held = perm[:m // 2], perm[m // 2:]
        for f in [0.5, 0.2, 0.1, 0.05, 0.02]:
            for rep in range(5):
                sub = np.random.default_rng(100 * seed + rep).permutation(ref)[:max(3, int(f * m))]
                sub = sub[comp[sub].sum(1) > 0]
                loo = {k: g["score_density"](g["knn_density"](xy[sub], comp[sub], xy[sub], k, True), comp[sub], clones)
                       for k in g["KGRID"] if k < len(sub)}
                ks = max(loo, key=loo.get)
                rows.append(dict(tag=tag, seed=seed, frac=f, n_ref=len(sub), k=ks,
                                 r=g["score_density"](g["knn_density"](xy[sub], comp[sub], xy[held], ks), comp[held], clones)))
df = pd.DataFrame(rows); df.to_csv(HERE / "gonogo_etracer_density.csv", index=False)
print(df.groupby(["tag", "frac"]).agg(n_ref=("n_ref", "mean"), r=("r", "mean"), r_sd=("r", "std")).round(3))
print("\nFGW (w by other seeds), unchanged by density: LE 0.568  EE 0.470")
