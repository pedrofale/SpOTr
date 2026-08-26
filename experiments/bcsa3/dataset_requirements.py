"""Why do these datasets not support a SpOTr demonstration?

Four measurable preconditions, scored on every BCSA dataset/section with a joint
SS3+Visium Numbat run. Each is necessary; none is sufficient alone.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tree_evidence_all import RUNS, clone_tree_distances, load_coords  # noqa: E402

rng = np.random.default_rng(0)


def analyse(label, nbdir, source, tree_csv, spatial_dir):
    nbdir = Path(nbdir)
    nb = pd.read_csv(nbdir / "clone_post_2.tsv", sep="\t").merge(
        pd.read_csv(nbdir / "cell_metadata.csv"), on="cell")
    v = nb[nb["source"] == source].copy()
    pre = source.replace("Visium_", "")
    v["bc"] = v["cell"].str.replace(f"^{pre}_", "", regex=True)
    coords = load_coords(spatial_dir)
    v = v[v["bc"].isin(coords.index)]
    xy = coords.loc[v["bc"]].to_numpy()
    truth = v["clone_opt"].to_numpy()
    ss3 = nb[nb["source"] == "SS3"]["clone_opt"]

    nodes, _ = clone_tree_distances(HERE / tree_csv)
    clones = np.array(sorted(set(truth) | set(ss3.unique())))

    # (a) usable clones: enough SS3 reference AND enough spots to score
    cnt_ss3 = ss3.value_counts()
    cnt_sp = pd.Series(truth).value_counts()
    usable = [c for c in clones if cnt_ss3.get(c, 0) >= 10 and cnt_sp.get(c, 0) >= 20]

    # (b) spatial coherence of the clone labels: do clones form territories at all?
    k = 12
    nnb = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    nbrs = nnb.kneighbors(xy, return_distance=False)[:, 1:]
    same = (truth[nbrs] == truth[:, None]).mean()
    # chance level given the same clone frequencies
    p = np.array([(truth == c).mean() for c in np.unique(truth)])
    chance = (p ** 2).sum()
    coherence = (same - chance) / (1 - chance)      # 0 = random, 1 = perfectly segregated

    # (c) composition mismatch between the modalities (total variation distance)
    fs = np.array([cnt_ss3.get(c, 0) for c in clones], float); fs /= fs.sum()
    fv = np.array([cnt_sp.get(c, 0) for c in clones], float); fv /= fv.sum()
    tvd = 0.5 * np.abs(fs - fv).sum()

    # (d) how much of the Visium is in clones the SS3 barely covers
    thin = sum(fv[i] for i, c in enumerate(clones) if cnt_ss3.get(c, 0) < 10)

    return dict(dataset=label, clones=len(nodes), usable=len(usable),
                coherence=coherence, comp_TVD=tvd, visium_frac_unreferenced=thin,
                ss3_cells=len(ss3), spots=len(truth))


if __name__ == "__main__":
    df = pd.DataFrame([analyse(*r) for r in RUNS])
    ev = pd.read_csv(HERE / "tree_evidence_all.csv")[["dataset", "r_pairs", "r_mantel", "p_mantel"]]
    df = df.merge(ev, on="dataset")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(HERE / "dataset_requirements.csv", index=False)
    print("""
coherence               : clone labels' spatial segregation, 0 = random, 1 = perfect territories
comp_TVD                : total-variation distance between SS3 and Visium clone composition (0 = identical)
visium_frac_unreferenced: fraction of spots whose clone has < 10 SS3 cells to map from
r_mantel                : agreement between tree distance and territory distance (SpOTr's premise)""")
