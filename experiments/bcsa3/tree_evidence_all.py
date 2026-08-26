"""Tests 1 and 2 across every BCSA dataset/section with a joint SS3+Visium Numbat run.

Both tests are data-level: they use only Numbat's clone tree, Numbat's per-spot clone
calls and the spot coordinates. No SpOTr, no expression prep. They ask whether the
premise Gromov-Wasserstein relies on -- tree metric agrees with spatial metric -- holds.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)

RUNS = [
    # label,           numbat run,                     source tag,      tree csv,           visium spatial dir
    ("BCSA2/TumC2", "/mnt/data/pedro/BCSA/BCSA2/numbat_out", "Visium_TumC2", "trees/bcsa2_TumC2.csv",
     "/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro/BCSA2/BCSA2_TumC2_visium/spatial"),
    ("BCSA3/B1", "/mnt/data/pedro/BCSA/BCSA3/numbat_out", "Visium_B1", "trees/bcsa3_B1D1.csv",
     "/mnt/data/pedro/BCSA/BCSA3/V19T26-031_B1/outs/spatial"),
    ("BCSA3/D1", "/mnt/data/pedro/BCSA/BCSA3/numbat_out", "Visium_D1", "trees/bcsa3_B1D1.csv",
     "/mnt/data/pedro/BCSA/BCSA3/V19T26-031_D1/outs/spatial"),
    ("BCSA3/C1", "/mnt/data/pedro/BCSA/BCSA3/numbat_out_ss3_C1", "Visium_C1", "trees/bcsa3_C1.csv",
     "/mnt/data/pedro/BCSA/BCSA3/V19T26-031_C1/outs/spatial"),
    ("BCSA6/TumA1", "/mnt/data/pedro/BCSA/BCSA6/numbat_out", "Visium_TumA1", "trees/bcsa6_TumA1.csv",
     "/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro/visium/BCSA6_TumA1_visium_CR/outs/spatial"),
]


def clone_tree_distances(edges_csv):
    e = pd.read_csv(edges_csv)
    nodes = sorted(set(e["from_clone"]) | set(e["to_clone"]))
    idx = {c: i for i, c in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(D, 0.0)
    for _, r in e.iterrows():
        i, j = idx[r["from_clone"]], idx[r["to_clone"]]
        D[i, j] = D[j, i] = float(r["n_events"])
    for k in range(len(nodes)):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return np.array(nodes), D


def load_coords(spatial_dir):
    p = Path(spatial_dir)
    f = p / "tissue_positions.csv"
    if f.exists():
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        df = df.set_index("barcode")
        return df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].astype(float)
    df = pd.read_csv(p / "tissue_positions_list.csv", header=None,
                     names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"]).set_index("barcode")
    return df[["pxl_row", "pxl_col"]].astype(float)


def run_one(label, nbdir, source, tree_csv, spatial_dir):
    nbdir = Path(nbdir)
    cp = pd.read_csv(nbdir / "clone_post_2.tsv", sep="\t")
    meta = pd.read_csv(nbdir / "cell_metadata.csv")
    nb = cp.merge(meta, on="cell")

    ss3_n = (nb["source"] == "SS3").sum()
    v = nb[nb["source"] == source].copy()
    pre = source.replace("Visium_", "")
    v["bc"] = v["cell"].str.replace(f"^{pre}_", "", regex=True)

    coords = load_coords(spatial_dir)
    v = v[v["bc"].isin(coords.index)]
    xy = coords.loc[v["bc"]].to_numpy()
    truth = v["clone_opt"].to_numpy()

    nodes, Dtree = clone_tree_distances(HERE / tree_csv)
    pos = {c: i for i, c in enumerate(nodes)}
    keep = np.array([c in pos for c in truth])
    xy, truth = xy[keep], truth[keep]
    ti = np.array([pos[c] for c in truth])
    n = len(truth)

    # SS3 cells per clone -- how much reference each clone actually has
    ss3_counts = nb[nb["source"] == "SS3"]["clone_opt"].value_counts()

    # Test 1: spot pairs
    ii, jj = rng.integers(0, n, 300_000), rng.integers(0, n, 300_000)
    ok = ii != jj; ii, jj = ii[ok], jj[ok]
    dt = Dtree[ti[ii], ti[jj]]
    ds = np.linalg.norm(xy[ii] - xy[jj], axis=1)
    r1 = np.corrcoef(dt, ds)[0, 1] if dt.std() > 0 else np.nan
    null = np.array([np.corrcoef(Dtree[ti[p][ii], ti[p][jj]], ds)[0, 1]
                     for p in (rng.permutation(n) for _ in range(200))]) if dt.std() > 0 else np.array([np.nan])
    p1 = (np.sum(np.abs(null) >= abs(r1)) + 1) / (len(null) + 1)

    # Test 2: clone territories
    cl = [c for c in np.unique(truth) if (truth == c).sum() >= 20]
    if len(cl) >= 4:
        cent = np.stack([xy[truth == c].mean(0) for c in cl])
        Ds = np.linalg.norm(cent[:, None] - cent[None, :], axis=-1)
        Dt = Dtree[np.ix_([pos[c] for c in cl], [pos[c] for c in cl])]
        iu = np.triu_indices(len(cl), 1)
        r2 = np.corrcoef(Dt[iu], Ds[iu])[0, 1]
        n2 = np.array([np.corrcoef(Dt[np.ix_(p, p)][iu], Ds[iu])[0, 1]
                       for p in (rng.permutation(len(cl)) for _ in range(5000))])
        p2 = (np.sum(np.abs(n2) >= abs(r2)) + 1) / (len(n2) + 1)
    else:
        r2, p2 = np.nan, np.nan

    # how many clones are usable at all: >=20 spots AND >=10 SS3 cells
    usable = [c for c in cl if ss3_counts.get(c, 0) >= 10]
    return dict(dataset=label, spots=n, ss3=ss3_n, clones_tree=len(nodes),
                clones_20spots=len(cl), clones_usable=len(usable),
                r_pairs=r1, p_pairs=p1, r_mantel=r2, p_mantel=p2)


if __name__ == "__main__":
    rows = [run_one(*r) for r in RUNS]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(HERE / "tree_evidence_all.csv", index=False)
