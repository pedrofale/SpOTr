"""The tissue layout (run in the spotr environment): scCube's reference-free custom cluster pattern (Qian et al. 2024, Nat Commun 15:5021;
github.com/ZJUFanLab/scCube, commit d5b423d), as its "cold tumour-immune microenvironment" tutorial builds it: tumour nests in a stroma.

`nests` oval tumour nests with scCube's default boundary noise (centres uniform over the inner 70% of the section, semi-axes 13-19% and 9-14% of its side, random orientation), each
`nest_purity` tumour with the rest T cells and endothelial cells (0.8 / 0.2) infiltrating; `aggregates` T-cell aggregates in the stroma (small
ovals, 70% T cells, the rest fibroblasts); the background fibroblasts with a share `vessels` of endothelial cells.  scCube places the cells as a
Poisson process; the number of cells is set so that the layout holds about n_cells tumour cells, and the layout is scaled so that cells are cell_spacing um
(default 10) apart on average -- 5 um gives about one cell per 100 um^2, real tissue's density, and ~12 cells per Visium spot of a half section.  Cell types: 0 tumour, 1 fibroblast, 2 T cell, 3 endothelial.

Usage: python layout.py <params.json> <seed> <out.npz>
"""
import sys, json, warnings
import numpy as np

TYPES = ["tumour", "fibroblast", "T cell", "endothelial"]; SIZE = 50.0


def draw(p, seed, cell_num, geom):
    import anndata as ad, pandas as pd
    from scCube.utils import SPatternGeneratorCustom
    a = ad.AnnData(np.zeros((len(TYPES), 1)), obs=pd.DataFrame({"Cell_type": TYPES}, index=[f"c{i}" for i in range(len(TYPES))]))
    g = SPatternGeneratorCustom(sc_adata=a, cell_num=int(cell_num), celltype_key="Cell_type", set_seed=True, seed=int(seed), spatial_size=int(SIZE))
    v = float(p["vessels"]); return g.simulate_cluster(background_celltype=["fibroblast", "endothelial"], background_prop=[1.0 - v, v], **geom)


def main(pfile, seed, out):
    warnings.filterwarnings("ignore"); p = json.load(open(pfile)); rng = np.random.default_rng(int(seed) + 31); n, m = int(p["nests"]), int(p["aggregates"])
    geom = dict(shape_list=["Oval"] * (n + m), cluster_celltype_list=["tumour"] * n + ["T cell"] * m,
                cluster_purity_list=[float(p["nest_purity"])] * n + [0.7] * m,
                infiltration_celltype_list=[["T cell", "endothelial"]] * n + [["fibroblast"]] * m, infiltration_prop_list=[[0.8, 0.2]] * n + [[1.0]] * m,
                center_x_list=(rng.uniform(0.15, 0.85, n + m) * SIZE).tolist(),
                center_y_list=(rng.uniform(0.15, 0.85, n + m) * SIZE).tolist(),
                a_list=(np.r_[rng.uniform(0.13, 0.19, n), rng.uniform(0.04, 0.06, m)] * SIZE).tolist(),
                b_list=(np.r_[rng.uniform(0.09, 0.14, n), rng.uniform(0.04, 0.06, m)] * SIZE).tolist(),
                theta_list=rng.uniform(0, np.pi, n + m).tolist(), scale_value_list=[4.0] * (n + m), twist_value_list=[0.2] * (n + m))
    first = draw(p, seed, 20000, geom); frac = float((first.Cell_type == "tumour").mean())
    lay = draw(p, seed, round(int(p["n_cells"]) / max(frac, 1e-3)), geom); xy = lay[["point_x", "point_y"]].to_numpy(float); ty = np.array([TYPES.index(t) for t in lay.Cell_type])
    from scipy.spatial import cKDTree
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]; xy = (xy - xy.mean(0)) * (float(p.get("cell_spacing", 10.0)) / max(np.mean(nn), 1e-12))
    np.savez_compressed(out, xy=xy, type=ty, types=np.array(TYPES), n_tumour=int((ty == 0).sum()))
    print(f"layout: {len(ty)} cells, " + ", ".join(f"{t} {int((ty == i).sum())}" for i, t in enumerate(TYPES)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
