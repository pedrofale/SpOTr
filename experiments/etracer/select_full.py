"""Select ALL spatial bins for a stage + composition-matched cells (no bin subsampling)."""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from ete3 import Tree

HERE = Path(__file__).resolve().parent
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"


def key_mat(x):
    x = re.sub(r"-\d+$", "", x); p = x.split("_")
    if len(p) > 1 and p[0] == p[1]: p = p[1:]
    a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])


def key_lin(x):
    p = x.split("_"); a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])


def main(stage="LE", total_cells=2000):
    bcs = [l.strip() for l in open(HERE / "export/sc/barcodes.tsv")]
    mk = {key_mat(b): b for b in bcs}
    lin = pd.read_csv(B / "single_cell/sc_cell_lineage.csv")
    lin["matrix_id"] = lin.cellBC.map(key_lin).map(mk)
    lin = lin[lin.matrix_id.notna()]

    sc_trees = {f.name.split("-")[0]: f for f in (B / f"single_cell/sc_trees_{stage}").glob("*.phy.treefile")}
    sp_trees = {f.name.split("-")[0] for f in (B / f"spatial/spatial_trees_{stage}").glob("*.phy.treefile")}
    shared = sorted(set(sc_trees) & sp_trees, key=lambda c: int(c[1:]))

    sp = pd.read_csv(B / f"spatial/spatial_lineage_{stage}.csv").dropna(subset=["lineageGrp", "x", "y"])
    sp["lineageGrp"] = sp.lineageGrp.astype(str)
    sp = sp[sp.lineageGrp.isin(shared)]                    # ALL bins of shared clones
    target = sp.lineageGrp.value_counts(normalize=True)

    # available cells per clone (must be in that clone's tree)
    avail = {}
    for clone in target.index:
        g = lin[lin.lineageGrp.astype(str) == clone]
        leaves = {l.name for l in Tree(str(sc_trees[clone]), format=1).iter_leaves()}
        avail[clone] = g[g.cellBC.isin(leaves)]
    usable = [c for c in target.index if len(avail[c]) >= 8]
    tgt = target[usable] / target[usable].sum()

    # Largest total at which every clone can still hit its target fraction. Without this the
    # dominant clones run out of cells and the composition drifts -- and a mismatched
    # composition is exactly what makes the tree term harmful under balanced OT.
    feasible = min(len(avail[c]) / tgt[c] for c in usable)
    total = int(min(total_cells, feasible))
    rows = []
    for clone in usable:
        want = max(4, int(round(tgt[clone] * total)))
        g = avail[clone]
        rows.append((g if len(g) <= want else g.sample(want, random_state=0)).assign(clone=clone))
    cells = pd.concat(rows, ignore_index=True)
    print(f"[{stage}] feasible total = {int(feasible)}, using {total}")
    sp = sp[sp.lineageGrp.isin(set(cells.clone))]

    md = pd.read_csv(HERE / f"export/spatial_{stage}/metadata.csv", index_col=0)
    # the index is a large integer bin id; pandas may parse it as float, which then
    # stringifies with a trailing ".0" and matches nothing in the matrix barcodes
    md.index = [str(int(float(i))) if str(i).replace(".", "", 1).isdigit() else str(i)
                for i in md.index]
    md["xy"] = md.x.astype(int).astype(str) + "_" + md.y.astype(int).astype(str)
    sp["xy"] = sp.x.astype(int).astype(str) + "_" + sp.y.astype(int).astype(str)
    sp["matrix_id"] = sp.xy.map(dict(zip(md.xy, md.index)))
    sp = sp[sp.matrix_id.notna()]

    comp = cells.clone.value_counts(normalize=True)
    tgt = sp.lineageGrp.value_counts(normalize=True)
    tvd = 0.5 * np.abs(comp.reindex(tgt.index).fillna(0) - tgt).sum()
    print(f"[{stage}] {len(cells)} cells, {len(sp)} bins, {sp.lineageGrp.nunique()} clones, TVD={tvd:.3f}")
    vc = sp.lineageGrp.value_counts()
    print(f"[{stage}] clones with >=40 bins: {(vc>=40).sum()}   >=80: {(vc>=80).sum()}")

    cells.to_csv(HERE / f"sel_cells_{stage}.csv", index=False)
    sp.to_csv(HERE / f"sel_bins_{stage}.csv", index=False)
    pd.Series(cells.matrix_id.values).to_csv(HERE / f"sel_cell_ids_{stage}.txt", index=False, header=False)
    pd.Series(sp.matrix_id.astype(str).values).to_csv(HERE / f"sel_bin_ids_{stage}.txt", index=False, header=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="LE")
    ap.add_argument("--total-cells", type=int, default=2000, dest="total_cells")
    main(**vars(ap.parse_args()))
