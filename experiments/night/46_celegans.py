"""DOES THE CLONE-SPREAD LAW HOLD IN A COMPLETELY DIFFERENT SYSTEM?

packer19 is the C. elegans embryo: 988 cells with a full transcriptome, 3D positions, and the
INVARIANT lineage -- so the tree is known exactly and divergence times are known in minutes, with
no recorder noise and no reconstruction error.  If r(a) is a real feature of how tissue develops
rather than an artefact of one tumour and one recorder, it should be visible here too.

Caveat to state up front: the positions come from the C. elegans embryonic atlas keyed on cell
identity, not from imaging the same cells that were sequenced.  For r(a) that is fine -- it needs
lineage and position, not expression -- but it is not an in-situ joint measurement.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, pycea, networkx as nx
from pathlib import Path
from scipy.stats import pearsonr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.clonespread import fit_spread

t = pycea.datasets.packer19()
o = t.obs
xyz = np.c_[o["x"].to_numpy(float), o["y"].to_numpy(float), o["z"].to_numpy(float)]
ok = np.isfinite(xyz).all(1) & o["birth_time"].notna().to_numpy() & o["time"].notna().to_numpy()
cells = np.array(o.index)[ok]
xyz = xyz[ok]
bt = o["birth_time"].to_numpy(float)[ok]
tm = o["time"].to_numpy(float)[ok]
print(f"{ok.sum()} of {len(o)} cells usable (finite position, birth and sample time)")
print(f"spatial extent x/y/z: {np.ptp(xyz, 0).round(1)}   sample times {tm.min():.0f}-{tm.max():.0f} min")

G = t.obst["tree"]
UG = G.to_undirected()
root = [n for n in G if G.in_degree(n) == 0][0]
depth = nx.single_source_shortest_path_length(G, root)
# birth time of every node: leaves from obs, internal from the earliest descendant leaf's parent
bt_map = dict(zip(cells, bt))
lca = nx.tree_all_pairs_lowest_common_ancestor
idx = {c: i for i, c in enumerate(cells)}
pairs = [(a, b) for i, a in enumerate(cells) for b in cells[i+1:]]
print(f"computing LCAs for {len(pairs):,} pairs ...", flush=True)
L = dict(lca(G, root=root, pairs=pairs))

# elapsed divergence time for each cell = its sample time minus the LCA's birth time.
# the LCA's birth time is taken as the max birth_time among leaves at or below it that we know,
# approximated by the birth time of the earliest observed descendant.
bt_node = {}
for n in nx.dfs_postorder_nodes(G, root):
    kids = list(G.successors(n))
    if not kids:
        bt_node[n] = bt_map.get(n, np.nan)
    else:
        v = [bt_node[k] for k in kids if np.isfinite(bt_node.get(k, np.nan))]
        bt_node[n] = min(v) if v else np.nan

rows = []
for (a, b), anc in L.items():
    ta = bt_node.get(anc, np.nan)
    if not np.isfinite(ta):
        continue
    ia, ib = idx[a], idx[b]
    age = ((tm[ia] - ta) + (tm[ib] - ta)) / 2.0      # mean elapsed time since the split, minutes
    if age <= 0:
        continue
    rows.append((age, float(np.linalg.norm(xyz[ia] - xyz[ib]))))
A = np.array(rows)
print(f"{len(A):,} pairs with a resolvable common ancestor")
age, sep = A[:, 0], A[:, 1]
print(f"divergence age {age.min():.0f}-{age.max():.0f} min;  separation {sep.min():.1f}-{sep.max():.1f}")
print(f"\ncorr(divergence age, separation) = {pearsonr(age, sep).statistic:+.3f}")
print(f"  PEtracer for comparison: +0.41 / +0.27 / +0.32")

sp = fit_spread(age * 2, sep)          # fit_spread halves its first argument
print("\nr(a), the clone-spread curve, in the C. elegans embryo:")
for a_, r_ in zip(sp["age"], sp["r"]):
    print(f"   {a_:6.1f} min   r = {r_:6.2f}")
sm = np.array([sep[(age >= lo) & (age < hi)].mean()
               for lo, hi in zip(sp["age"][:-1], sp["age"][1:])])
print(f"\nmonotone increasing: {bool(np.all(np.diff(sp['r']) >= -1e-9))}")
print(f"r spans {sp['r'].min():.2f} to {sp['r'].max():.2f}  ({sp['r'].max()/sp['r'].min():.2f}x)")
np.save(HERE / "celegans_law.npy", A)
