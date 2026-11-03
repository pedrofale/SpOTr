"""A simulated tumor's files and what the steps share.

A world is the base world of config.yaml with some parameters changed.  Its id is the first ten hex digits of the SHA-1 of its parameters
(JSON, sorted keys), and each seed has a folder <out>/worlds/<id>/seed<s>/ with

    layout.npz   the tissue layout (layout.py)
    sim.npz      the lineage, the tumor cells' positions and clones (simulate.py)
    problem.npz  the spots and the dissociated sample (express.py)
    trees.npz    the true tree's distances between the traced dissociated cells, and their clones (tree.py)
    spotr*.npz   SpOTr's maps (fit_spotr.py); <method>.npz each other method's output (fit_baselines.py)
"""
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = {"c2l": "cell2location", "destvi": "DestVI", "tangram": "Tangram", "scdot": "scDOT", "exact_ot": "expression-only OT", "fgw": "FGW"}


def world_id(p):
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]


def params(config, **changes):
    p = dict(config["base"]); p.update(changes); return p


def studies(config):
    """[(study, parameter, value, world id, parameters)] of Fig. 2's three studies"""
    rows = []
    for study, spec in config["studies"].items():
        (knob, values), = spec["vary"].items()
        rows += [(study, knob, v, world_id(params(config, **{knob: v})), params(config, **{knob: v})) for v in values]
    return rows


def weight_worlds(config):
    """[(sigma_tree, migrate, world id, parameters)] of Fig. S5"""
    w = config["weights"]
    return [(s, m, world_id(params(config, sigma_tree=s, migrate=m)), params(config, sigma_tree=s, migrate=m)) for s in w["sigma_tree"] for m in w["migrate"]]


def example(config, root):
    """the example world's seed folder: the base world"""
    return Path(root) / config["out"] / "worlds" / world_id(params(config)) / f"seed{config['example']['seed']}"


def labels(q, t):
    """each dissociated cell's label: its non-tumor type me<e>, its clone c<k> (or "pooled", the smaller clades) when its lineage is read, and
    "untraced" for the other tumor cells"""
    kind, me, tr = q["kind"], q["me_type"], q["traced"].astype(bool)
    lab = np.array([f"me{e}" for e in me], dtype=object); lab[(kind == 0) & ~tr] = "untraced"
    lab[tr] = [f"c{c}" if c >= 0 else "pooled" for c in t["clone_true_tree"]]; return lab.astype(str)


def masses(d):
    """each method's mass of every true clone per spot (spots x clones) in one seed folder: a map of cells through each traced cell's true clone
    (each cell's row normalised); cell2location's abundances and DestVI's proportions of each label through the true clones of that label's cells.
    Returns ({method: masses}, problem)."""
    d = Path(d); q = np.load(d / "problem.npz", allow_pickle=True); t = np.load(d / "trees.npz"); K = int(q["K"]); tr = q["traced"].astype(bool)
    oh = np.eye(K)[q["clone"][tr]]; rn = lambda P: P / np.maximum(P.sum(1, keepdims=True), 1e-300); out = {}
    if (d / "spotr.npz").exists():
        out["SpOTr"] = np.load(d / "spotr.npz")["phi"].astype(float).T @ oh
    lab = labels(q, t)[tr]
    for f, nm in METHODS.items():
        if not (d / f"{f}.npz").exists():
            continue
        z = np.load(d / f"{f}.npz", allow_pickle=True)
        if "P" in z.files:                                                                          # a map of the dissociated cells
            out[nm] = rn(z["P"].astype(float)[np.where(tr)[0]]).T @ oh
        else:                                                                                       # abundances or proportions per label
            ty = [str(x) for x in z["types"]]; W = z["w"].astype(float); P = np.zeros((len(W), K))
            for L in np.unique(lab):
                if L in ty:
                    P += W[:, [ty.index(L)]] * oh[lab == L].mean(0)[None, :]
            out[nm] = P
    return out, q
