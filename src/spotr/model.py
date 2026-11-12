"""The SpOTr model, with an interface in the style of DestVI in scvi-tools.

    spotr.SpOTr.setup_anndata(sc_adata, labels_key="cell_type", clone_key="clone", tree=newick_or_graph_or_tdata)
    model = spotr.SpOTr(sc_adata, st_adata, spatial_key="spatial")
    model.train(alpha=0.2)
    model.get_proportions(); model.get_mapping(); model.get_cell_locations(); model.choose_alpha([0, 0.1, 0.2, 0.4, 0.95])

The dissociated cells (sc_adata, raw counts) have the cell type of every cell (labels_key) and the clone of each cancer cell (clone_key, missing
for the other cells), and SpOTr maps the cells with a clone to the spots. The cells without a clone whose cell type is that of the mapped cells
are untraced cancer cells, which SpOTr deconvolves as one more reference type. Every other cell type is a reference type of the microenvironment.
The structural term compares the tree distances between the mapped cells with the distances between the spots they are mapped to when a tree is
given, and penalizes the spread of each clone over the spots when only clone labels are given (Methods of the paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import signatures as _sig
from . import tree as _tree

_KEY = "_spotr_setup"


def _counts(adata, layer=None) -> np.ndarray:
    X = adata.layers[layer] if layer is not None else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return X.astype(float)


class SpOTr:
    """Map dissociated cancer cells to the spots of spatial transcriptomics data with their tree or clone labels, jointly with a deconvolution of
    the spots.
    """

    # ------------------------------------------------------------------------------------------------------------ setup
    @staticmethod
    def setup_anndata(adata, labels_key: str, clone_key: str | None = None, tree=None, n_clones: int | None = None, layer: str | None = None,
                      tree_key: str | None = None, length_key: str = "length", depth_key: str | None = None):
        """Register the dissociated cells.

        Args:
            adata: The dissociated cells, with raw counts.
            labels_key: Column of `.obs` with the cell type of every cell.
            clone_key: Column of `.obs` with the clone of each cancer cell, missing (NaN) for the other cells. SpOTr maps the cells with a clone.
            tree: Optional tree over the cancer cells, with leaves named by `obs_names`. It can be a Newick string, a networkx DiGraph, a
                TreeData object (`tree_key` selects one of its `.obst` trees), a DataFrame of tree distances whose rows and columns are named by
                `obs_names`, or the name of an `.obsp` matrix of tree distances between cells. SpOTr maps the leaves of the tree (or the rows of
                the DataFrame) and cuts the tree into `n_clones` clones if `clone_key` is not given.
            n_clones: Number of clones to cut the tree into if `clone_key` is not given.
            layer: Layer with the raw counts (default `.X`).
            tree_key: Tree of `.obst` to use for a TreeData object (default the first).
            length_key: Edge attribute with the branch lengths of a networkx or TreeData tree.
            depth_key: Node attribute with the depth or time of each node, from which the branch lengths are computed instead.
        """
        if labels_key not in adata.obs:
            raise KeyError(f"labels_key '{labels_key}' is not a column of .obs")
        if clone_key is None and tree is None:
            raise ValueError("give clone_key, tree, or both")
        names = np.asarray(adata.obs_names).astype(str)
        if clone_key is not None:
            if clone_key not in adata.obs:
                raise KeyError(f"clone_key '{clone_key}' is not a column of .obs")
            mapped = adata.obs[clone_key].notna().to_numpy()
        else:
            mapped = None
        D = None
        if tree is not None:
            if isinstance(tree, pd.DataFrame):
                T = tree.copy(); T.index, T.columns = T.index.astype(str), T.columns.astype(str)
                if mapped is None:
                    mapped = np.isin(names, list(T.index))
                missing = sorted(set(names[mapped]) - set(T.index))
                if missing:
                    raise ValueError(f"{len(missing)} mapped cells have no tree distances, e.g. {missing[:3]}")
                D = T.loc[names[mapped], names[mapped]].to_numpy(float)
            elif isinstance(tree, str) and tree in adata.obsp:
                full = adata.obsp[tree]; full = full.toarray() if hasattr(full, "toarray") else np.asarray(full)
                if mapped is None:
                    raise ValueError("with a distance matrix in .obsp, give clone_key to say which cells are mapped")
                D = np.asarray(full)[np.ix_(mapped, mapped)].astype(float)
            else:
                if mapped is None:                                                                  # the leaves of the tree are the mapped cells
                    leaves = set(_tree.parse_newick(tree)[0]) if isinstance(tree, str) else None
                    if leaves is None:
                        G = tree.obst[tree_key or list(tree.obst.keys())[0]] if hasattr(tree, "obst") else tree
                        leaves = {str(n) for n in G.nodes if G.out_degree(n) == 0}
                    mapped = np.isin(names, list(leaves))
                D = _tree.tree_distances(tree, names[mapped], tree_key=tree_key, length_key=length_key, depth_key=depth_key)
        if clone_key is not None:
            clones = adata.obs[clone_key].to_numpy()[mapped].astype(str)
        else:
            if n_clones is None:
                raise ValueError("without clone_key, give n_clones to cut the tree into clones")
            clones = np.array([f"clone {k + 1}" for k in _tree.cut_clones(D, n_clones)])
        if mapped.sum() == 0:
            raise ValueError("no cell to map: no cell has a clone")
        adata.uns[_KEY] = dict(labels_key=labels_key, clone_key=clone_key, layer=layer, mapped=np.where(mapped)[0], clones=clones,
                               tree_distances=D, has_tree=D is not None)

    # ------------------------------------------------------------------------------------------------------------ model
    def __init__(self, adata, st_adata, spatial_key: str = "spatial", signatures="mean", st_layer: str | None = None):
        """Set up SpOTr on the registered dissociated cells and the spots.

        Args:
            adata: The dissociated cells, registered with `setup_anndata`.
            st_adata: The spots, with raw counts and positions in `.obsm[spatial_key]`.
            spatial_key: Key of `.obsm` with the positions of the spots.
            signatures: Reference signatures for the deconvolution. "mean" uses the mean counts per cell of each label, "cell2location" uses the
                regression model of cell2location, and a DataFrame (genes x labels) provides signatures computed elsewhere, with a column for
                every reference cell type, every clone and, when some cancer cells have no clone, their cell type.
            st_layer: Layer of `st_adata` with the raw counts (default `.X`).
        """
        if _KEY not in adata.uns:
            raise ValueError("call SpOTr.setup_anndata(adata, ...) first")
        self.adata, self.st_adata, self.setup = adata, st_adata, adata.uns[_KEY]
        genes = [g for g in adata.var_names if g in set(st_adata.var_names)]
        if len(genes) < 10:
            raise ValueError(f"only {len(genes)} genes are shared by the dissociated cells and the spots")
        self.genes = genes
        Xc = _counts(adata[:, genes], self.setup["layer"]); Y = _counts(st_adata[:, genes], st_layer)
        labels = adata.obs[self.setup["labels_key"]].astype(str).to_numpy()
        mapped = np.zeros(adata.n_obs, bool); mapped[self.setup["mapped"]] = True
        cancer_types = set(labels[mapped].tolist()); untraced = ~mapped & np.isin(labels, list(cancer_types))
        ref = ~mapped & ~untraced; self.types = sorted(set(labels[ref].tolist()))
        if not self.types:
            raise ValueError("no reference cell type: every cell is a cancer cell")
        self.clone_names = sorted(set(self.setup["clones"].tolist()), key=lambda c: (len(c), c))
        cidx = {c: k for k, c in enumerate(self.clone_names)}; self.clone = np.array([cidx[c] for c in self.setup["clones"]])
        clash = sorted(set(self.clone_names) & (set(self.types) | cancer_types))
        if clash:
            raise ValueError(f"clone names that are also cell types: {clash}")
        self.untraced_label = (next(iter(cancer_types)) if len(cancer_types) == 1 else "untraced") if untraced.any() else None
        lab = np.array(labels, dtype=object); lab[mapped] = self.setup["clones"]; lab[untraced] = self.untraced_label
        S = _sig.signatures(Xc, lab, genes, signatures); self.signatures_ = S
        mapped_sig = np.stack([S[c].to_numpy() for c in self.setup["clones"]])
        gu = S[self.untraced_label].to_numpy()[None, :] if untraced.any() else None
        self.xy = np.asarray(st_adata.obsm[spatial_key], float)[:, :2]
        self._args = (Y, np.stack([S[t].to_numpy() for t in self.types]), gu, mapped_sig)
        self._X_mapped = Xc[mapped]; self._has_untraced = bool(untraced.any())
        self.mapped_names = np.asarray(adata.obs_names)[mapped]; self._phi = None

    def _fit(self, alpha, beta, clade_weights, verbose):
        from . import _engine
        omega = None
        if clade_weights:
            unknown = sorted(set(map(str, clade_weights)) - set(self.clone_names))
            if unknown:
                raise ValueError(f"unknown clones in clade_weights: {unknown}")
            omega = {self.clone_names.index(str(c)): float(w) for c, w in clade_weights.items()}
        Y, G_ref, g_untraced, X_sig = self._args
        return _engine.fit(Y, G_ref, g_untraced, X_sig, self._X_mapped, self.xy, self.clone, alpha=float(alpha), beta=float(beta),
                           tree_d=self.setup["tree_distances"] if self.setup["has_tree"] else None, omega=omega, verbose=verbose)

    def train(self, alpha: float = 0.2, beta: float = 0.5, clade_weights: dict | None = None, verbose: bool = False):
        """Fit the mapping and the deconvolution jointly.

        Args:
            alpha: Importance of the structural term in [0, 1). SpOTr maps the cells by their expression alone at 0.
            beta: Importance of the expression cost in [0, 1).
            clade_weights: Optional weights {clone: weight} on the structural term. SpOTr maps a clone with weight 0 by its expression alone, and
                weakens the structural term for a clone with a weight below 1.
            verbose: Print the log-likelihood after each round.

        Returns:
            The trained model. The proportions of each spot are also written to `st_adata.obsm["spotr_proportions"]` and the expected positions
            of the mapped cells to `adata.obsm["spotr_spatial"]` (NaN for the other cells).
        """
        from . import _engine
        r = self._fit(alpha, beta, clade_weights, verbose)
        self._phi = np.asarray(r["phi"], float); self._abund = np.asarray(_engine.abund(r["th"]), float); self.costs_ = r["costs"]
        self.history_ = r["trace"]; self.alpha_, self.beta_ = float(alpha), float(beta)
        self.st_adata.obsm["spotr_proportions"] = self.get_proportions()
        loc = np.full((self.adata.n_obs, 2), np.nan); loc[self.setup["mapped"]] = self.get_cell_locations().to_numpy(); self.adata.obsm["spotr_spatial"] = loc
        return self

    # ------------------------------------------------------------------------------------------------------------ outputs
    def _check(self):
        if self._phi is None:
            raise RuntimeError("train the model first")

    def get_mapping(self) -> pd.DataFrame:
        """Return the transport plan as a DataFrame of mapped cancer cells x spots, with each row summing to 1."""
        self._check(); return pd.DataFrame(self._phi, index=self.mapped_names, columns=self.st_adata.obs_names)

    def get_proportions(self) -> pd.DataFrame:
        """Return the proportions of the reference cell types, the untraced cancer cells (under their cell type, if any) and each clone in each
        spot, as a DataFrame of spots x labels.
        """
        self._check(); E = len(self.types); ab = self._abund; u = int(self._has_untraced)
        if ab.shape[1] != E + u + 1:
            raise RuntimeError(f"unexpected abundance layout {ab.shape}")
        mass = self._phi.T @ np.eye(len(self.clone_names))[self.clone]; comp = mass / np.maximum(mass.sum(1, keepdims=True), 1e-300)
        cols = [ab[:, :E]] + ([ab[:, [E]]] if u else []) + [ab[:, [E + u]] * comp]
        P = np.concatenate(cols, axis=1); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
        names = list(self.types) + ([self.untraced_label] if self._has_untraced else []) + list(self.clone_names)
        return pd.DataFrame(P, index=self.st_adata.obs_names, columns=names)

    def get_cell_locations(self) -> pd.DataFrame:
        """Return the expected position of each mapped cell under the mapping, as a DataFrame of cells x 2."""
        self._check(); P = self._phi / np.maximum(self._phi.sum(1, keepdims=True), 1e-300)
        return pd.DataFrame(P @ self.xy, index=self.mapped_names, columns=["x", "y"])

    def choose_alpha(self, alphas=(0.0, 0.1, 0.2, 0.4, 0.95), beta: float = 0.5, verbose: bool = False):
        """Choose alpha at the elbow of the trade-off between the expression cost and the structural cost of the mapping (Fig. S4 of the paper).

        Fit SpOTr at each alpha, normalize both costs between the smallest and the largest alpha, and choose the alpha that minimizes their sum.

        Args:
            alphas: Values of alpha to fit.
            beta: Importance of the expression cost in [0, 1).
            verbose: Print the log-likelihood after each round.

        Returns:
            The chosen alpha, and a DataFrame with the two costs and their normalized sum at each alpha.
        """
        alphas = sorted(float(a) for a in alphas); rows = []
        for a in alphas:
            c = self._fit(a, beta, None, verbose)["costs"]; rows.append(dict(alpha=a, expression=c["expression"], structure=c["structure"]))
        df = pd.DataFrame(rows).set_index("alpha"); E, S = df["expression"].to_numpy(), df["structure"].to_numpy()
        e = (E - E[0]) / max(E[-1] - E[0], 1e-30); s = (S - S[-1]) / max(S[0] - S[-1], 1e-30); df["sum"] = e + s
        return float(alphas[int(np.argmin(e + s))]), df
