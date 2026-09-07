"""Shared PEtracer solver/scorer. Import-safe: no work at module level.

(run_transfer_matrix.py does its work at module level, so importing it re-runs the whole
transfer matrix -- that is why run_split_half_oracle.py replayed it. Use this instead.)
"""
import sys
from pathlib import Path
import numpy as np, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J, TAU = 0.02, 50, 3, 0.01
SECS = ["M3-1", "M3-2", "M3-4"]
GRID_G = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
GRID_K = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]


class Sec:
    def __init__(self, name):
        d = np.load(HERE / f"petracer_{name}.npz", allow_pickle=True)
        self.name, self.d = name, d
        self.clades = list(np.asarray(d["clades"]).astype(str))
        self.K = len(self.clades)
        cc = np.asarray(d["cell_clade"]).astype(str)
        self.cidx = np.array([self.clades.index(c) for c in cc])
        self.COMP = np.load(HERE / f"comp_{name}.npy")
        A = np.asarray(d["A"], np.float32)
        X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
        self.n, self.m = len(cc), X_st.shape[0]
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(
            ad.AnnData(X_sc), ad.AnnData(X_st), np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
        g0 = a[:, None] * b[None, :]
        sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
        self._p = (Cf, C_T, C_S, a, b, jnp.asarray(A), sF, sL)
        self.floor = float(self.COMP.mean(0)[self.cidx].mean())

    def solve(self, ak):
        Cf, C_T, C_S, a, b, omega, sF, sL = self._p
        def build(g):
            return build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)), Cf,
                                                C_T, C_S, a, b, g, omega, sF, sL,
                                                equalize="measured")
        g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
        return np.asarray(g)

    def score(self, g, cols=None, rows=None):
        if not np.isfinite(g).all(): return np.nan
        cols = np.arange(self.m) if cols is None else cols
        j = cols[np.asarray(g)[:, cols].argmax(1)]
        sel = slice(None) if rows is None else rows
        return float(self.COMP[j[sel], self.cidx[sel]].mean())

    def acc(self, ak, cols=None, rows=None):
        return self.score(self.solve(ak), cols, rows)
