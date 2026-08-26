"""SpOTr on PEtracer, disjoint split. Tests claim (a) and claim (b) on the same data.

PEtracer reads lineage in situ, so this is a CONSTRUCTED unmatched problem: the traced cells of
a section are split disjointly, half keeping their tree as the dissociated side, half binned
into pseudo-spots. No cell appears in its own spot (verified: expression reconstructed from raw
counts at 0.000e+00 error, and no cell index shared between the sides).

Why this dataset. Premise per section: coherence 0.241-0.359, Mantel +0.33 to +0.53, all
p < 0.002, signs agreeing across three sections - the strongest of anything screened, and it
survives the binning (spot coherence +0.368, Mantel +0.302). And 18 well-populated clades,
where eTRACER's 45 left a third with no data and per-clade alpha unestimable.

SCORING. spot_clade is a strict majority in only 52% of spots and its tie-breaks skew toward
low clade ids, so argmax-against-spot_clade is biased. The primary metric is therefore SOFT
accuracy: for a cell assigned to a spot, the fraction of that spot's member cells sharing the
cell's clade - the probability a random member of the assigned spot is a true clade-mate. It
uses the full composition and no tie-break. Argmax accuracy is reported alongside for
comparability with the eTRACER numbers.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
SEC = sys.argv[1] if len(sys.argv) > 1 else "M3-4"
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)
d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
cell_clade = np.asarray(d["cell_clade"]).astype(str)
clades = np.asarray(d["clades"]).astype(str)
A = np.asarray(d["A"], np.float32)
K, n, m = len(clades), len(cell_clade), X_st.shape[0]
cidx = np.array([list(clades).index(c) for c in cell_clade])

# ---- true spot composition, from the spot-side cells' clades ----
import pycea as py
t = py.datasets.koblan25("tumor")
cl_of = t.obs["clade"].astype(str).to_dict()
sc_names = np.asarray(d["spot_cell_names"]).astype(str)
spot_of = np.asarray(d["spot_of_cell"])
COMP = np.zeros((m, K))
for nm, sp in zip(sc_names, spot_of):
    c = cl_of.get(nm)
    if c in list(clades):
        COMP[sp, list(clades).index(c)] += 1
COMP = COMP / np.maximum(COMP.sum(1, keepdims=True), 1e-12)
print(f"[run] {SEC}: {n} cells x {m} spots, {K} clades")
print(f"[run] spot composition built from {len(sc_names)} spot-side cells; "
      f"mean max-share {COMP.max(1).mean():.3f}")

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
L = np.asarray(S.compute_Lgw(C_T, C_S, a, b, g0))
print(f"[run] |L|/|F| = {np.abs(L).mean()/np.abs(np.asarray(Cf)).mean():.3e}")


def solve(ak, tau=None):
    def build(g):
        return build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)), Cf, C_T,
                                            C_S, a, b, g, omega, sF, sL, equalize="measured")
    if tau is None:
        g = g0
        for _ in range(J):
            g, _u = S.sinkhorn_unrolled_safe(build(g), a, b, EPS, T)
        return np.asarray(g)
    g, _uv = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=tau)
    return np.asarray(g)


def score(g):
    if not np.isfinite(g).all():
        return dict(soft=np.nan, arg_acc=np.nan)
    j = g.argmax(1)
    soft = COMP[j, cidx].mean()                       # unbiased, uses full composition
    dom = np.asarray(d["spot_clade"]).astype(str)[j]
    return dict(soft=float(soft), arg_acc=float((dom == cell_clade).mean()))


# baselines: assign every cell to the spot maximising its own clade share would be cheating;
# the honest floor is the mean composition a random spot offers for a cell's clade
rand_soft = float(np.mean([COMP[:, k].mean() for k in cidx]))
maj = np.bincount(cidx, minlength=K).argmax()
print(f"[run] floors -- random spot {rand_soft:.4f} soft; "
      f"majority clade {clades[maj]} is {(cidx==maj).mean():.4f} of cells")

GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
rows = []
for tau in [None, 0.1, 0.01]:
    for av in GRID:
        s = score(solve(np.full(K, av), tau))
        rows.append(dict(tau=("balanced" if tau is None else tau), alpha=av, **s))
r = pd.DataFrame(rows)
print(f"\n{'tau':>9}{'alpha':>7}{'soft acc':>10}{'argmax':>9}")
print("-" * 36)
for _, x in r.iterrows():
    print(f"{str(x.tau):>9}{x.alpha:>7.2f}{x.soft:>10.4f}{x.arg_acc:>9.4f}")
print("-" * 36)
for t_, s_ in r.groupby("tau", sort=False):
    a0 = float(s_[s_.alpha == 0].soft.iloc[0]); bi = s_.loc[s_.soft.idxmax()]
    print(f"  tau {str(t_):>8}: alpha=0 {a0:.4f} -> best {bi.soft:.4f} at alpha={bi.alpha} "
          f"({bi.soft-a0:+.4f})")
r.to_csv(HERE / f"petracer_{SEC}_sweep.csv", index=False)
print(f"\nrandom-spot floor {rand_soft:.4f}")
