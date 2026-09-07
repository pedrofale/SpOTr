"""Does an ORACLE per-clade alpha transfer between serial sections?

The 3 M3 sections are disjoint cell sets sharing the SAME 18 clade labels, so alpha_k
carries over directly -- no rule-learning needed (NOTES: "VALID -- serial sections").

This removes BOTH blockers at once. Estimability: all clades populated. Objective: none
used -- alpha is fitted against TRUE accuracy on the source section. If per-clade alpha is
a property of the CLADE, the fitted vector must help on a different section of the same
tumour. If it is a property of the clade IN THAT TISSUE REGION, it will not.

Prediction from petracer_perclade_replication.csv: per-clade coherence correlates
r=+0.956 between M3-1/M3-2 (composition TVD 0.113, same tissue block) and r~0.00 with
M3-4 (TVD 0.39-0.45). So M3-1<->M3-2 should transfer and anything involving M3-4 should not.
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J, TAU = 0.02, 50, 3, 0.01
SECS = ["M3-1", "M3-2", "M3-4"]
GRID_G = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
GRID_K = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
rng = np.random.default_rng(0)


class Sec:
    def __init__(self, name):
        d = np.load(HERE / f"petracer_{name}.npz", allow_pickle=True)
        self.name = name
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
        # random-spot floor: expected soft acc if cells placed uniformly at random
        self.floor = float(self.COMP.mean(0)[self.cidx].mean())

    def solve(self, ak):
        Cf, C_T, C_S, a, b, omega, sF, sL = self._p
        def build(g):
            return build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)), Cf,
                                                C_T, C_S, a, b, g, omega, sF, sL,
                                                equalize="measured")
        g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
        return np.asarray(g)

    def score(self, g, cols=None):
        """Soft accuracy: fraction of the assigned spot's members sharing the cell's clade."""
        if not np.isfinite(g).all(): return np.nan
        cols = np.arange(self.m) if cols is None else cols
        j = cols[np.asarray(g)[:, cols].argmax(1)]
        return float(self.COMP[j, self.cidx].mean())

    def acc(self, ak, cols=None):
        return self.score(self.solve(ak), cols)


D = {s: Sec(s) for s in SECS}
for s in SECS:
    print(f"[load] {s}: {D[s].n} cells x {D[s].m} spots, {D[s].K} clades, "
          f"random floor {D[s].floor:.4f}", flush=True)

# ---- 1. global alpha sweep, oracle-best global per section (the bar to beat) ----
print("\n" + "=" * 78 + "\nGLOBAL ALPHA SWEEP (true soft accuracy)", flush=True)
BG = {}
for s in SECS:
    sc = {v: D[s].acc(np.full(D[s].K, v)) for v in GRID_G}
    BG[s] = max(sc, key=lambda k: sc[k])
    print(f"  {s}: " + " ".join(f"{v}:{sc[v]:.4f}" for v in GRID_G) +
          f"   -> best global alpha={BG[s]} ({sc[BG[s]]:.4f}), alpha=0 {sc[0.0]:.4f}", flush=True)
    D[s].best_global, D[s].bg_acc, D[s].a0_acc = BG[s], sc[BG[s]], sc[0.0]

# ---- 2. oracle per-clade alpha per section (coordinate ascent on TRUE accuracy) ----
print("\n" + "=" * 78 + "\nORACLE PER-CLADE ALPHA (coordinate ascent on true accuracy)", flush=True)
ORACLE = {}
for s in SECS:
    S_ = D[s]
    ak = np.full(S_.K, float(BG[s]), np.float32)
    cur = S_.acc(ak)
    for sweep in range(2):
        for k in range(S_.K):
            best, bv = cur, ak[k]
            for v in GRID_K:
                if v == ak[k]: continue
                t = ak.copy(); t[k] = v
                a_ = S_.acc(t)
                if a_ > best: best, bv = a_, v
            ak[k], cur = bv, best
        print(f"  {s} sweep {sweep}: acc {cur:.4f}  alpha mean {ak.mean():.3f} "
              f"range {ak.min():.2f}-{ak.max():.2f} sd {ak.std():.3f}", flush=True)
    ORACLE[s] = ak
    S_.oracle_acc = cur
    pd.DataFrame(dict(clade=S_.clades, alpha=ak)).to_csv(
        HERE / f"petracer_{s}_oracle_alpha.csv", index=False)

# ---- 3. transfer matrix ----
print("\n" + "=" * 78 + "\nTRANSFER MATRIX: oracle alpha from SOURCE applied to TARGET", flush=True)
rows = []
for src, tgt in itertools.product(SECS, SECS):
    ak = ORACLE[src]
    a_t = D[tgt].acc(ak)
    a_u = D[tgt].acc(np.full(D[tgt].K, float(ak.mean())))
    sh = [D[tgt].acc(rng.permutation(ak)) for _ in range(10)]
    rows.append(dict(source=src, target=tgt, transferred=a_t, uniform_at_mean=a_u,
                     shuffled=float(np.nanmean(sh)), shuffled_sd=float(np.nanstd(sh)),
                     best_global=D[tgt].bg_acc, alpha0=D[tgt].a0_acc,
                     oracle_on_target=D[tgt].oracle_acc, is_diag=int(src == tgt)))
    print(f"  {src} -> {tgt}: transferred {a_t:.4f} | uniform@mean {a_u:.4f} | "
          f"shuffled {np.nanmean(sh):.4f}+-{np.nanstd(sh):.4f} | best global {D[tgt].bg_acc:.4f} | "
          f"oracle {D[tgt].oracle_acc:.4f}", flush=True)

r = pd.DataFrame(rows)
r.to_csv(HERE / "petracer_transfer_matrix.csv", index=False)
print("\n" + "=" * 78 + "\nSUMMARY  (gain of transferred alpha over the target's own best global)")
print(f"{'source -> target':<20}{'transf-global':>15}{'transf-uniform':>16}{'transf-shuffled':>17}")
print("-" * 68)
for _, x in r.iterrows():
    tag = "  (oracle, upper bound)" if x.is_diag else ""
    print(f"{x.source+' -> '+x.target:<20}{x.transferred-x.best_global:>+15.4f}"
          f"{x.transferred-x.uniform_at_mean:>+16.4f}{x.transferred-x.shuffled:>+17.4f}{tag}")
off = r[r.is_diag == 0]
same = off[off.apply(lambda x: {x.source, x.target} <= {"M3-1", "M3-2"}, axis=1)]
diff = off[~off.index.isin(same.index)]
print("-" * 68)
print(f"same tissue block (M3-1<->M3-2), n={len(same)}: "
      f"mean transf-global {(same.transferred-same.best_global).mean():+.4f}, "
      f"transf-uniform {(same.transferred-same.uniform_at_mean).mean():+.4f}")
print(f"across tissue blocks (involving M3-4), n={len(diff)}: "
      f"mean transf-global {(diff.transferred-diff.best_global).mean():+.4f}, "
      f"transf-uniform {(diff.transferred-diff.uniform_at_mean).mean():+.4f}")
print("\nwrote petracer_transfer_matrix.csv")
