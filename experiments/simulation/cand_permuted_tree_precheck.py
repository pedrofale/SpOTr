"""Zero-solve pre-check demanded by the review of candidate `permuted-tree-excess-agreement`.

Review's closed form for the excess-agreement statistic under a linear compromise
    mu^v = (1-a) x_e^v + a x_t ,  x_e^v = m_e + n_v
    u = (1-a)^2 V_e + a^2 V_t ,  x = 2a(1-a)c ,  N = (1-a)^2 V_n
    A_T = (u+x)/(u+x+N) ,  A_P = u/(u+N) ,  S = A_T - A_P = x*N/[(u+x+N)(u+N)]

FALSIFIER (review): if argmax_a S varies by < 0.05 across c at fixed variance ratios while
shifting substantially with V_t/V_e, the peak encodes cost normalisation, not tree quality.
"""
import numpy as np, itertools, pandas as pd

A = np.linspace(1e-4, 1 - 1e-4, 20001)

def S_of_a(a, Ve, Vt, Vn, c):
    u = (1 - a) ** 2 * Ve + a ** 2 * Vt
    x = 2 * a * (1 - a) * c
    N = (1 - a) ** 2 * Vn
    return x * N / ((u + x + N) * (u + N))

rows = []
for Vt_r, Vn_r in itertools.product([0.25, 1.0, 4.0], [0.25, 1.0, 4.0]):
    for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = S_of_a(A, 1.0, Vt_r, Vn_r, c)
        rows.append(dict(Vt_over_Ve=Vt_r, Vn_over_Ve=Vn_r, c=c,
                         argmax=float(A[s.argmax()]), Smax=float(s.max())))
r = pd.DataFrame(rows)
print(r.pivot_table(index=["Vt_over_Ve", "Vn_over_Ve"], columns="c",
                    values="argmax").round(3).to_string())
print()
spread_c = r.groupby(["Vt_over_Ve", "Vn_over_Ve"]).argmax.agg(lambda v: v.max() - v.min())
print("range of argmax S across c (0.1..0.9) at fixed variance ratios:")
print(spread_c.round(4).to_string())
print(f"\nMAX range across c, over all 9 variance-ratio cells : {spread_c.max():.4f}")
print(f"range of argmax S across Vt/Ve at c=0.5, Vn/Ve=1     : "
      f"{r[(r.c==0.5)&(r.Vn_over_Ve==1.0)].argmax.max()-r[(r.c==0.5)&(r.Vn_over_Ve==1.0)].argmax.min():.4f}")
print(f"range of argmax S across Vn/Ve at c=0.5, Vt/Ve=1     : "
      f"{r[(r.c==0.5)&(r.Vt_over_Ve==1.0)].argmax.max()-r[(r.c==0.5)&(r.Vt_over_Ve==1.0)].argmax.min():.4f}")
print(f"\nheight Smax range across c at Ve=Vt=Vn=1             : "
      f"{r[(r.Vt_over_Ve==1)&(r.Vn_over_Ve==1)].Smax.min():.4f} .. "
      f"{r[(r.Vt_over_Ve==1)&(r.Vn_over_Ve==1)].Smax.max():.4f}")
ok = spread_c.max() >= 0.05
print(f"\nPRE-CHECK: {'SURVIVES' if ok else 'FALSIFIED'} "
      f"(criterion: argmax must move >= 0.05 with c)")
r.to_csv("/home/pedro/projects/spotr/phylospatial_mapping/experiments/simulation/cand_permuted_tree_precheck.csv", index=False)
