"""Verification for the clade-aware (Omega/theta) exact FGW in spotr.perclade_exact."""
import sys, numpy as np, ot
sys.path.insert(0, "/home/pedro/projects/spotr/phylospatial_mapping/src")
import spotr.perclade_exact as pe

rng = np.random.default_rng(0)
n, m = 60, 40
X1, X2 = rng.normal(size=(n, 8)), rng.normal(size=(m, 8))
M = ot.dist(X1, X2); M = M / M.max()
Z1, Z2 = rng.normal(size=(n, 3)), rng.normal(size=(m, 2))
C1 = ot.dist(Z1, Z1); C1 = C1 / C1.mean()
C2 = ot.dist(Z2, Z2); C2 = C2 / C2.mean()
p, q = np.ones(n)/n, np.ones(m)/m
constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, "square_loss")
T0 = np.outer(p, q); Tr = ot.emd(p, q, M)
lab = rng.integers(0, 4, n)                       # 4 clades
blocks = pe._blocks(lab)

print("1) UNMASKED path unchanged: uniform alpha == POT's FGW gradient\n")
print(f"{'alpha':>7}{'max |ours-POT|':>18}{'rel':>12}")
print("-" * 37)
for a in [0.0, 0.3, 0.5, 0.9, 1.0]:
    ours = pe.cost(M, C1, C2, T0, np.full(n, a))
    pot = (1-a)*M + a*ot.gromov.gwggrad(constC, hC1, hC2, T0)
    d = np.abs(ours-pot).max()
    print(f"{a:>7.2f}{d:>18.3e}{d/max(np.abs(pot).max(),1e-12):>12.3e}")

print("\n2) ONE clade + theta=1: Om^w is all-ones, Om^x empty -> must equal POT again")
for a in [0.3, 0.75]:
    ours = pe.cost(M, C1, C2, Tr, np.full(n, a), theta=np.ones(n), blocks=[np.arange(n)])
    pot = (1-a)*M + a*ot.gromov.gwggrad(constC, hC1, hC2, Tr)
    print(f"   alpha {a}: max |diff| = {np.abs(ours-pot).max():.3e}")

print("\n3) theta=0.5 must give exactly HALF the unmasked structural gradient,")
print("   for ANY clade partition (because L^x + L^w = L). Partition-independence check.")
al = rng.uniform(.15, .9, n)
gu = pe.grad_gw(C1, C2, Tr, al)
for tag, lb in [("4 clades", lab), ("2 clades", (lab < 2).astype(int)),
                ("random 7", rng.integers(0, 7, n))]:
    gm = pe.grad_gw(C1, C2, Tr, al, theta=np.full(n, .5), blocks=pe._blocks(lb))
    print(f"   {tag:>9}: max |grad(theta=.5) - grad_unmasked/2| = {np.abs(gm - gu/2).max():.3e}")

print("\n4) The paper's cost is a gradient iff alpha_t (1-theta_t) is constant.")
al_k = np.array([0.30, 0.50, 0.70, 0.90])                    # per-clade alpha
c = 0.20                                                      # shared cross weight
th_k = 1.0 - c / al_k
al_n, th_n = al_k[lab], th_k[lab]
d_ok = np.abs(pe.cost(M, C1, C2, Tr, al_n, th_n, blocks)
              - pe.cost_paper(M, C1, C2, Tr, al_n, th_n, blocks)).max()
th_bad = np.array([0.2, 0.5, 0.8, 0.4])[lab]                  # v_t not constant
d_bad = np.abs(pe.cost(M, C1, C2, Tr, al_n, th_bad, blocks)
               - pe.cost_paper(M, C1, C2, Tr, al_n, th_bad, blocks)).max()
print(f"   alpha_t(1-theta_t) = {c} for every clade : max |paper - gradient| = {d_ok:.3e}")
print(f"   alpha_t(1-theta_t) varying               : max |paper - gradient| = {d_bad:.3e}")
print(f"   theta_t = 1 for every clade (no cross term), alpha free:")
d_th1 = np.abs(pe.cost(M, C1, C2, Tr, al_n, np.ones(n), blocks)
               - pe.cost_paper(M, C1, C2, Tr, al_n, np.ones(n), blocks)).max()
print(f"                                              max |paper - gradient| = {d_th1:.3e}")

print("\n5) Descent with per-clade alpha at theta=1 (the fully separable regime)")
T, log = pe.perclade_fgw(M, C1, C2, p, q, al_n, theta=np.ones(n), labels=lab,
                         max_iter=300, tol=1e-12)
h = np.array(log["objective"])
print(f"   {log['n_iter']} iters, F {h[0]:.6f} -> {h[-1]:.6f}, "
      f"monotone: {bool(np.all(np.diff(h) <= 1e-12))}")

print("\n6) Does theta=1 actually decouple the clades? Change ONE clade's alpha and see")
print("   whether the other clades' rows of the cost move at all.")
for th_tag, th_v in [("theta=1 (no cross term)", np.ones(n)), ("theta=0.5 (cross term on)", np.full(n, .5))]:
    a2 = al_n.copy(); a2[lab == 0] = 0.05
    G1 = pe.cost(M, C1, C2, Tr, al_n, th_v, blocks)
    G2 = pe.cost(M, C1, C2, Tr, a2, th_v, blocks)
    off = np.abs(G1 - G2)[lab != 0].max()
    print(f"   {th_tag:<26} max change on OTHER clades' rows = {off:.3e}")
