"""54_ continued: r(a)^2 = L^2 + c^2 r_pack(a)^2 -- compact packing plus a constant local mixing length L.
Fitted by least squares on r^2 over age bins (pair-count weighted), resolved ages and all ages."""
from pathlib import Path
import numpy as np
src = (Path(__file__).resolve().parent / "54_packing_law.py").read_text()
src = src.replace('print(', 'P.append(') .replace('P.append(f"\\n{sec}', 'P.append(f"\\n{sec}')
G = {"__file__": str(Path(__file__).resolve().parent / "54_packing_law.py"), "P": []}
# re-run 54_ silently but keep its per-section arrays
code = src.split('for sec in')[0]
exec(code, G)
for sec in ["M3-4", "M3-1", "M3-2"]:
    ns = dict(G); ns["P"] = []
    exec("for sec in [%r]:" % sec + src.split('for sec in ["M3-4", "M3-1", "M3-2"]:')[1], ns)
    M, pack, sat = ns["M"], ns["pack"], ns["sat"]
    for label, m in [("resolved", ~sat), ("all ages", np.ones(len(M), bool))]:
        X = np.c_[np.ones(m.sum()), pack[m] ** 2]; y = M[m, 1] ** 2; w = np.sqrt(M[m, 2])
        coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        L, c = np.sqrt(max(coef[0], 0)), np.sqrt(max(coef[1], 0))
        fit = np.sqrt(np.maximum(coef[0] + coef[1] * pack[m] ** 2, 1e-9))
        rel = np.abs(fit - M[m, 1]) / M[m, 1]
        r2 = 1 - np.sum(w**2 * (np.log(M[m,1]) - np.log(fit))**2) / np.sum(w**2 * (np.log(M[m,1]) - np.average(np.log(M[m,1]), weights=w**2))**2)
        print(f"{sec} {label:9s}: mixing length L = {L:6.1f} um, packing factor c = {c:.2f}, "
              f"median |rel err| {np.median(rel):.2f}, max {rel.max():.2f}, weighted R2(log) {r2:.2f}")

print("\nunweighted fit on log r over age bins (each bin counts once), grid search:")
Ls = np.linspace(0, 600, 121); cs = np.linspace(0.2, 3.0, 141)
for sec in ["M3-4", "M3-1", "M3-2"]:
    ns = dict(G); ns["P"] = []
    exec("for sec in [%r]:" % sec + src.split('for sec in ["M3-4", "M3-1", "M3-2"]:')[1], ns)
    M, pack = ns["M"], ns["pack"]
    lr = np.log(M[:, 1])
    best = min(((np.sum((lr - 0.5*np.log(L*L + c*c*pack**2 + 1e-9))**2), L, c) for L in Ls for c in cs))
    sse, L, c = best
    fit = np.sqrt(L*L + c*c*pack**2)
    r2 = 1 - sse / np.sum((lr - lr.mean())**2)
    # compare: packing alone (c only) and a flat curve
    sse_p = min(np.sum((lr - np.log(c*pack))**2) for c in cs)
    print(f"{sec}: L = {L:5.0f} um, c = {c:.2f} | R2(log) packing+mixing {r2:.2f}, packing alone "
          f"{1 - sse_p/np.sum((lr-lr.mean())**2):.2f} | median |rel err| {np.median(np.abs(fit/M[:,1]-1)):.2f}"
          f" | predicted dynamic range {fit.max()/fit.min():.2f}x vs measured {M[:,1].max()/M[:,1].min():.2f}x")
