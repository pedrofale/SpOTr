"""SpOTr v1 against the two-kind version (v0) on worlds with a structured microenvironment.  2026-10-27.
FIXED BEFORE RUNNING.
Pedro, 2026-10-27: "I agree with your suggestion for the method itself, of relying on DestVI's own proportions and SpOTr purely giving
what DestVI cannot. That would at least be v1 of SpOTr"; "I'd rather the simulator didn't place the microenvironment randomly"; on the
order v1 check / lineage cross-validation / rerun S1-S2: "Yes go ahead".
v1 (decision-capacities-from-tool; 250_ kinds_from_shares): every type the tool resolves is its own kind with capacity q x the tool's own
share -- traced q f_T t, untraced q (1 - f_T) t, microenvironment type e q pi_e.  v0: two kinds, traced q f_T t and everything else (S1-S6).
Shares: DestVI from its own proportions (250_ destvi_shares); Tangram from its own map (map_shares: each kind's mass per spot over the
spot's mass; traced / untraced split by f_T as for DestVI, random tracing).
Worlds: 239_'s design (neighbouring region, margin, compact clones, whole region dissociated) with the microenvironment 'structured' (189_,
section 98: macrophages at the edge, T cells in aggregates, endothelium even, the margin's immune cells near the tumour); tree_sd 0.3;
seeds 46, 47.  Lineage tree, cell-level on the traced block; lambda over {0, 0.1, 0.3, 0.5, 1, 2.5} with continuation (260_'s path).
Comparators: DestVI alone, Tangram alone, exact OT, the floor, the global mix.
Readouts (239_ readout / 189_ metrics): per-spot subclone error, the same over 3 x 3 windows, the traced cells' placement, the tumour-share
error, the microenvironment-type error (S6's rule); DestVI's own tumour-share and type errors as in 263_.
Expectations stated before running:
  E113  SpOTr-DestVI v1's microenvironment-type and tumour-share errors equal DestVI's own within 0.01 at every lambda (by construction),
        both worlds
  E114  v1's subclone error at lambda 0.5 is no more than 0.03 above v0's, for both tools in both worlds (4 of 4)
  E115  SpOTr-DestVI v1 at lambda 0.5 has a lower subclone error than DestVI alone and Tangram alone, and SpOTr-Tangram v1 than Tangram
        alone, in both worlds (S1's E91 / E92 on structured worlds)
  read: v0's type error against DestVI's own when the per-spot mix has structure (S6 had it only from sampling inside the tumour);
        SpOTr-Tangram v1's composition against Tangram alone; v1 against v0 at the other lambdas.
One job per world (about 10 minutes), saved to spotr_out/v1_check/<seed>.json with the per-spot clone and type masses of every map, so
the figure needs no GPU; resumable.  Usage: python 267_v1_check.py [SUMMARY]
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "v1_check"
SEEDS, TREE_SD, PATTERN, TOOLS, VERSIONS = [46, 47], 0.3, "structured", ["destvi", "tangram"], ["v0", "v1"]
PATH = [0.0, 0.1, 0.3, 0.5, 1.0, 2.5]; DEFAULT = 0.5


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def build(seed, tree_sd=TREE_SD, pattern=PATTERN):
    m189, m208 = mod("m189", "189_me_sim.py"), mod("m208", "208_neighbour_sim.py")
    NB = m208.simulate_nb(seed, pattern=pattern, margin=1.0, tree_sd=tree_sd)
    X = m208.problem_nb(NB, "all", "single", capture=dict(keep=0.5), dissoc_frac=1.0)
    t_f = float(NB["section"]["sim"]["ntime"][2 * 127 - 1]); Xc = NB["region"]["counts"][X["mapped"]].astype(float)
    return NB, X, Xc, t_f, m189


def map_shares(T, traced, tumour, me_type, nT):
    """as 250_ destvi_shares, from a tool's own cell-to-spot map: each kind's mass per spot over the spot's mass"""
    tot = np.maximum(T.sum(0), 1e-300); t = T[tumour].sum(0) / tot; fT = traced.sum() / max(tumour.sum(), 1)
    me = [T[~tumour & (np.asarray(me_type) == e)].sum(0) / tot for e in range(nT)]
    return np.where(traced, 0, np.where(tumour, 1, 2 + np.asarray(me_type))), np.vstack([fT * t, (1.0 - fT) * t] + me)


def job(seed):
    import warnings; warnings.filterwarnings("ignore")
    core, m239 = mod("core", "250_spotr_core.py"), mod("m239", "239_deconv_fgw.py"); t0 = time.time()
    NB, X, Xc, t_f, m189 = build(seed); n = len(X["cl"]); tum = X["kind"] == 0; tr = tum & X["traced"]; Y = X["Sc"]; Nb = m239.window(X); K, nT = X["K"], X["nT"]
    labels = np.array([f"c{c}" if t else ("untraced" if k == 0 else f"me{e}") for k, c, t, e in zip(X["kind"], X["cl"], tr, X["me_type"])])
    kind6 = np.where(tr, 0, np.where(tum, 1, 2 + X["me_type"])); H6 = np.eye(2 + nT)[kind6]
    mass = lambda G: (np.asarray(G[tr], float).T @ np.eye(K)[X["cl"][tr]])
    out = dict(seed=seed, tree_sd=TREE_SD, pattern=PATTERN, arms={}, sweeps={}, masses={}, types={}, timing={})
    true_t = X["Ncl"].sum(1) / np.maximum(X["ncell"], 1); hm = X["Nme"].sum(1) > 0; Qm = X["Nme"][hm] / X["Nme"][hm].sum(1, keepdims=True)
    out["types"]["truth"] = np.hstack([X["Ncl"].sum(1, keepdims=True), X["Nme"]]).tolist()
    fit = core.fit_destvi(Xc, labels, Y, seed=0); names, own = fit["names"], fit["own"]; q = fit["library"] / fit["library"].sum()
    Pd = np.zeros((len(Y), K))
    for k in range(K):
        if f"c{k}" in names:
            Pd[:, k] = own[:, names.index(f"c{k}")] * fit["library"]
    out["arms"]["DestVI alone"] = m239.readout(None, X, Nb, m189, P_clone=Pd); out["masses"]["DestVI alone"] = Pd.tolist()
    t_d = own[:, [names.index(l) for l in names if l.startswith("c") or l == "untraced"]].sum(1)
    Pm = np.zeros((len(Y), nT))
    for e in range(nT):
        if f"me{e}" in names:
            Pm[:, e] = own[:, names.index(f"me{e}")]
    out["types"]["DestVI alone"] = np.hstack([t_d[:, None], Pm]).tolist(); Pm = Pm / np.maximum(Pm.sum(1, keepdims=True), 1e-300)
    out["arms"]["DestVI alone"].update(tumour_share_err=float(np.abs(t_d - true_t).mean()), me_type_tv=float(0.5 * np.abs(Pm[hm] - Qm).sum(1).mean()))
    Tt = core.tangram_own(Xc, Y, q, seed=0); out["arms"]["Tangram alone"] = m239.readout(Tt / n, X, Nb, m189); out["masses"]["Tangram alone"] = mass(Tt).tolist()
    out["types"]["Tangram alone"] = (Tt.T @ H6).tolist(); t_t = Tt[tum].sum(0) / np.maximum(Tt.sum(0), 1e-300)
    import ot
    To = ot.emd(np.full(n, 1.0 / n), X["q"], X["M"]); out["arms"]["exact OT"] = m239.readout(To, X, Nb, m189); out["masses"]["exact OT"] = mass(To).tolist()
    Tf = mod("m89", "89_growth_example_figure.py").oracle(X); out["arms"]["floor"] = m239.readout(Tf, X, Nb, m189); out["masses"]["floor"] = mass(Tf).tolist()
    gm = np.bincount(X["cl"][tr], minlength=K).astype(float); out["arms"]["global mix"] = m239.readout(None, X, Nb, m189, P_clone=np.outer(X["Ncl"].sum(1) > 0, gm))
    out["timing"]["tools"] = time.time() - t0
    ti, dt = m239.tree_dists(X, NB, t_f, q * t_d)
    for tool in TOOLS:
        if tool == "destvi":
            fit["kappa"] = fit["V"][:, :-1].sum(1) / q; R, E = core.expression_term("destvi", fit, Xc, Y); t_hat = t_d
            ck, sh = core.destvi_shares(fit, tr, tum, X["me_type"], nT)
        else:
            R, E = core.expression_term("tangram", None, Xc, Y); t_hat = t_t; ck, sh = map_shares(Tt, tr, tum, X["me_type"], nT)
        for ver in VERSIONS:
            if ver == "v0":
                prob = core.Problem(R, E, q, t_hat, tr, tum, X["D"])
            else:
                kk, caps = core.kinds_from_shares(ck, sh, q); prob = core.Problem(R, E, q, t_hat, tr, tum, X["D"], kind=kk, caps=caps)
            A_fn, c = prob.tree("cell", idx=ti, d=dt["lineage"]); path = prob.solve_path(A_fn, c, lambdas=PATH, keep=True); sw = {}
            for lam, v in path.items():
                r = dict(fit=v["fit"], tree=v["tree"], last_step=v["last_step"], finite=v["finite"])
                if v["finite"]:
                    r.update(m239.readout(v["G"], X, Nb, m189)); out["masses"][f"{tool}|{ver}|{lam}"] = mass(v["G"]).tolist()
                    out["types"][f"{tool}|{ver}|{lam}"] = (v["G"].T @ H6).tolist()
                sw[str(lam)] = r
            out["sweeps"][f"{tool}|{ver}"] = sw; out["arms"][f"SpOTr-{tool} {ver}"] = dict(sw[str(DEFAULT)], lam=DEFAULT)
            out["arms"][f"SpOTr-{tool} {ver} lambda 0"] = sw["0.0"]; out["timing"][f"{tool} {ver}"] = time.time() - t0
            print(f"    seed {seed} {tool} {ver}: subclone error at 0.5 {sw[str(DEFAULT)].get('clone_tv', np.nan):.3f}, type error {sw[str(DEFAULT)].get('me_type_tv', np.nan):.3f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    out["timing"]["total"] = time.time() - t0
    return out


def summary():
    rs = [json.load(open(f)) for f in sorted(OUT.glob("*.json"))]
    keys = ["clone_tv", "clone_tv_win", "traced_err", "tumour_share_err", "me_type_tv"]
    for r in rs:
        print(f"######## seed {r['seed']} ({r['pattern']}, tree_sd {r['tree_sd']}): " + " / ".join(keys))
        for a in ["DestVI alone", "Tangram alone", "exact OT", "SpOTr-destvi v0", "SpOTr-destvi v1", "SpOTr-tangram v0", "SpOTr-tangram v1",
                  "SpOTr-destvi v1 lambda 0", "SpOTr-tangram v1 lambda 0", "global mix", "floor"]:
            print(f"   {a:26s}: " + " / ".join(f"{r['arms'][a].get(k) if r['arms'][a].get(k) is not None else np.nan:7.3f}" for k in keys))
        for tool in TOOLS:
            print(f"   {tool} subclone error over lambda {PATH}: " + " | ".join(f"{ver} " + " ".join(f"{r['sweeps'][f'{tool}|{ver}'][str(l)].get('clone_tv', np.nan):.3f}" for l in PATH) for ver in VERSIONS))
            print(f"   {tool} type error over lambda:           " + " | ".join(f"{ver} " + " ".join(f"{r['sweeps'][f'{tool}|{ver}'][str(l)].get('me_type_tv', np.nan):.3f}" for l in PATH) for ver in VERSIONS))
    e113 = [all(abs(r["sweeps"]["destvi|v1"][str(l)][k] - r["arms"]["DestVI alone"][k]) <= 0.01 for l in PATH for k in ["me_type_tv", "tumour_share_err"]) for r in rs]
    e114 = [r["arms"][f"SpOTr-{t} v1"]["clone_tv"] - r["arms"][f"SpOTr-{t} v0"]["clone_tv"] <= 0.03 for r in rs for t in TOOLS]
    e115 = [r["arms"]["SpOTr-destvi v1"]["clone_tv"] < min(r["arms"]["DestVI alone"]["clone_tv"], r["arms"]["Tangram alone"]["clone_tv"]) and
            r["arms"]["SpOTr-tangram v1"]["clone_tv"] < r["arms"]["Tangram alone"]["clone_tv"] for r in rs]
    print(f"E113: {sum(e113)}/{len(e113)}   E114: {sum(e114)}/{len(e114)}   E115: {sum(e115)}/{len(e115)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "SUMMARY":
        summary(); sys.exit()
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for s in SEEDS:
        f = OUT / f"{s}.json"
        if f.exists():
            continue
        r = job(s); json.dump(r, open(f, "w"), default=float); print(f"seed {s}: done in {r['timing']['total']:.0f}s ({time.time() - t0:.0f}s total)", flush=True)
    summary()
