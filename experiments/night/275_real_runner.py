"""SpOTr v1 on a real mapping problem: both plug-ins, lambda by the anchored lineage cross-validation, the comparators.  2026-10-28.
The procedure of 270_ (fold fits, full-data path) and 272_ (the anchored score, its argmax) on a problem built from real data
(decision-real-data-plan); no simulator objects.  A problem P is a dict:
  X       189_-style readout dict: kind (0 tumour, 1 microenvironment), traced, cl (clade index for traced cells, K - 1 = traced
          without a clade), me_type, K, nT, spots, pitch, D (spot distances, um), q (true cells per spot / total), M (expression cost
          for exact OT), Ncl (true traced cells per spot and clade), Nme (true microenvironment cells per spot and type), Ntum
          (true tumour cells per spot), ncell (true cells per spot), xy_true (the dissociated cells' true positions in the spots'
          frame, NaN when unknown)
  Xc, Y   the dissociated cells' and the spots' counts; labels: DestVI labels ('c<k>' traced, 'untraced', 'me<e>')
  d_raw   tree distances between the traced dissociated cells, in the order of np.where(traced & tumour)
  tree_w  optional (2026-10-28, 281_): a weight per traced cell (same order); 0 leaves a cell out of the tree term and of the
          cross-validation's pairs (e.g. traced cells without a clade under a clone-level tree); None = all 1, as before
  seed    for the folds (stream 15e6 + seed)
Readouts (239_ readout): per-spot subclone error over spots with a true traced cell, the same over 3 x 3 windows, the traced cells'
placement (when xy_true is known), the microenvironment-type error; the tumour-share error against Ntum (189_'s own rule assumes
every tumour cell has a clone).  Fold maps of the traced rows are saved (rows normalised, float16).
Usage: imported (run).
"""
import os, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TOOLS, PATH, N_FOLDS = ["destvi", "tangram"], [0.0, 0.1, 0.3, 0.5, 1.0, 2.5], 4


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def anchored(maps, fold, d, D, on=None):
    """272_'s score: folds x lambdas, concordance of d_ij with G0_i D G^lambda_j^T (held-out i at lambda 0, training j); on: cells kept"""
    S = np.zeros((N_FOLDS, len(PATH))); on = np.ones(len(fold), bool) if on is None else on
    for k in range(N_FOLDS):
        hid, trn = (fold == k) & on, (fold != k) & on; A = maps[f"fold{k}|0.0"].astype(float)[hid] @ D; dd = d[np.ix_(hid, trn)].ravel()
        for j, l in enumerate(PATH):
            S[k, j] = np.corrcoef(dd, (A @ maps[f"fold{k}|{l}"].astype(float)[trn].T).ravel())[0, 1]
    return S


def run(P, f, maps_dir):
    import warnings; warnings.filterwarnings("ignore")
    core, m239, m267, m189 = mod("core", "250_spotr_core.py"), mod("m239", "239_deconv_fgw.py"), mod("m267", "267_v1_check.py"), mod("m189", "189_me_sim.py")
    m270 = mod("m270", "270_lineage_cv_concordance.py"); t0 = time.time()
    X, Xc, Y, labels = P["X"], P["Xc"], P["Y"], P["labels"]; n = len(X["cl"]); tum = X["kind"] == 0; tr = tum & X["traced"]; K, nT = X["K"], X["nT"]
    D = np.asarray(X["D"], float); Nb = m239.window(X); has_xy = bool(np.isfinite(X["xy_true"]).all())
    mass = lambda G: (np.asarray(G[tr], float).T @ np.eye(K)[X["cl"][tr]])
    def readout(T=None, P_clone=None):
        r = m239.readout(T, X, Nb, m189, P_clone=P_clone)
        if T is not None:
            share = np.asarray(T)[tum].sum(0) / np.maximum(np.asarray(T).sum(0), 1e-300); r["tumour_share_err"] = float(np.abs(share - X["Ntum"] / np.maximum(X["ncell"], 1)).mean())
            if not has_xy:
                r["traced_err"] = None
        return r
    out = json.load(open(f)) if f.exists() else dict(meta=P.get("meta", {}), arms={}, cv={}, masses={}, timing={})
    fit = core.fit_destvi(Xc, labels, Y, seed=0); names, own = fit["names"], fit["own"]; q = fit["library"] / fit["library"].sum()
    t_d = own[:, [names.index(l) for l in names if l.startswith("c") or l == "untraced"]].sum(1); Tt = core.tangram_own(Xc, Y, q, seed=0)
    if not out["arms"]:
        Pd = np.zeros((len(Y), K))
        for k in range(K):
            if f"c{k}" in names:
                Pd[:, k] = own[:, names.index(f"c{k}")] * fit["library"]
        out["arms"]["DestVI alone"] = readout(P_clone=Pd); out["masses"]["DestVI alone"] = Pd.tolist()
        out["arms"]["Tangram alone"] = readout(Tt / n); out["masses"]["Tangram alone"] = mass(Tt).tolist()
        import ot
        To = ot.emd(np.full(n, 1.0 / n), X["q"], X["M"]); out["arms"]["exact OT"] = readout(To); out["masses"]["exact OT"] = mass(To).tolist()
        if has_xy:
            Tf = mod("m89", "89_growth_example_figure.py").oracle(X); out["arms"]["floor"] = readout(Tf); out["masses"]["floor"] = mass(Tf).tolist()
        gm = np.bincount(X["cl"][tr], minlength=K).astype(float); out["arms"]["global mix"] = readout(P_clone=np.outer(X["Ncl"].sum(1) > 0, gm))
        out["timing"]["tools"] = time.time() - t0; json.dump(out, open(f, "w"), default=float)
    ti = np.where(tr)[0]; d_raw = np.asarray(P["d_raw"], float); tw = P.get("tree_w"); on = np.ones(len(ti), bool) if tw is None else np.asarray(tw) > 0
    off = ~np.eye(len(ti), dtype=bool) & on[:, None] & on[None, :]
    w = q * t_d; w = w / w.sum(); d = d_raw * float(w @ D @ w) / d_raw[off].mean(); np.fill_diagonal(d, 0.0)
    fold = np.random.default_rng(15 * 10 ** 6 + P["seed"]).integers(0, N_FOLDS, len(ti)); maps_dir.mkdir(parents=True, exist_ok=True)
    for tool in TOOLS:
        if tool in out["cv"]:
            continue
        if tool == "destvi":
            fit["kappa"] = fit["V"][:, :-1].sum(1) / q; R, E = core.expression_term("destvi", fit, Xc, Y); t_hat = t_d
            ck, sh = core.destvi_shares(fit, tr, tum, X["me_type"], nT)
        else:
            R, E = core.expression_term("tangram", None, Xc, Y); t_hat = Tt[tum].sum(0) / np.maximum(Tt.sum(0), 1e-300); ck, sh = m267.map_shares(Tt, tr, tum, X["me_type"], nT)
        kk, caps = core.kinds_from_shares(ck, sh, q); prob = core.Problem(R, E, q, t_hat, tr, tum, D, kind=kk, caps=caps); maps = {}
        for k in range(N_FOLDS):
            hid = fold == k; A_fn, c = prob.tree("cell", idx=ti[~hid], d=d[np.ix_(~hid, ~hid)], w=None if tw is None else np.asarray(tw, float)[~hid])
            path = prob.solve_path(A_fn, c, lambdas=PATH, keep=True)
            for l in PATH:
                Gt = np.asarray(path[l]["G"], float)[ti]; maps[f"fold{k}|{l}"] = (Gt / np.maximum(Gt.sum(1, keepdims=True), 1e-300)).astype(np.float16)
        A_fn, c = prob.tree("cell", idx=ti, d=d, w=None if tw is None else np.asarray(tw, float)); path = prob.solve_path(A_fn, c, lambdas=PATH, keep=True); full = {}
        for lam, v in path.items():
            full[str(lam)] = dict(fit=v["fit"], tree=v["tree"], finite=v["finite"], **(readout(v["G"]) if v["finite"] else {}))
            if v["finite"]:
                out["masses"][f"{tool}|{lam}"] = mass(v["G"]).tolist()
        np.savez_compressed(maps_dir / f"{f.stem}_{tool}.npz", ti=ti, fold=fold, d=d.astype(np.float32), **maps)
        S = anchored(maps, fold, d, D, on); am, se1 = m270.rules(S); best = min(full, key=lambda l: full[l].get("clone_tv", np.inf))
        out["cv"][tool] = dict(per_fold=S.tolist(), argmax=am, one_se=se1, truth_best=float(best), full=full); out["timing"][tool] = time.time() - t0
        json.dump(out, open(f, "w"), default=float)
        alone = out["arms"]["DestVI alone" if tool == "destvi" else "Tangram alone"]["clone_tv"]
        print(f"  {f.stem} {tool}: anchored " + " ".join(f"{x:+.3f}" for x in S.mean(0)) + f" -> lambda {am:g}: subclone error {full[str(am)]['clone_tv']:.3f} "
              f"(lambda 0 {full['0.0']['clone_tv']:.3f}, 0.5 {full['0.5']['clone_tv']:.3f}, truth-best {best} {full[best]['clone_tv']:.3f}, alone {alone:.3f})  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; out["timing"]["total"] = time.time() - t0; json.dump(out, open(f, "w"), default=float)
    return out
