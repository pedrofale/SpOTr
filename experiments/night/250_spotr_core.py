"""SpOTr core for the paper's studies (decision-deconvolution-plugin, 2026-10-25).  2026-10-25.
A cell-to-spot map gamma replaces a tool's free per-spot abundances; the tool's own objective is the expression term, the masked tree term
(GW on pairs of traced cells) and an entropy term are added:
    minimise over gamma:   E_tool(P(gamma)) + lambda G_Omega(gamma) + eps sum gamma (log gamma - 1),     P(gamma) = gamma^T R
with R the cells' expression representation the tool uses.  Plug-ins:
  destvi   scvi-tools' CondSCVI on the dissociated cells (labels: each cell's clone, 'untraced' for untraced tumour cells, microenvironment
           types; within a label a continuous latent state) and DestVI fitted as published on the spots; R = each cell's decoded expression
           at its own latent, times DestVI's per-gene platform scale beta; E = DestVI's negative binomial of the spot counts with
           rate_m = library_m (kappa_m P_m + v_noise,m eta), DestVI's fitted beta, eta, dispersion logits and noise weights frozen,
           kappa_m = DestVI's total label weight at the spot over the map's mass there; predicted rates floored
  tangram  Tangram's cells-mode objective: E = - mean over genes of the cosine similarity across spots between predicted and observed
           expression (log-normalised, as 73_tangram.py feeds it); R = the cells' log-normalised expression; its density prior
           (rna_count_based) as hard capacities
Capacities: q_m from the tool (DestVI: the spot's library, as its density; Tangram: rna_count_based); the traced cells' capacity q_m f_T t_m
(random tracing; t_m the tool's own tumour share), every other cell the rest (one kind); cells of a kind share its total equally.
v1 (decision-capacities-from-tool, 2026-10-27): every type the tool resolves is its own kind with capacity q_m x the tool's own share --
traced q f_T t, untraced q (1 - f_T) t, each microenvironment type q pi_e (kinds_from_shares, destvi_shares) -- so the map's composition
is the tool's and SpOTr decides only which cells (so which clones) go where; Problem(kind=, caps=).  The two kinds stay the default.
Tree term: 'lineage' cell-level on the traced block; 'clone tree' and 'clones' (partition) at clone level on clone totals; Omega = 1 on traced
pairs (weights w_k optional); target distances scaled to the tumour region (the tool's tumour share).
Solver: generalised conditional gradient (Sinkhorn best response to the gradient, line search over {1, 1/2, ..., 1/1024, 0} on the full
objective); eps = EPS_REL x the doubly-centred spread of the expression gradient at the product map; the tree scale c makes lambda = 1 equate
the spreads of the two gradients on the traced rows; continuation along lambda; the knee (239_'s rule, fit = expression term + entropy).
"""
import os, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EPS_REL, J, IT0, IT1 = 0.02, 40, 1000, 150
ETAS = [2.0 ** -k for k in range(11)] + [0.0]
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def dc_std(A):
    return float((A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()).std())


# ------------------------------------------------------------------------------------------------ tools
def fit_destvi(Xc, labels, Y, seed=0, epochs_sc=300, epochs_st=1500):
    """CondSCVI on the cells (labels: clone / 'untraced' / microenvironment type) and DestVI on the spots, as published"""
    import anndata, scvi, torch, pandas as pd
    scvi.settings.seed = seed
    sc = anndata.AnnData(np.asarray(Xc, np.float32)); sc.obs["label"] = pd.Categorical(labels)
    scvi.model.CondSCVI.setup_anndata(sc, labels_key="label"); msc = scvi.model.CondSCVI(sc, weight_obs=False)
    msc.train(max_epochs=epochs_sc, enable_progress_bar=False)
    st = anndata.AnnData(np.asarray(Y, np.float32)); scvi.model.DestVI.setup_anndata(st); mst = scvi.model.DestVI.from_rna_model(st, msc)
    mst.train(max_epochs=epochs_st, enable_progress_bar=False)
    names = list(msc.adata_manager.get_state_registry("labels").categorical_mapping); code = np.array([names.index(l) for l in labels])
    z = msc.get_latent_representation(); dev = next(msc.module.parameters()).device
    with torch.no_grad():
        cats = [torch.tensor(code[:, None], device=dev)] + ([torch.zeros((len(code), 1), dtype=torch.long, device=dev)] if msc.module.n_batch > 0 else [])
        h = msc.module.decoder(torch.tensor(z, device=dev), *cats); rho = msc.module.px_decoder(h).cpu().numpy().astype(float)
        md = mst.module; beta = torch.exp(md.beta).cpu().numpy().astype(float); eta = torch.nn.functional.softplus(md.eta).cpu().numpy().astype(float)
        px_o = md.px_o.detach().cpu().numpy().astype(float); V = torch.nn.functional.softplus(md.V).cpu().numpy().T.astype(float)   # spots x (labels + 1)
    own = mst.get_proportions(); own = own[names].values if hasattr(own, "values") else own
    return dict(rho=rho, beta=beta, eta=eta, px_o=px_o, V=V, names=names, own=np.asarray(own, float), library=np.asarray(Y, float).sum(1))


def tangram_own(Xc, Y, q, seed=0):
    """Tangram as published (73_tangram.py's settings): its own map, rows summing to 1"""
    m73 = mod("m73", "73_tangram.py"); T = m73.tangram(lognorm(Xc), lognorm(Y), q, seed); return T / np.maximum(T.sum(1, keepdims=True), 1e-300)


def lognorm(A):
    A = np.asarray(A, float); return np.log1p(A / np.maximum(A.sum(1, keepdims=True), 1.0) * 1e4)


def expression_term(tool, fit, Xc, Y):
    """R (cells x genes) and E(P) in JAX for the tool"""
    import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
    if tool == "destvi":
        R = fit["rho"] * fit["beta"][None, :]; Yj = jnp.asarray(np.asarray(Y, float)); lib = jnp.asarray(fit["library"]); eta = jnp.asarray(fit["eta"])
        po = jnp.asarray(fit["px_o"]); vn = jnp.asarray(fit["V"][:, -1]); kap = jnp.asarray(fit["kappa"])
        sp = jax.nn.softplus(po)
        def E(P):
            r = jnp.maximum(lib[:, None] * (kap[:, None] * P + vn[:, None] * eta[None, :]), 1e-8)          # NB total_count, DestVI's form
            return -jnp.sum(jax.scipy.special.gammaln(Yj + r) - jax.scipy.special.gammaln(r) - r * sp[None, :] + Yj * (po - sp)[None, :])
        return R, E
    if tool == "tangram":
        R = lognorm(Xc); Yl = jnp.asarray(lognorm(Y)); Yn = Yl / jnp.maximum(jnp.linalg.norm(Yl, axis=0, keepdims=True), 1e-12)
        def E(P):
            return -jnp.mean(jnp.sum(Yn * P, 0) / jnp.sqrt(jnp.sum(P * P, 0) + 1e-24))      # safe norm: a gene absent from every cell has a zero column (2026-10-28)
        return R, E
    raise ValueError(tool)


# ------------------------------------------------------------------------------------------------ v1 capacities
def kinds_from_shares(cell_kind, share, q):
    """v1 capacities (decision-capacities-from-tool): cell_kind (n,) in 0..K-1, share (K x spots) the tool's own per-spot share of each kind;
    a kind without cells gives its share to the others at that spot; returns (kind per cell, renumbered; capacities kinds x spots, sum 1)"""
    K = share.shape[0]; present = np.bincount(cell_kind, minlength=K) > 0
    s = np.where(present[:, None], np.clip(share, 0.0, None), 0.0); s = s / np.maximum(s.sum(0, keepdims=True), 1e-300)
    caps = (s * (np.asarray(q, float) / np.sum(q))[None, :])[present]; remap = -np.ones(K, int); remap[present] = np.arange(present.sum())
    return remap[cell_kind], caps / caps.sum()


def destvi_shares(fit, traced, tumour, me_type, nT):
    """the kinds traced / untraced / microenvironment type e (cell_kind 0 / 1 / 2 + e) and their per-spot shares from DestVI's own
    proportions: traced f_T t, untraced (1 - f_T) t (random tracing; t = the clone labels' and 'untraced''s share), type e its label's"""
    names = fit["names"]; own = fit["own"] / np.maximum(fit["own"].sum(1, keepdims=True), 1e-300)
    t = own[:, [i for i, l in enumerate(names) if l.startswith("c") or l == "untraced"]].sum(1); fT = traced.sum() / max(tumour.sum(), 1)
    me = [own[:, names.index(f"me{e}")] if f"me{e}" in names else np.zeros(len(t)) for e in range(nT)]
    cell_kind = np.where(traced, 0, np.where(tumour, 1, 2 + np.asarray(me_type)))
    return cell_kind, np.vstack([fT * t, (1.0 - fT) * t] + me)


# ------------------------------------------------------------------------------------------------ the map
class Problem:
    """one map problem: cells, spots, the tool's expression term, capacities, the tree"""

    def __init__(self, R, E, q, t_hat, traced, tumour, D, eps=None, kind=None, caps=None):
        """q: spot capacities; t_hat: the tool's tumour share per spot; traced / tumour: boolean per cell; kind / caps (v1): a kind per cell
        and per-kind capacities (kinds x spots, from kinds_from_shares) in place of the two kinds traced / everything else"""
        import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp; n, M_ = R.shape[0], len(q); self.n, self.M = n, M_
        if kind is None:
            q = np.asarray(q, float) / np.sum(q); fT = traced.sum() / max(tumour.sum(), 1)
            qT = q * np.clip(fT * np.asarray(t_hat, float), 0.0, 1.0); qk = np.vstack([qT, q - qT])       # traced cells; everything else
            kor = np.where(traced, 0, 1); rows = [np.where(kor == k)[0] for k in (0, 1)]
            a = np.where(traced, qk[0].sum() / max(traced.sum(), 1), qk[1].sum() / max((~traced).sum(), 1))
        else:
            kor = np.asarray(kind); qk = np.asarray(caps, float) / np.sum(caps); rows = [np.where(kor == k)[0] for k in range(len(qk))]
            assert all(len(r) for r in rows), "every kind needs cells (kinds_from_shares drops empty ones)"
            a = (qk.sum(1) / np.array([len(r) for r in rows]))[kor]; q = qk.sum(0)
        self.a, self.qk, self.kor, self.rows, self.q = a, qk, kor, rows, q
        self.R, self.Rj, self.E = R, jnp.asarray(R), E; self.D = np.asarray(D, float); self.Dj = jnp.asarray(self.D); self.D2 = self.Dj * self.Dj
        self.G0 = a[:, None] * qk[kor] / np.maximum(qk.sum(1), 1e-300)[kor][:, None]                     # product within each kind
        gE = jax.jit(jax.grad(lambda P: E(P))); self.C0 = np.asarray(self.Rj @ gE(jnp.asarray(self.G0).T @ self.Rj).T)
        self.eps = EPS_REL * dc_std(self.C0) if eps is None else eps

    def tree(self, kind, idx=None, d=None, clone=None, dK=None, w=None, W=None, sigma=None, G_ref=None):
        """the tree term: kind 'cell' (idx: traced rows, d: their target distances) or 'clone' (clone: label per row, -1 off; dK: targets),
        both squared distance matching; or 'coloc' (2026-10-28, Pedro: "You can go ahead and build it"): the co-location term on clone
        mass fields, - sum_kl W_kl C_k^T K_sigma C_l with K_sigma(m, m') = exp(-D^2 / 2 sigma^2) -- clone: label per row (-1 off), W: clone
        affinities (symmetric, W_kk = 1; the identity for a star tree, i.e. clone labels alone), sigma: the neighbourhood / border width
        (um); bounded, no push-apart.  G_ref: the map at which the scale c is calibrated (default the product map G0, as before)"""
        jnp = self.jnp
        if kind == "coloc":
            K = W.shape[0]; cl = jnp.asarray(np.where(clone >= 0, clone, 0)); on = jnp.asarray((clone >= 0).astype(float)); Wj = jnp.asarray(W)
            Ks = jnp.asarray(np.exp(-self.D ** 2 / (2.0 * sigma ** 2)))
            def A_fn(G, c, cl=cl, on=on, Wj=Wj, Ks=Ks):
                Cm = self.jax.ops.segment_sum(G * on[:, None], cl, num_segments=K); A = -c * (Wj @ (Cm @ Ks))
                return A[cl] * on[:, None], (Cm * A).sum()
        elif kind == "cell":
            w_ = np.ones(len(idx)) if w is None else w; Om = (1.0 - np.eye(len(idx))) * np.sqrt(np.outer(w_, w_))
            arrs = [jnp.asarray(Om), jnp.asarray(Om * d), jnp.asarray(Om * d * d)]; ij = jnp.asarray(idx)
            def A_fn(G, c, arrs=arrs, ij=ij):
                Ts = G[ij]; A = -2.0 * (c * arrs[1] @ Ts @ self.Dj) + c * arrs[0] @ Ts @ self.D2 + (c * arrs[2] @ Ts.sum(1))[:, None]
                return jnp.zeros_like(G).at[ij].set(A), (Ts * A).sum()
        else:
            K = dK.shape[0]; cl = jnp.asarray(np.where(clone >= 0, clone, 0)); on = jnp.asarray((clone >= 0).astype(float))
            wk = jnp.asarray(np.ones(K) if w is None else w); dKj = jnp.asarray(dK)
            def A_fn(G, c, cl=cl, on=on, wk=wk, dKj=dKj):
                Cm = self.jax.ops.segment_sum(G * on[:, None], cl, num_segments=K) * wk[:, None]
                A = ((c * dKj ** 2 @ Cm.sum(1))[:, None] - 2.0 * c * dKj @ (Cm @ self.Dj) + c * (Cm.sum(0) @ self.D2)[None, :]) * wk[:, None]
                return A[cl] * on[:, None], (Cm * A).sum()
        if G_ref is None:
            Gc, Cc = self.G0, self.C0
        else:
            Gc = np.asarray(G_ref); Cc = np.asarray(self.Rj @ self.jax.grad(lambda P: self.E(P))(jnp.asarray(Gc).T @ self.Rj).T)
        A0 = np.asarray(A_fn(self.jnp.asarray(Gc), 1.0)[0]); rws = np.where(np.abs(A0).sum(1) > 0)[0]
        c = dc_std(Cc[rws]) / max(dc_std(2.0 * A0[rws]), 1e-300)
        return A_fn, c

    def solve_path(self, A_fn, c, lambdas=LAMBDAS, J_=J, keep=False):
        """continuation along lambda; returns per lambda: fit (E + entropy), tree energy / c, the map (if keep)"""
        jax, jnp = self.jax, self.jnp; from jax.scipy.special import logsumexp
        eps, kor = self.eps, jnp.asarray(self.kor); rows = [jnp.asarray(r) for r in self.rows]; loga = jnp.log(jnp.asarray(self.a)); logq = jnp.log(jnp.maximum(jnp.asarray(self.qk), 1e-300))
        etas = jnp.asarray(ETAS); E = self.E; Rj = self.Rj

        def sink(Cst, f, g, iters):
            def body(_, s):
                f, g = s; f = eps * loga - eps * logsumexp((g[kor] - Cst) / eps, axis=1)
                g = jnp.stack([eps * logq[k] - eps * logsumexp((f[r][:, None] - Cst[r]) / eps, axis=0) for k, r in enumerate(rows)]); return (f, g)
            return jax.lax.fori_loop(0, iters, body, (f, g))
        ent = lambda G: eps * jnp.sum(G * (jnp.log(jnp.maximum(G, 1e-300)) - 1.0))
        gradE = jax.grad(lambda P: E(P))

        def run(G, f, g, lam, cold):
            def cost(G):
                P = G.T @ Rj; A, _ = A_fn(G, c); return Rj @ gradE(P).T + lam * 2.0 * A, P
            if cold:
                f, g = sink(cost(G)[0], f, g, IT0)
            def step(carry, _):
                G, f, g = carry; Cst, P = cost(G); f, g = sink(Cst, f, g, IT1); Gh = jnp.exp((f[:, None] + g[kor] - Cst) / eps); Ph = Gh.T @ Rj
                phi = lambda e: E(P + e * (Ph - P)) + lam * A_fn(G + e * (Gh - G), c)[1] + ent(G + e * (Gh - G))
                vals = jax.lax.map(phi, etas); e = etas[jnp.argmin(vals)]
                return (G + e * (Gh - G), f, g), jnp.stack([vals.min(), e])
            (G, f, g), tr = jax.lax.scan(step, (G, f, g), None, length=J_)
            return G, f, g, tr
        runj = jax.jit(run, static_argnums=(4,))
        st = (jnp.asarray(self.G0), jnp.zeros(self.n), jnp.zeros((len(self.rows), self.M))); out = {}
        for i, lam in enumerate(lambdas):
            G, f, g, tr = runj(*st, float(lam), i == 0); st = (G, f, g); Gn = np.asarray(G)
            P = G.T @ Rj; fit = float(E(P) + ent(G)); te = float(A_fn(G, c)[1]) / c; tr = np.asarray(tr)
            out[lam] = dict(fit=fit, tree=te, last_step=float(tr[-1, 1]), finite=bool(np.isfinite(Gn).all()), G=Gn if keep else None)
        return out


def knee(path):
    L = sorted(path); Ev = np.array([path[l]["fit"] for l in L]); Qv = np.array([path[l]["tree"] for l in L])
    sc = (Ev - Ev[0]) / max(Ev[-1] - Ev[0], 1e-12) + (Qv - Qv[-1]) / max(Qv[0] - Qv[-1], 1e-12)
    return L[int(np.argmin(sc))]
