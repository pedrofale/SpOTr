"""Per-clade weights that match the true dispersal of the clones, on the target-scenario benchmark (Pedro, 2026-11-03: "check if we can show
that when we give SpOTr per clade weights that match the true clone dispersions, that the mapping becomes better too? That way we could still
keep the clade aware arguments"); replaces 467_ (alpha 0.8 on the earlier simulations) as the supplementary figure fig_options.

The benchmark's dispersed-clone worlds (study migration, migrate 0.2 and 0.4: one and two of the five clones dispersed, sim.npz "migrating"),
seeds 1 and 2, SpOTr exactly as the benchmark runs it (scripts/fit_spotr.build, the tree, the config's alpha): default weights against weight 0
on the dispersed clones (omega; Methods, the structural term), both fitted here so that the two arms differ only in the weights.
Per clone: the Pearson correlation of its per-spot proportion with the truth (451_.per_clone, as Fig. 2) and its spread ratio, the root mean
square distance of its cells' expected positions from their centroid over that of their true positions.
Writes spotr_out/options_demo/dispersed_weights.json and SpOTr-overleaf/figures/fig_options.{pdf,png}.

`strong` (2026-11-03): the same with distinct clone expression (sigma_tree 0.3; worlds built with config_weights.yaml), where expression can
place a released clade -- at sigma_tree 0.06 the weights changed the mean per-clone PCC by only +0.003 to +0.027 (4 of 4 world-seeds better),
as a released clade is then mapped by expression that can barely place it; its figure goes to spotr_out/options_demo until chosen.

Usage: python 471_clade_weights_dispersed.py [strong]   (benchmark_env, GPU)
"""
import os, sys, json, importlib.util, hashlib
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; WF = ROOT / "workflows" / "cassiopeia_benchmark"; OUT = HERE / "spotr_out" / "options_demo"
FIG = ROOT.parent / "SpOTr-overleaf" / "figures"; sys.path.insert(0, str(ROOT / "src")); GRID = "#e6e5e0"; PANELS = []


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def per_clone(P, q, f289, m451):
    """per clone (clones with at least MIN_CELLS cells in the spots, as Fig. 2): PCC and spread ratio"""
    tr = q["traced"].astype(bool); K = int(q["K"]); cl = q["clone"][tr]; xyt = q["xy_true"][tr]; sp = q["spots"].astype(float)
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-300); pos = Pn @ sp; Ncl = q["Ncl"].astype(float)
    r = m451.per_clone(f289, P.T @ np.eye(K)[cl], Ncl, f289.MIN_CELLS); h = Ncl.sum(1) > 0; Q = Ncl[h] / Ncl[h].sum(1, keepdims=True)
    ks = [k for k in range(K) if Ncl[:, k].sum() >= f289.MIN_CELLS and Q[:, k].std() > 0]; assert len(ks) == len(r["PCC"])   # per_clone's clones, in order
    out = {}
    for i_, k in enumerate(ks):
        m = cl == k
        sr = float(np.sqrt(((pos[m] - pos[m].mean(0)) ** 2).sum(1).mean() / ((xyt[m] - xyt[m].mean(0)) ** 2).sum(1).mean())) if m.sum() > 1 else None
        out[str(k)] = dict(PCC=float(r["PCC"][i_]), spread=sr)
    return out


def main(strong=False):
    import warnings, yaml; warnings.filterwarnings("ignore")
    import spotr.c2l_gw as M
    fs = mod("fs", WF / "scripts" / "fit_spotr.py"); f289 = mod("f289", HERE / "289_field_metrics.py"); m451 = mod("m451", HERE / "451_fig_petracer_paper.py")
    c = yaml.safe_load(open(WF / "config.yaml")); out = WF / c["out"] / "worlds"; A = float(c["spotr_alpha"]); OUT.mkdir(parents=True, exist_ok=True)
    fj = OUT / f"dispersed_weights{'_strong' if strong else ''}.json"; res = json.load(open(fj)) if fj.exists() else {}
    for v in [x for x in c["studies"]["migration"]["vary"]["migrate"] if x > 0]:
        p = dict(c["base"]); p["migrate"] = v; p.update(dict(sigma_tree=0.3) if strong else {}); w = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
        for sd in ("seed1", "seed2"):
            d = out / w / sd; I = fs.build(str(out / w / "params.json"), str(d / "problem.npz"), str(d / "trees.npz"), str(d / "spotr.npz"))
            mig = np.load(d / "sim.npz")["migrating"].astype(int).tolist()
            for arm in ("default", "weights"):
                key = f"{v} {sd} {arm}"; fn = OUT / f"dispersed_{w}_{sd}_{arm}.npz"
                if not fn.exists():
                    extra = dict(omega={k: 0.0 for k in mig}) if arm == "weights" else {}
                    np.savez_compressed(fn, phi=np.asarray(M.fit(*I["args"], **I["common"], structure="dm", alpha_tree=A, tree_d=I["d"], **extra)["phi"], np.float16))
                res[key] = dict(world=w, dispersed=mig, clones=per_clone(np.load(fn)["phi"].astype(float), I["q"], f289, m451))
            json.dump(res, open(fj, "w"), indent=1)
            for grp in (True, False):
                a_, b_ = [[x["PCC"] for k, x in res[f"{v} {sd} {arm}"]["clones"].items() if (int(k) in mig) == grp] for arm in ("default", "weights")]
                print(f"migrate {v} {sd} {'dispersed' if grp else 'other    '} clones: PCC default {np.round(a_, 2).tolist()} -> weights {np.round(b_, 2).tolist()}", flush=True)
    figure(res, OUT if strong else FIG, "_strong" if strong else "")


def figure(res, dest=FIG, tag=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from spotr.figlayout import finish, compress_pdf
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4)); fig.subplots_adjust(wspace=0.35)
    keys = sorted({k.rsplit(" ", 1)[0] for k in res})
    def paired(ax, x0, a_, b_, mk):
        ax.plot([x0 - 0.15, x0 + 0.15], [a_, b_], color="#bdbdbd", lw=1, zorder=1)
        ax.scatter([x0 - 0.15], [a_], marker=mk, s=45, color="#F47B20", zorder=3); ax.scatter([x0 + 0.15], [b_], marker=mk, s=45, color="#7a3a10", zorder=3)
    for j, (met, ttl, yl) in enumerate((("spread", "Spread of each clone", "mapped / true spread"), ("PCC", "Per-clone agreement", "per-clone PCC"))):
        ax = axs[j]; PANELS.append((ax, "AB"[j]))
        for x0, grp in ((0, True), (1, False)):
            for key in keys:
                r0, r1 = res[f"{key} default"], res[f"{key} weights"]; mk = "o" if key.endswith("seed1") else "s"
                for k, x in r0["clones"].items():
                    y = r1["clones"].get(k)
                    if (int(k) in r0["dispersed"]) == grp and y is not None and x[met] is not None and y[met] is not None:
                        paired(ax, x0, x[met], y[met], mk)
        if met == "spread":
            ax.axhline(1.0, color="#9a9a9a", lw=1, ls=(0, (4, 3)))
        ax.set_xticks([0, 1], ["dispersed clones", "other clones"]); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=11)
        ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); ax.set_xlim(-0.5, 1.5); [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]
    axs[0].scatter([], [], color="#F47B20", label="default weights"); axs[0].scatter([], [], color="#7a3a10", label="weight 0 on the dispersed clones")
    axs[0].legend(frameon=False, fontsize=9)
    finish(fig, PANELS); dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"fig_options{tag}.pdf", bbox_inches="tight"); compress_pdf(dest / f"fig_options{tag}.pdf"); fig.savefig(dest / f"fig_options{tag}.png", dpi=80, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main(strong="strong" in sys.argv[1:])
