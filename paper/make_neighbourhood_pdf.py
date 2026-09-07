"""Render the neighbourhood-regularised OT formulation to a PDF (matplotlib mathtext)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

W, H = 8.27, 11.69                      # A4 inches
L, R, TOP, BOT = 0.105, 0.895, 0.945, 0.075
INK, INK2, ACC = "#111111", "#444444", "#8c3a1e"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})


MATH = re.compile(r"\$[^$]*\$")
TALL = re.compile(r"\\(sum|frac|partial|nabla|hat|otimes|langle)")


def _tokens(t):
    """Split into words, keeping each $...$ span as one atomic token."""
    out, i = [], 0
    for m in MATH.finditer(t):
        out += t[i:m.start()].split()
        out.append(m.group(0))
        i = m.end()
    return out + t[i:].split()


def _width(tok):
    """Rough rendered width in characters; math renders narrower than its source."""
    return len(tok) * 0.55 if tok.startswith("$") else len(tok)


def _wrap(t, cols):
    lines, cur, w = [], [], 0.0
    for tok in _tokens(" ".join(t.split())):
        tw = _width(tok)
        if cur and w + 1 + tw > cols:
            lines.append(" ".join(cur)); cur, w = [tok], tw
        else:
            cur.append(tok); w += (1 if cur[:-1] else 0) + tw
    if cur:
        lines.append(" ".join(cur))
    return lines


def _tall(ln):
    return bool(TALL.search(ln))


class Doc:
    def __init__(self, path):
        self.pdf = PdfPages(path); self.new_page()

    def new_page(self):
        self.fig = plt.figure(figsize=(W, H)); self.y = TOP

    def _room(self, dy):
        if self.y - dy < BOT:
            self.pdf.savefig(self.fig); plt.close(self.fig); self.new_page()

    def title(self, t, sub=None):
        self._room(.10)
        for ln in _wrap(t, 52):
            self.fig.text(L, self.y, ln, fontsize=17, color=INK, weight="bold", va="top")
            self.y -= .030
        self.y -= .006
        if sub:
            for ln in _wrap(sub, 92):
                self.fig.text(L, self.y, ln, fontsize=10.5, color=INK2, va="top"); self.y -= .019
        self.y -= .012
        self.fig.add_artist(plt.Line2D([L, R], [self.y, self.y], color=ACC, lw=1.4))
        self.y -= .028

    def head(self, t):
        self._room(.06)
        self.y -= .012
        self.fig.text(L, self.y, t, fontsize=13, color=ACC, weight="bold", va="top")
        self.y -= .030

    def para(self, t, size=10.2, indent=0.0):
        for ln in _wrap(t, int(94 - indent * 200)):
            dy = .0250 if _tall(ln) else .0182        # tall math needs its own leading
            self._room(dy + .004)
            self.fig.text(L + indent, self.y, ln, fontsize=size, color=INK, va="top")
            self.y -= dy
        self.y -= .008

    def eq(self, e, size=13, tag=None):
        self._room(.050)
        self.y -= .010
        self.fig.text((L + R) / 2, self.y, e, fontsize=size, color=INK, va="top", ha="center")
        if tag:
            self.fig.text(R, self.y, tag, fontsize=9.5, color=INK2, va="top", ha="right")
        self.y -= .046

    def rows(self, rows, widths, size=9.6, header=False):
        for r in rows:
            self._room(.020)
            x = L
            for cell, w in zip(r, widths):
                self.fig.text(x, self.y, cell, fontsize=size, va="top", color=INK,
                              weight="bold" if header else "normal")
                x += w
            self.y -= .0185
            if header:
                self.fig.add_artist(plt.Line2D([L, R], [self.y + .006, self.y + .006],
                                               color="#999999", lw=.7))
                header = False
        self.y -= .010

    def close(self):
        self.pdf.savefig(self.fig); plt.close(self.fig); self.pdf.close()


d = Doc(Path(__file__).resolve().parent / "neighbourhood_formulation.pdf")

d.title("Neighbourhood-regularised optimal transport for lineage-guided spatial mapping",
        "The tree constrains who is near whom, not how far apart. "
        "SpOTr working note, 31 August 2026.")

d.head("1.  Why not Gromov-Wasserstein")
d.para(r"The PEtracer tree is ultrametric: its distance is twice the number of days since the "
       r"last common ancestor, and on 20,000 random triples from M3-4 not one violates the "
       r"ultrametric condition. An ultrametric on $n$ points generally requires $n-1$ "
       r"dimensions. Concretely, on 600 M3-4 cells the top two principal coordinates of the "
       r"tree metric carry 18.1% of its variance and 90% needs 298 of them.")
d.para(r"Gromov-Wasserstein penalises $(D^T_{ik} - D^S_{jl})^2$ over coupled pairs, so it asks "
       r"the tree metric to be reproduced by 2-D tissue coordinates. That is unattainable, and "
       r"the failure is not graceful: 49% of M3-4 cell pairs sit at the tree maximum, so about "
       r"half of the structural budget is spent insisting that cells be far apart, using a "
       r"value identical for all of them.")
d.para(r"What survives the collapse is local. Embedding the tree in 2-D as well as possible "
       r"preserves only 0.607 correlation with the tree metric, but retains 37% of each cell's "
       r"5 nearest tree neighbours against a 0.8% chance rate, rising to 54% at 50 neighbours. "
       r"In the real tissue, 10.4% of a cell's 5 nearest relatives are among its 5 nearest "
       r"spatial neighbours, thirteen times chance. Neighbourhood structure is a local, "
       r"low-degree constraint that a plane can accommodate; the full metric is not.")

d.head("2.  Setup")
d.para(r"Cells $i \in \{1..n\}$ carry marginal $p$, with $P = \mathrm{diag}(p)$. Spots "
       r"$j \in \{1..m\}$ carry marginal $q$ and positions $Y \in R^{m \times 2}$ with rows "
       r"$y_j$. The expression cost is $M \in R^{n \times m}$ and the tree distance "
       r"$D^T \in R^{n \times n}$ is symmetric with zero diagonal. Couplings are")
d.eq(r"$\Pi(p,q) = \{\, T \geq 0 \;:\; T\mathbf{1}_m = p,\;\; T^\top \mathbf{1}_n = q \,\}$")

d.head("3.  The model")
d.para(r"The position the coupling assigns to cell $i$ is the conditional expectation of spot "
       r"position given that cell, which is linear in $T$:")
d.eq(r"$\hat{Y}(T) = P^{-1} T Y \in R^{n \times 2}, \qquad "
     r"\hat{y}_i = \frac{1}{p_i} \sum_j T_{ij}\, y_j$", tag="(1)")
d.para(r"From the tree take a kernel with a per-cell bandwidth $\sigma_i$, symmetrised so that "
       r"the penalty is a genuine quadratic form:")
d.eq(r"$w_{ik} = \exp(-D^T_{ik} / \sigma_i), \qquad "
     r"W = \frac{1}{2}(w + w^\top), \qquad W_{ii} = 0$", tag="(2)")
d.para(r"and penalise relatives that land apart. With $d_i = \sum_k W_{ik}$ and the graph "
       r"Laplacian $L = \mathrm{diag}(d) - W$,")
d.eq(r"$R(T) = \sum_{i,k} W_{ik} \, \| \hat{y}_i - \hat{y}_k \|^2 "
     r"= 2\, \mathrm{tr}( \hat{Y}^\top L \hat{Y} ) "
     r"= 2\, \mathrm{tr}( T^\top A\, T K )$", tag="(3)")
d.para(r"where $A = P^{-1} L P^{-1}$ and $K = Y Y^\top$. The problem is")
d.eq(r"$\min_{T \in \Pi(p,q)} \;\; F(T) = \langle M, T \rangle + \lambda R(T)$", tag="(4)")

d.head("4.  What is absent, and why it matters")
d.para(r"No target distance appears in (3). It states that relatives should be near one another "
       r"and is silent about how near, and silent about non-relatives, for which "
       r"$W_{ik} \approx 0$. Three consequences follow.")
d.para(r"The tree's metric scale becomes irrelevant. Calibration, the 70.0 ceiling and the 49% "
       r"of pairs piled there all drop out, because those pairs carry no weight.", indent=.030)
d.para(r"The penalty is one-sided and attractive. Gromov-Wasserstein is two-sided: it also "
       r"insists distant pairs be far apart, which is where half its budget went and which a "
       r"plane cannot honour in any case.", indent=.030)
d.para(r"$\sigma_i$ is a resolution, not a trust weight. A motile cell that has left its "
       r"siblings but not its clade keeps a constraint, applied at the scale on which it "
       r"remains true. Lowering $\alpha$ instead removes the constraint and hands the cell to "
       r"expression, which is much weaker: 549.7 um for expression alone against 356.8 um with "
       r"the tree on M3-4.", indent=.030)

d.head("5.  Properties")
d.para(r"Gradient. Since $A$ and $K$ are symmetric,")
d.eq(r"$\nabla_T R = 4\, A\, T K = 4\, P^{-1} L P^{-1} T Y Y^\top, \qquad "
     r"\nabla F = M + \lambda \nabla_T R$", tag="(5)")
d.para(r"verified against numerical differentiation to eight decimal places.")
d.para(r"Convexity. In vectorised form $R(T) = 2\, \mathrm{vec}(T)^\top (K \otimes A) "
       r"\mathrm{vec}(T)$. Both $L$ and $K = YY^\top$ are positive semi-definite, $A$ is "
       r"congruent to $L$, and a Kronecker product of PSD matrices is PSD, so the Hessian "
       r"$2(K \otimes A) \geq 0$ and $F$ is convex on a polytope. Checked numerically: the "
       r"minimum Hessian eigenvalue is $-3 \times 10^{-11}$, zero to machine precision. This is "
       r"the structural difference from FGW, which is non-convex; it removes the "
       r"initialisation-dependent basin selection that has been contaminating comparisons, "
       r"where a 15 um swing arose from initialisation alone.")
d.para(r"Cost. $L$ is sparse and $Y$ has two columns, so a gradient costs "
       r"$O(\mathrm{nnz}(L) + nm)$ against Gromov-Wasserstein's $O(n^2 m + n m^2)$. The "
       r"quadratic is rank two in the spatial dimension.")
d.para(r"Separability. $\partial W_{kl} / \partial \sigma_i = 0$ unless $k = i$ or $l = i$, so "
       r"cell $i$'s resolution touches only pairs incident to $i$. Under FGW the structural "
       r"gradient at row $a$ is $\sum_{k,l} (\alpha_a + \alpha_k) L_{akbl} T_{kl}$, so every "
       r"other cell's $\alpha_k$ enters row $a$. That non-separability is precisely what made "
       r"per-clade $\alpha$ ill-posed.")

d.head("6.  Algorithm")
d.para(r"Frank-Wolfe. At $T$, solve the linear transport problem "
       r"$S = \mathrm{argmin}_{S \in \Pi} \langle \nabla F(T), S \rangle$ with a network "
       r"simplex, and set $D = S - T$. Because $R$ is a pure quadratic,")
d.eq(r"$F(T + gD) = F(T) + g \langle \nabla F(T), D \rangle + g^2 \lambda R(D)$", tag="(6)")
d.para(r"holds exactly, verified to ten significant figures at $g = 0.13,\, 0.50,\, 0.91$. "
       r"Hence")
d.eq(r"$g^\star = \mathrm{clip}\left( \frac{-\langle \nabla F(T), D \rangle}"
     r"{2 \lambda R(D)},\; 0,\; 1 \right)$", tag="(7)")
d.para(r"and $R(D) \geq 0$ guarantees this is a minimiser rather than a saddle.")

d.head("7.  Estimating the bandwidth from ground truth")
d.para(r"Lineage-traced spatial data gives the true position $x_i$, so $\sigma_i$ can be read "
       r"off by leave-one-out prediction. With $Z_i = \sum_{k \neq i} w_{ik}$,")
d.eq(r"$\mu_i(\sigma) = \frac{1}{Z_i} \sum_{k \neq i} w_{ik} x_k, \qquad "
     r"s_i^2(\sigma) = \frac{1}{2 Z_i} \sum_{k \neq i} w_{ik} \| x_k - \mu_i \|^2$", tag="(8)")
d.eq(r"$\ell_i(\sigma) = -\frac{\| x_i - \mu_i(\sigma) \|^2}{2 s_i^2(\sigma)} "
     r"- \log s_i^2(\sigma), \qquad \hat{\sigma}_i = \mathrm{argmax}_\sigma \; \ell_i(\sigma)$",
     tag="(9)")
d.para(r"$\ell_i$ is the predictive log-density of $x_i$ under $\mathcal{N}(\mu_i, s_i^2 I_2)$ "
       r"up to a constant. The $-\log s_i^2$ term is what penalises breadth. Without it - "
       r"scoring by $\| x_i - \mu_i \| / \mathrm{spread}$, say - a wider neighbourhood wins by "
       r"construction and every cell selects the same scale.")

d.head("8.  What the ground truth says")
d.para(r"Applying (8) and (9) to the three PEtracer M3 sections, with positions scaled by the "
       r"tissue radius so the numbers are comparable across sections:")
d.rows([["section", "best global sigma", "mean loglik", "per-cell", "gain"],
        ["M3-4", "5", "1.571", "2.473", "+0.902"],
        ["M3-1", "5", "0.765", "1.690", "+0.925"],
        ["M3-2", "5", "0.701", "1.629", "+0.929"]],
       [.16, .21, .17, .15, .12], header=True)
d.para(r"A per-cell bandwidth is worth about 0.9 nats over the best global one, consistently "
       r"across all three sections. The distribution on M3-4 has median 3 and interquartile "
       r"range 2 to 5, with a 2% tail at $\sigma = 80$: cells that only their whole clade can "
       r"locate. Since the tree spans 0.7 to 70, the informative neighbourhood is the nearest "
       r"few percent of it.")
d.para(r"The bandwidth tracks motility, defined as a cell's distance from its 15 nearest tree "
       r"relatives in units of their spread: Spearman rho = +0.298, p = 4e-48. It is unrelated "
       r"to clade size (rho = -0.066), so it measures the cell's spatial behaviour rather than "
       r"an artefact of how clades were drawn. Scoring instead by which tree cut best predicts "
       r"a cell gives the same picture: settled cells are best served by their 10 nearest "
       r"relatives and motile cells by about 100, rho = +0.500 over 2281 cells, and the "
       r"likelihood gained by using a cell's own scale rather than the finest is 0.12 for "
       r"settled cells against 7.19 for motile ones.")

d.head("9.  What is not yet established")
d.para(r"The bandwidths above are estimated from true positions. They establish that a "
       r"per-cell scale exists and is worth having; they do not show it can be recovered "
       r"without ground truth. The natural candidate is the dispersion of a cell's relatives "
       r"under the current coupling, which is observable and updates as the map improves, but "
       r"an earlier attempt at an observable per-cell trust statistic reached only rho = +0.106 "
       r"against its oracle. That is the next thing to test.")
d.para(r"Nor has the model itself been run on the mapping task. Everything above is a "
       r"specification with verified mathematics and a measured motivation. Whether it "
       r"produces a better map than fused Gromov-Wasserstein is untested, and this project has "
       r"retired several well-motivated proposals at exactly this stage.")
d.close()
print("wrote paper/neighbourhood_formulation.pdf")
