"""Builds the four-slide SpOTr conclusion deck of 13 October 2026 (16:9, 1920x1080 canvas scaled to the window;
arrows / space / click to page, n or the Notes button for speaker notes) as one self-contained HTML file.
Inputs: the PNGs and numbers.json that 72_slide_figs.py writes into <slides_dir>.
Published as https://claude.ai/artifact/Akh2u4WXxaCbbasqjpwtka.
Usage: python 74_build_deck.py <slides_dir> <out.html>
"""
import base64, json, sys
from pathlib import Path

F = Path(sys.argv[1]); OUT_HTML = Path(sys.argv[2])
D = F.parent
N = json.load(open(F / "numbers.json"))


def img(name, w, h, alt):
    b = base64.b64encode((F / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{alt}" style="width:{w}px;height:{h}px;object-fit:contain">'


def clone_map(w, h):
    b = base64.b64encode((D / "clonemap" / "clone_map_light.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="PEtracer M3-4: tree, true clone labels in space, clone labels inferred by SpOTr" style="width:{w}px;height:{h}px;object-fit:contain">'


pet = N["petracer"]["err"]; ag = N["petracer"]["agree"]; nspl = N["petracer"]["n_splits"]
bf = N["petracer"]["per-clade_beats_fgw"]; bs = N["petracer"]["per-clade_beats_shared"]
et = N["etracer"]
rs = N["real_perclade_summary"]; rs_pc, rs_mom = rs["per-clade"], rs["per-clade-mom"]
sel = N["selection"]
ANC = N["petracer"]["err"].get("path-law-ancfit", float("nan"))
auc = N["auc"]; ident = N["identifiability"]
law_ident = ident["Law: mixing length L"]; pa = ident["Paper: tree weight α"]; pt = ident["Paper: within-clade weight θ"]

slides = []

# ---------------------------------------------------------------- 1. intro
slides.append(f"""
<section class="slide">
  <div class="kicker">SpOTr · conclusion · 13 October 2026</div>
  <h1>Placing lineage-traced cells in tissue that has no lineage readout</h1>
  <div class="row" style="gap:64px;align-items:flex-start">
    <div class="col" style="width:860px;gap:30px">
      <p><b>Two assays, two halves.</b> Lineage tracing read in dissociated cells gives each cell a phylogeny and a transcriptome but no position. Spatial transcriptomics gives position and expression but often no lineage: many platforms cannot read the recorder, and patient tissue carries none.</p>
      <p><b>The question.</b> Where does each traced cell sit, and which clones stay where their lineage put them and which spread out?</p>
      <p><b>Where the method is needed.</b> If the spatial assay reads barcodes in more than a few percent of spots, a vote among those spots assigns lineage better than any mapping (eTRACER, PEtracer). SpOTr is for the remaining case: spatial data with no usable lineage readout.</p>
    </div>
    <div class="col" style="width:740px;gap:24px">
      <svg viewBox="0 0 740 470" width="740" height="470" aria-label="Schematic: dissociated cells with a tree on the left, spatial spots on the right, joined by an optimal transport coupling">
        <rect x="0" y="0" width="300" height="470" rx="18" fill="#f1efe8"/>
        <text x="150" y="44" text-anchor="middle" font-size="24" font-weight="600" fill="#1d1d1b" font-family="IBM Plex Sans, sans-serif">Dissociated cells</text>
        <text x="150" y="76" text-anchor="middle" font-size="19" fill="#5b5a55" font-family="IBM Plex Sans, sans-serif">expression + tree</text>
        <g stroke="#5b5a55" stroke-width="3" fill="none">
          <path d="M150 110 V150 M80 150 H220 M80 150 V200 M220 150 V200 M45 200 H115 M185 200 H255 M45 200 V250 M115 200 V250 M185 200 V250 M255 200 V250"/>
        </g>
        <g>
          <circle cx="45" cy="268" r="16" fill="#2a78d6"/><circle cx="115" cy="268" r="16" fill="#2a78d6"/>
          <circle cx="185" cy="268" r="16" fill="#eb6834"/><circle cx="255" cy="268" r="16" fill="#eb6834"/>
        </g>
        <text x="150" y="340" text-anchor="middle" font-size="19" fill="#5b5a55" font-family="IBM Plex Sans, sans-serif">no positions</text>
        <rect x="440" y="0" width="300" height="470" rx="18" fill="#f1efe8"/>
        <text x="590" y="44" text-anchor="middle" font-size="24" font-weight="600" fill="#1d1d1b" font-family="IBM Plex Sans, sans-serif">Spatial spots</text>
        <text x="590" y="76" text-anchor="middle" font-size="19" fill="#5b5a55" font-family="IBM Plex Sans, sans-serif">expression + position</text>
        <g fill="#d9d6cc">
          {''.join(f'<rect x="{470 + 46 * i}" y="{110 + 46 * j}" width="40" height="40" rx="5"/>' for i in range(5) for j in range(5))}
        </g>
        <text x="590" y="380" text-anchor="middle" font-size="19" fill="#5b5a55" font-family="IBM Plex Sans, sans-serif">no lineage</text>
        <path d="M310 230 H430" stroke="#2a78d6" stroke-width="5" fill="none"/>
        <path d="M430 230 l-18 -11 v22 z" fill="#2a78d6"/>
        <text x="370" y="205" text-anchor="middle" font-size="20" font-weight="600" fill="#2a78d6" font-family="IBM Plex Sans, sans-serif">SpOTr</text>
        <text x="370" y="266" text-anchor="middle" font-size="17" fill="#5b5a55" font-family="IBM Plex Sans, sans-serif">FGW coupling</text>
        <text x="370" y="440" text-anchor="middle" font-size="20" fill="#1d1d1b" font-family="IBM Plex Sans, sans-serif">output: each cell's place + each clone's spread</text>
      </svg>
    </div>
  </div>
  <div class="foot">1 / 4</div>
  <aside>Lineage-traced scRNA-seq and spatial transcriptomics each see half the picture. The regime where a mapping beats simply reading barcodes in situ was measured on 12 October: on eTRACER the vote still wins at 5% barcoded spots and loses only near 2%.</aside>
</section>""")

# ---------------------------------------------------------------- 2. method
slides.append(f"""
<section class="slide">
  <div class="kicker">Method</div>
  <h2>Clade-aware FGW whose tree term follows a measured growth law</h2>
  <div class="row" style="gap:56px;align-items:flex-start">
    <div class="col" style="width:930px;gap:26px">
      <div class="eq">F(T) = (1&minus;&alpha;) &lang;M, T&rang; + &alpha; &sum;<sub>i,k,j,l</sub> W<sub>ik</sub> (&Phi;<sub>ik</sub> &minus; d<sub>jl</sub>)<sup>2</sup> T<sub>ij</sub> T<sub>kl</sub></div>
      <p><b>Expression</b> (M) ties each cell to spots; <b>the tree</b> says how far apart two cells should sit (&Phi;) and how much to trust it (W). Square-loss GW is the special case W = 1, &Phi; = tree distance.</p>
      <div class="eq">r<sub>ik</sub><sup>2</sup> = L<sub>&tau;</sub><sup>2</sup> + c<sup>2</sup> n<sub>ik</sub> / 2&pi;&rho; &nbsp;&nbsp; &Phi;<sub>ik</sub> = &radic;(&pi;/2) r<sub>ik</sub> &nbsp;&nbsp; W<sub>ik</sub> &prop; 1 / r<sub>ik</sub><sup>2</sup></div>
      <p><b>Growth law:</b> the clone founded at the pair's common ancestor holds n<sub>ik</sub> cells (tree) at density &rho; (section). <b>L<sub>&tau;</sub></b>, one per clade, is how far its cells mix beyond packing: the paper's clade-aware weight, made physical.</p>
      <p><b>Learning</b> (the paper's scheme): exact FGW map &rarr; linearise at the map &rarr; Sinkhorn best response &rarr; <b>anchor loss</b> (clade mix at a few spots) &rarr; Adam on &alpha; and L<sub>&tau;</sub>.</p>
    </div>
    <div class="col" style="width:678px;gap:12px">
      {img("s2_growth_law", 678, 470, "Measured clone spread against divergence age with the packing and packing-plus-mixing predictions")}
      <p class="cap">Clone spread against divergence age, PEtracer M3-4. Packing alone fails; one mixing length fits to about 10%.</p>
    </div>
  </div>
  <div class="foot">2 / 4</div>
  <aside>Departure from paper.tex: per-clade alpha/theta on a square loss became per-clade mixing lengths in a likelihood-based tree term, because the anchor cannot identify alpha/theta (slide 3). Architecture, decomposition and learning scheme are the paper's. The law's L and c: on PEtracer they come from the other two sections (which have ground truth); fitted on the anchor alone the gain over FGW shrinks to 5-11 um. On eTRACER nothing is transferred (c = 0, L from the anchor). The learner starts each step from the exact FGW map, so the map that is scored is never the entropic one.</aside>
</section>""")

# ---------------------------------------------------------------- 3. simulation
slides.append(f"""
<section class="slide">
  <div class="kicker">Simulated: planted dispersal</div>
  <h2>A planted difference in spread is recovered; the paper's per-clade weights miss it</h2>
  <div class="row" style="gap:40px;align-items:flex-start">
    <div class="col" style="width:812px;gap:10px">
      {img("s3_identifiability", 760, 526, "Share of draws in which the anchor finds the dispersed group, for SpOTr's mixing length and the paper's alpha and theta")}
      <p class="cap">Half of PEtracer M3-4's clades get extra random dispersal σ; spots rebuilt from the moved cells; the anchor is the clade mix of 20% or 5% of spots.</p>
    </div>
    <div class="col" style="width:812px;gap:10px">
      {img("s3_perclade", 760, 526, "Learned against measured mixing length per clade at sigma 500, coloured by whether the clade was dispersed")}
      <p class="cap">One mixing length per clade, learned by gradient from the anchor alone. AUC dispersed &gt; undisturbed: {auc['s300'][0]:.2f}&ndash;{auc['s300'][1]:.2f} at σ = 300 µm, {auc['s500'][0]:.2f}&ndash;{auc['s500'][1]:.2f} at 500 µm, {min(auc['s0']):.2f}&ndash;{max(auc['s0']):.2f} with nothing planted.</p>
    </div>
  </div>
  <div class="foot">3 / 4 &nbsp;·&nbsp; SpOTr finds the dispersed group in {law_ident['300.0']:.0f}% and {law_ident['500.0']:.0f}% of draws; the paper's α in {pa['300.0']:.0f}% and {pa['500.0']:.0f}%, θ in {pt['300.0']:.0f}% and {pt['500.0']:.0f}%. Better spread estimates do not improve placement here (±25 µm).</div>
  <aside>Grid version (exact solver, 2 parameters) on the left; gradient learner with 18 parameters on the right. The moved cells keep the expression of their origin, so nothing in the data can place them better.</aside>
</section>""")

# ---------------------------------------------------------------- 4. real data
e_le, e_ee = et.get("LE", {}), et.get("EE", {})
slides.append(f"""
<section class="slide">
  <div class="kicker">Real data</div>
  <h2>Real tumours: SpOTr places cells better than FGW and Tangram; per-clade spreads rarely matter</h2>
  <div class="row" style="gap:32px;align-items:flex-start">
    <div class="col" style="width:640px;gap:8px">
      {img("s4_petracer", 640, 530, "Placement error on PEtracer for expression only, Tangram, FGW, SpOTr, the true per-clade ceiling and two controls")}
      <p class="cap"><b>PEtracer</b>, {nspl} splits, weights chosen by the anchor: SpOTr {pet['per-clade']:.0f}&ndash;{pet['path-law']:.0f} µm, FGW {pet['path-fgw']:.0f} (SpOTr better in {bf}), Tangram {pet['tangram']:.0f}. Per-clade L ties shared L, as does the true per-clade L.</p>
    </div>
    <div class="col" style="width:480px;gap:8px">
      {img("s4_perclade", 480, 452, "Rank correlation of per-clade spread with the truth, per split, for three estimates")}
      <p class="cap"><b>Which clades spread</b>: learned from a shared start, weakly ({rs_pc['mean_rho']:+.2f}). The anchor alone gives {rs_mom['mean_rho_start']:+.2f}; learning from there keeps it ({rs_mom['mean_rho']:+.2f}, {rs_mom['positive']} splits positive).</p>
    </div>
    <div class="col" style="width:480px;gap:8px">
      {img("s4_etracer", 480, 452, "Clone-density agreement on eTRACER for expression only, Tangram, FGW and SpOTr")}
      <p class="cap"><b>eTRACER</b>, scRNA-seq trees + Stereo-seq, 8 anchor draws. Early escape: SpOTr {e_ee.get('law-path', float('nan')):.2f}, FGW {e_ee.get('fgw', float('nan')):.2f}, Tangram {e_ee.get('tangram', float('nan')):.2f}. Late escape: per-clone L from the anchor {e_le.get('per-clone-mom', float('nan')):.2f}, FGW {e_le.get('fgw', float('nan')):.2f}.</p>
    </div>
  </div>
  <div class="foot">4 / 4 &nbsp;·&nbsp; Random clade-sized groups: {pet.get('per-random', float('nan')):.0f} µm. PEtracer's law is fitted on its other sections; fitted on the anchor alone, {ANC:.0f} µm. TemSOMap not run.</div>
  <aside>24 splits = 3 sections x 8 random halves of the cells, each with its own 20% anchor (half fit, half validation). Mean error: expression only {pet['expression']:.0f} um; Tangram {pet['tangram']:.0f} (density term on; same with it off); FGW {pet['path-fgw']:.0f}; SpOTr shared L {pet['path-law']:.0f}; learned weight {pet['weight']:.0f}; per-clade L {pet['per-clade']:.0f}; per-clade L started at the anchor's moments {pet['per-clade-mom']:.0f}; true per-clade L at the shared level {pet['path-oracle-rel']:.0f}. The gain over FGW is on M3-1 and M3-2; FGW wins on M3-4, where the transferred L is about twice the section's own. PEtracer's L and c are transferred from the other two sections; eTRACER uses only its own anchor. Choosing among the SpOTr variants by the anchor's validation loss looked like a gain on seeds 0-3 ({sel['seeds 0-3 (rule designed)']['selected']:.0f} um) and did not replicate on seeds 4-7 ({sel['seeds 4-7 (rule confirmed)']['selected']:.0f} vs {sel['seeds 4-7 (rule confirmed)']['path-law']:.0f} shared). Dominant clade per spot right: expression {ag['expression']:.2f}, Tangram {ag['tangram']:.2f}, FGW {ag['path-fgw']:.2f}, SpOTr {ag['per-clade']:.2f}. eTRACER late escape: expression {e_le.get('expression', float('nan')):.3f}, Tangram {e_le.get('tangram', float('nan')):.3f}, FGW {e_le.get('fgw', float('nan')):.3f}, SpOTr shared {e_le.get('law-path', float('nan')):.3f}, per-clone {e_le.get('per-clone', float('nan')):.3f}; early escape: expression {e_ee.get('expression', float('nan')):.3f}, Tangram {e_ee.get('tangram', float('nan')):.3f}, FGW {e_ee.get('fgw', float('nan')):.3f}, SpOTr shared {e_ee.get('law-path', float('nan')):.3f}, per-clone {e_ee.get('per-clone', float('nan')):.3f}. eTRACER per-clone L started at the anchor's moments: late escape {e_le.get('per-clone-mom', float('nan')):.3f} (above the shared L in 8/8 draws), early escape {e_ee.get('per-clone-mom', float('nan')):.3f} (below it in 8/8). PEtracer law fitted on the anchor alone: c = 1, {pet.get('path-law-anc1', float('nan')):.0f} um; L and c fitted, {ANC:.0f} um. Weight: the learner starts at w = 0.197, the leave-one-section-out pick from the 12 October campaign (PEtracer ground truth). That start alone, with no learning, gives 390 um (FGW at the same weight 405), and neither the anchor's picks nor learning beat it: with 5/10/20% of spots as the anchor, FGW 434/427/416, SpOTr shared L 444/406/391, SpOTr per-clade L 396/383/387 um. On eTRACER at that fixed weight: late escape SpOTr 0.47 vs FGW 0.35 (8/8 draws), early escape 0.40 vs 0.39 (5/8). SpOTr barely moves with the weight; FGW swings with it, so the eTRACER gap depends on how FGW's weight is chosen. eTRACER cells were sampled to match the tissue's clone mix using the barcodes, a leak in the cell selection that applies to every arm.</aside>
</section>""")

html = """<title>SpOTr in four slides</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap">
<style>
/* Layout: fixed 1920x1080 slides scaled to the window; one slide at a time; arrows, space or click to page; n or the Notes button shows speaker notes. */
:root { --bg: #24231f; --slide: #fbfaf7; --ink: #1d1d1b; --ink2: #5b5a55; --rule: #e2e1dc; --blue: #2a78d6; --orange: #eb6834;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif; --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  color-scheme: dark; }
html, body { height: 100%; }
body { background: var(--bg); margin: 0; overflow: hidden; font-family: var(--sans); }
.stage { position: fixed; inset: 0; display: grid; place-items: center; }
.deck { width: 1920px; height: 1080px; position: relative; transform-origin: center center; }
.slide { position: absolute; inset: 0; background: var(--slide); color: var(--ink); padding: 84px 120px 150px; box-sizing: border-box;
  display: flex; flex-direction: column; gap: 30px; border-radius: 6px; }
.slide[hidden] { display: none !important; }
.kicker { font-size: 24px; letter-spacing: .08em; text-transform: uppercase; color: var(--blue); font-weight: 600; }
h1 { font-family: var(--serif); font-weight: 500; font-size: 76px; line-height: 1.08; margin: 0; max-width: 1500px; text-wrap: balance; }
h2 { font-family: var(--serif); font-weight: 500; font-size: 54px; line-height: 1.1; margin: 0; max-width: 1660px; text-wrap: balance; }
p { font-size: 29px; line-height: 1.42; margin: 0; color: var(--ink); }
p b { font-weight: 600; }
.cap { font-size: 24px; line-height: 1.35; color: var(--ink2); }
.row { display: flex; }
.col { display: flex; flex-direction: column; }
.eq { font-family: var(--serif); font-size: 36px; padding: 18px 26px; background: #f1efe8; border-radius: 12px; color: var(--ink); }
.foot { position: absolute; left: 120px; right: 120px; bottom: 48px; font-size: 24px; color: var(--ink2); border-top: 1px solid var(--rule); padding-top: 16px; }
aside { display: none; }
.nav { position: fixed; right: 18px; bottom: calc(14px + env(safe-area-inset-bottom, 0px)); display: flex; gap: 8px; align-items: center; color: #d8d6cf; font: 14px var(--sans); }
.nav button { background: #3a3934; color: #f1efe8; border: 0; border-radius: 6px; padding: 6px 12px; font: inherit; cursor: pointer; }
.nav button:focus-visible { outline: 2px solid #86b6ef; }
.nav button[aria-pressed="true"] { background: #2a78d6; }
.notes { position: fixed; left: 16px; right: 16px; bottom: calc(56px + env(safe-area-inset-bottom, 0px)); max-height: 38vh; overflow: auto;
  background: #33322d; color: #f1efe8; font: 15px/1.55 var(--sans); padding: 14px 18px; border-radius: 8px; box-shadow: 0 6px 24px rgba(0,0,0,.35); }
.notes[hidden] { display: none; }
</style>
<div class="stage"><div class="deck" id="deck">
""" + "\n".join(slides) + """
</div></div>
<div class="notes" id="notes" hidden></div>
<div class="nav"><button id="nbtn" aria-pressed="false" title="Speaker notes (n)">Notes</button><button id="prev" aria-label="Previous slide">&larr;</button><span id="count">1 / 4</span><button id="next" aria-label="Next slide">&rarr;</button></div>
<script>
const slides = [...document.querySelectorAll('.slide')]; let cur = 0;
const notes = document.getElementById('notes'), nbtn = document.getElementById('nbtn');
function show(i) { cur = Math.max(0, Math.min(slides.length - 1, i)); slides.forEach((s, k) => s.hidden = k !== cur);
  document.getElementById('count').textContent = (cur + 1) + ' / ' + slides.length;
  const a = slides[cur].querySelector('aside'); notes.textContent = a ? a.textContent : ''; }
function toggleNotes() { notes.hidden = !notes.hidden; nbtn.setAttribute('aria-pressed', String(!notes.hidden)); }
nbtn.onclick = toggleNotes;
function fit() { const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080) * 0.98;
  document.getElementById('deck').style.transform = 'scale(' + s + ')'; }
document.addEventListener('keydown', e => { if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(cur + 1); }
  if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); show(cur - 1); }
  if (e.key === 'n' || e.key === 'N') toggleNotes(); });
document.getElementById('next').onclick = () => show(cur + 1); document.getElementById('prev').onclick = () => show(cur - 1);
window.addEventListener('resize', fit); fit(); show(0);
</script>
"""
OUT_HTML.write_text(html)
print("deck written", round(len(html) / 1e6, 2), "MB")
