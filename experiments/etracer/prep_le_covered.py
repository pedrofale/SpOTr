"""eTRACER LE restricted to WELL-COVERED clones, so per-clade alpha is estimable.

The full-tissue run gave the tree effect (+0.108) but nothing for per-clade alpha (+0.0036
over a uniform alpha at the same mean). The diagnosis was data starvation, not absence of the
effect: 10 of 45 clones drew ZERO labelled bins in a 5,000-bin uniform sample, 8 of the 23
comparison clones had fewer than five, and validation early stopping fired at iteration 2 of
60 before alpha could differentiate at all. The alpha-vs-premise correlation rose monotonically
as starved clones were dropped -- +0.171 at >=1 bin, +0.286 at >=5, +0.543 at >=10 -- which is
what an estimability problem looks like rather than a null.

So: keep only clones with at least MIN_BINS barcoded bins in the whole tissue, take ALL of
their labelled bins rather than a uniform sample, and add unlabelled bins for realism. Every
retained clade then has data behind its alpha_k.

This is a power restriction, not a fishing expedition: the criterion is coverage, decided
before looking at any alpha, and applied to clones not to results.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_le_full as P

HERE = Path(__file__).resolve().parent


def main(min_bins=20, n_unlabelled=2500, n_cells=4200, seed=0):
    B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
    sp = pd.read_csv(B / "spatial/spatial_lineage_LE.csv", low_memory=False)
    sp["lineageGrp"] = sp.lineageGrp.astype(str)
    vc = sp.lineageGrp.value_counts()
    keep = sorted(vc[vc >= min_bins].index)
    print(f"[cov] {len(vc)} clones with any barcoded bin; {len(keep)} have >= {min_bins}")
    print(f"[cov] retained clones cover {100*vc[keep].sum()/vc.sum():.1f}% of barcoded bins "
          f"({vc[keep].sum()} of {vc.sum()})")
    print(f"[cov] bins per retained clone: min {vc[keep].min()}, median "
          f"{int(vc[keep].median())}, max {vc[keep].max()}")
    (HERE / "le_covered_clones.txt").write_text("\n".join(keep))
    print(f"[cov] wrote le_covered_clones.txt -- pass to prep via --restrict-clones")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bins", type=int, default=20)
    main(**vars(ap.parse_args()))
