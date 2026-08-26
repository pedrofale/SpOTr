#!/bin/bash
# Build BCSA6 internal expression reference annotation for Numbat.
#
# Uses non-cancer cells (Fibroblasts + Endothelial cells) from BCSA56 metadata
# as the normal reference. Cell names are prefixed (BCSA56_) to match the
# Numbat count-matrix column names.
#
# Output: TSV with columns `cell` and `group`.
#
# Usage: bash make_bcsa6_ss3_ref_cells.sh

set -euo pipefail

DATA_ROOT="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA6"
META="${DATA_ROOT}/SS3/BCSA56_metadata.csv"
OUT="${OUTPUT_ROOT}/BCSA56_bcsa6_ref_cells.tsv"
SS3_PREFIX="BCSA56_"

python3 << PY
import pandas as pd
from pathlib import Path

REF_TYPES = ["Fibroblasts", "Endothelial cells"]

meta = pd.read_csv("${META}", index_col=0)
meta = meta.loc[meta["donor"] == "BCSA6"]
ref = meta.loc[meta["cluster_annotations"].isin(REF_TYPES)]
if len(ref) == 0:
    raise SystemExit("No reference cells found for: " + ", ".join(REF_TYPES))

df = pd.DataFrame({
    "cell": "${SS3_PREFIX}" + ref.index.astype(str),
    "group": ref["cluster_annotations"].values,
})

out = Path("${OUT}")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, sep="\t", index=False)
print(f"Wrote {len(df)} reference cells to {out}")
print(df["group"].value_counts().to_string())
PY
