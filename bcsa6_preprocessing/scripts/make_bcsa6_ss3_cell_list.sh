#!/bin/bash
# Build BCSA6-only SS3 cell barcode list from BCSA56 metadata.
#
# BCSA56 contains cells from BCSA5 and BCSA6; Numbat should use BCSA6 only.
# Barcodes match zUMIs XC / RDS colnames / pileup BC tags.
#
# Usage: bash make_bcsa6_ss3_cell_list.sh

set -euo pipefail

DATA_ROOT="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA6"
META="${DATA_ROOT}/SS3/BCSA56_metadata.csv"
OUT="${OUTPUT_ROOT}/BCSA56_bcsa6_cells.txt"

python3 << PY
import pandas as pd
from pathlib import Path

meta = pd.read_csv("${META}", index_col=0)
cells = meta.index[meta["donor"] == "BCSA6"].astype(str)
if len(cells) == 0:
    raise SystemExit("No BCSA6 cells found in metadata")

out = Path("${OUT}")
out.parent.mkdir(parents=True, exist_ok=True)
cells.to_series().to_csv(out, header=False, index=False)
print(f"Wrote {len(cells)} BCSA6 cells to {out}")
PY
