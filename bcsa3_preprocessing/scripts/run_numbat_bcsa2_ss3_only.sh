#!/bin/bash
# Numbat on BCSA2 SS3 only (no Visium)
#
# Reuses existing pileup output from numbat_preprocessing/BCSA2Plate2/
# Writes to a separate output directory (does not touch numbat_out/).
#
# Usage: bash run_numbat_bcsa2_ss3_only.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/pedro/miniconda3/envs/cnv/bin:${PATH}"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
BCSA2_ROOT="${DATA_ROOT}/BCSA2"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA2"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_ss3"

echo "=============================================="
echo " BCSA2 Numbat (SS3 only, no Visium)"
echo " Output: ${OUT_DIR}"
echo "=============================================="
echo ""

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${SCRIPT_DIR}/run_numbat_joint.R" \
    --ss3_plates "BCSA2Plate2" \
    --ss3_data_root "${BCSA2_ROOT}" \
    --allele_dir "${OUTPUT_ROOT}/numbat_preprocessing" \
    --output "${OUT_DIR}" \
    --genome "hg38" \
    --ncores "${NCORES}" \
    --gamma 20 \
    --init_k 3 \
    --max_iter 2 \
    --min_cells 10 \
    --call_clonal_loh TRUE \
    --reference "Epithelial" 2>&1 | tee "${OUT_DIR}/numbat_run.log"

echo ""
echo "Done. Results: ${OUT_DIR}/"
ls -lh "${OUT_DIR}/" 2>/dev/null
