#!/bin/bash
# Numbat on BCSA2 Visium only (no SS3)
#
# Reuses existing pileup output from numbat_preprocessing/TumC2/
# Writes to a separate output directory.
#
# Usage: bash run_numbat_bcsa2_visium_only.sh [ncores]
#
# Set VISIUM_LABEL to choose section (default: TumC2)
#   VISIUM_LABEL=TumE2 bash run_numbat_bcsa2_visium_only.sh 8

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/pedro/miniconda3/envs/cnv/bin:${PATH}"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
BCSA2_ROOT="${DATA_ROOT}/BCSA2"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA2"

export VISIUM_LABEL="${VISIUM_LABEL:-TumC2}"
VISIUM_COUNTS="${BCSA2_ROOT}/BCSA2_${VISIUM_LABEL}_visium/filtered_feature_bc_matrix"
# Default output preserves the original TumC2 dir name; override OUT_DIR for
# other sections (e.g. OUT_DIR=.../numbat_out_visium_TumE2) to avoid clobbering.
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/numbat_out_visium}"

echo "=============================================="
echo " BCSA2 Numbat (Visium only, no SS3)"
echo " Section: ${VISIUM_LABEL}"
echo " Output: ${OUT_DIR}"
echo "=============================================="
echo ""

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${SCRIPT_DIR}/run_numbat_joint.R" \
    --visium_path "${VISIUM_COUNTS}" \
    --visium_label "${VISIUM_LABEL}" \
    --allele_dir "${OUTPUT_ROOT}/numbat_preprocessing" \
    --output "${OUT_DIR}" \
    --genome "hg38" \
    --ncores "${NCORES}" \
    --gamma 20 \
    --init_k 3 \
    --max_iter 2 \
    --min_cells 10 \
    --max_entropy 0.8 \
    --call_clonal_loh TRUE \
    --reference "Epithelial" 2>&1 | tee "${OUT_DIR}/numbat_run.log"

echo ""
echo "Done. Results: ${OUT_DIR}/"
ls -lh "${OUT_DIR}/" 2>/dev/null
