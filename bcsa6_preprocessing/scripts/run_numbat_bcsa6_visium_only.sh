#!/bin/bash
# Numbat on BCSA6 Visium only (no SS3)
#
# Reuses existing pileup output from numbat_preprocessing/TumA1/
# Writes to a separate output directory.
#
# Usage: bash run_numbat_bcsa6_visium_only.sh [ncores]
#
# Set VISIUM_LABEL to choose section (default: TumA1)
#   VISIUM_LABEL=TumD1 bash run_numbat_bcsa6_visium_only.sh 8

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOINT_R="${SCRIPT_DIR}/../../bcsa3_preprocessing/scripts/run_numbat_joint.R"
export PATH="/home/pedro/miniconda3/envs/cnv/bin:${PATH}"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA6"

export VISIUM_LABEL="${VISIUM_LABEL:-TumA1}"
VISIUM_COUNTS="${DATA_ROOT}/visium/BCSA6_${VISIUM_LABEL}_visium_CR/outs/filtered_feature_bc_matrix"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_visium"

echo "=============================================="
echo " BCSA6 Numbat (Visium only, no SS3)"
echo " Section: ${VISIUM_LABEL}"
echo " Output: ${OUT_DIR}"
echo "=============================================="
echo ""

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${JOINT_R}" \
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
