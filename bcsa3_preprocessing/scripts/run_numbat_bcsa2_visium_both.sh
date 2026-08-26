#!/bin/bash
# Numbat on BCSA2 with BOTH Visium sections together (TumC2 + TumE2, no SS3).
#
# Modeling both sections jointly yields a single clone numbering / phylogeny
# shared across sections (clone labels become comparable between sections).
# Reuses existing pileup output for both sections.
#
# Usage: bash run_numbat_bcsa2_visium_both.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/pedro/miniconda3/envs/cnv/bin:${PATH}"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
BCSA2_ROOT="${DATA_ROOT}/BCSA2"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA2"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_visium_both"

VISIUM_PATHS="${BCSA2_ROOT}/BCSA2_TumC2_visium/filtered_feature_bc_matrix,${BCSA2_ROOT}/BCSA2_TumE2_visium/filtered_feature_bc_matrix"
VISIUM_LABELS="TumC2,TumE2"

echo "=============================================="
echo " BCSA2 Numbat (Visium only, both sections)"
echo " Sections: ${VISIUM_LABELS}"
echo " Output: ${OUT_DIR}"
echo "=============================================="
echo ""

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${SCRIPT_DIR}/run_numbat_joint.R" \
    --visium_path "${VISIUM_PATHS}" \
    --visium_label "${VISIUM_LABELS}" \
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
