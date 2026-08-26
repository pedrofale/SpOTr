#!/bin/bash
# Numbat on BCSA6 with BOTH Visium sections together (TumA1 + TumD1, no SS3).
#
# Modeling both sections jointly yields a single clone numbering / phylogeny
# shared across sections (clone labels become comparable between sections).
# Requires pileup output for both sections (run TumD1 pileup with SKIP_SS3=1).
#
# Usage: bash run_numbat_bcsa6_visium_both.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/pedro/miniconda3/envs/cnv/bin:${PATH}"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA6"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_visium_both"
JOINT_R="/home/pedro/projects/spotr/phylospatial_mapping/bcsa3_preprocessing/scripts/run_numbat_joint.R"

VISIUM_PATHS="${DATA_ROOT}/visium/BCSA6_TumA1_visium_CR/outs/filtered_feature_bc_matrix,${DATA_ROOT}/visium/BCSA6_TumD1_visium_CR/outs/filtered_feature_bc_matrix"
VISIUM_LABELS="TumA1,TumD1"

echo "=============================================="
echo " BCSA6 Numbat (Visium only, both sections)"
echo " Sections: ${VISIUM_LABELS}"
echo " Output: ${OUT_DIR}"
echo "=============================================="
echo ""

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${JOINT_R}" \
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
