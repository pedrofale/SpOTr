#!/bin/bash
# Full Numbat pipeline for BCSA6: SS3 + Visium joint analysis
#
# Usage: bash run_numbat_bcsa6.sh [ncores]
#
# Set VISIUM_LABEL env var to choose section (default: TumA1)
#   VISIUM_LABEL=TumD1 bash run_numbat_bcsa6.sh 8

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOINT_R="${SCRIPT_DIR}/../../bcsa3_preprocessing/scripts/run_numbat_joint.R"
export PATH="/home/pedro/miniconda3/envs/cnv/bin:${PATH}"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro"
SS3_ROOT="${DATA_ROOT}/SS3"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA6"

export VISIUM_LABEL="${VISIUM_LABEL:-TumA1}"
VISIUM_COUNTS="${DATA_ROOT}/visium/BCSA6_${VISIUM_LABEL}_visium_CR/outs/filtered_feature_bc_matrix"
BCSA6_CELLS="${OUTPUT_ROOT}/BCSA56_bcsa6_cells.txt"
bash "${SCRIPT_DIR}/make_bcsa6_ss3_cell_list.sh"

echo "=============================================="
echo " BCSA6 Numbat Pipeline (SS3 + Visium)"
echo " Visium section: ${VISIUM_LABEL}"
echo "=============================================="
echo ""

echo "Step 1: Pileup and phase"
echo "----------------------------------------------"
bash "${SCRIPT_DIR}/run_pileup_and_phase.sh" "${NCORES}"

echo ""
echo "Step 2: Joint Numbat run"
echo "----------------------------------------------"
mkdir -p "${OUTPUT_ROOT}/numbat_out"
${RSCRIPT} "${JOINT_R}" \
    --ss3_plates "BCSA56" \
    --ss3_data_root "${SS3_ROOT}" \
    --ss3_layout "zumis" \
    --ss3_cells "${BCSA6_CELLS}" \
    --visium_path "${VISIUM_COUNTS}" \
    --visium_label "${VISIUM_LABEL}" \
    --allele_dir "${OUTPUT_ROOT}/numbat_preprocessing" \
    --output "${OUTPUT_ROOT}/numbat_out" \
    --genome "hg38" \
    --ncores "${NCORES}" \
    --gamma 20 \
    --init_k 3 \
    --max_iter 2 \
    --min_cells 10 \
    --max_entropy 0.8 \
    --call_clonal_loh TRUE \
    --reference "Epithelial" 2>&1 | tee "${OUTPUT_ROOT}/numbat_out/numbat_run.log"

echo ""
echo "=============================================="
echo " Pipeline complete"
echo "=============================================="
echo "Results: ${OUTPUT_ROOT}/numbat_out/"
ls -lh "${OUTPUT_ROOT}/numbat_out/" 2>/dev/null
