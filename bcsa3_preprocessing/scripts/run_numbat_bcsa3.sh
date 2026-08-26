#!/bin/bash
# Full Numbat pipeline for BCSA2: SS3 + Visium joint analysis
#
# Usage: bash run_numbat_bcsa3.sh [ncores]
#
# Set VISIUM_LABEL env var to choose section (default: TumC2)
#   VISIUM_LABEL=TumE2 bash run_numbat_bcsa3.sh 8
#
# Steps:
#   1. pileup_and_phase on BCSA2 SS3 BAM + Visium BAM
#   2. Joint Numbat run combining SS3 cells + Visium spots

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
BCSA2_ROOT="${DATA_ROOT}/BCSA2"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA2"

export VISIUM_LABEL="${VISIUM_LABEL:-TumC2}"
VISIUM_COUNTS="${BCSA2_ROOT}/BCSA2_${VISIUM_LABEL}_visium/filtered_feature_bc_matrix"

echo "=============================================="
echo " BCSA2 Numbat Pipeline (SS3 + Visium)"
echo " Visium section: ${VISIUM_LABEL}"
echo "=============================================="
echo ""

# ── Step 1: pileup and phase (SS3 + Visium) ─────────────────────────────
echo "Step 1: Pileup and phase"
echo "----------------------------------------------"
bash "${SCRIPT_DIR}/run_pileup_and_phase.sh" "${NCORES}"

# ── Step 2: Run Numbat jointly ──────────────────────────────────────────
echo ""
echo "Step 2: Joint Numbat run"
echo "----------------------------------------------"
mkdir -p "${OUTPUT_ROOT}/numbat_out"
${RSCRIPT} "${SCRIPT_DIR}/run_numbat_joint.R" \
    --ss3_plates "BCSA2Plate2" \
    --ss3_data_root "${BCSA2_ROOT}" \
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
