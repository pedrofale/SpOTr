#!/bin/bash
# Run all three BCSA6 Numbat analyses (joint, SS3-only, Visium-only).
# Expects pileup_and_phase output under numbat_preprocessing/.
#
# Usage: bash run_numbat_bcsa6_all.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE="${SCRIPT_DIR}/run_pileup_and_phase.sh"
JOINT="${SCRIPT_DIR}/run_numbat_bcsa6.sh"
SS3="${SCRIPT_DIR}/run_numbat_bcsa6_ss3_only.sh"
VIS="${SCRIPT_DIR}/run_numbat_bcsa6_visium_only.sh"
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

wait_for_preprocessing() {
    local ss3_allele="${OUTPUT_ROOT}/numbat_preprocessing/BCSA56/BCSA56_allele_counts.tsv.gz"
    local vis_allele="${OUTPUT_ROOT}/numbat_preprocessing/${VISIUM_LABEL}/${VISIUM_LABEL}_allele_counts.tsv.gz"
    echo "Waiting for allele count files..."
    while [ ! -f "${ss3_allele}" ] || [ ! -f "${vis_allele}" ]; do
        sleep 120
        echo "  $(date -Iseconds)  SS3: $([ -f "${ss3_allele}" ] && echo done || echo pending)  Visium: $([ -f "${vis_allele}" ] && echo done || echo pending)"
    done
    echo "Preprocessing complete."
}

run_joint_only() {
    echo ""
    echo "=============================================="
    echo " BCSA6 Joint Numbat (SS3 + Visium)"
    echo " Visium section: ${VISIUM_LABEL}"
    echo "=============================================="
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
}

wait_for_preprocessing
run_joint_only
bash "${SS3}" "${NCORES}"
bash "${VIS}" "${NCORES}"

echo ""
echo "=============================================="
echo " All BCSA6 Numbat runs complete"
echo "=============================================="
ls -lh "${OUTPUT_ROOT}/numbat_out/" "${OUTPUT_ROOT}/numbat_out_ss3/" "${OUTPUT_ROOT}/numbat_out_visium/" 2>/dev/null
