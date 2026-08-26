#!/bin/bash
# Numbat on BCSA6 SS3 only, using an INTERNAL normal reference
# (Fibroblasts + Endothelial cells) instead of the external HCA reference.
#
# Reuses existing pileup output from numbat_preprocessing/BCSA56/.
# Writes to numbat_out_ss3_normref/ so the HCA-reference run is preserved.
#
# Usage: bash run_numbat_bcsa6_ss3_normref.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOINT_R="${SCRIPT_DIR}/../../bcsa3_preprocessing/scripts/run_numbat_joint.R"
export PATH="/home/pedro/miniconda3/envs/cnv/bin:${PATH}"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

DATA_ROOT="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro"
SS3_ROOT="${DATA_ROOT}/SS3"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA6"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_ss3_normref"
BCSA6_CELLS="${OUTPUT_ROOT}/BCSA56_bcsa6_cells.txt"
REF_CELLS="${OUTPUT_ROOT}/BCSA56_bcsa6_ref_cells.tsv"

bash "${SCRIPT_DIR}/make_bcsa6_ss3_cell_list.sh"
bash "${SCRIPT_DIR}/make_bcsa6_ss3_ref_cells.sh"

echo "=============================================="
echo " BCSA6 Numbat (SS3 only, internal normal ref)"
echo " Reference: Fibroblasts + Endothelial cells"
echo " Output: ${OUT_DIR}"
echo "=============================================="
echo ""

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${JOINT_R}" \
    --ss3_plates "BCSA56" \
    --ss3_data_root "${SS3_ROOT}" \
    --ss3_layout "zumis" \
    --ss3_cells "${BCSA6_CELLS}" \
    --ref_cells "${REF_CELLS}" \
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
