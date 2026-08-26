#!/bin/bash
# BCSA3 joint Numbat run: both SS3 plates + the new Visium section C1
# (same tumor region as D1) in one model, so SS3 cells and C1 spots share
# a single clone numbering -> lets us check how C1 matches the SS3 data.
# Output: /mnt/data/pedro/BCSA/BCSA3/numbat_out_ss3_C1
#
# Usage: bash run_numbat_bcsa3_joint_C1.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

NAS_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
VISIUM_ROOT="/mnt/data/pedro/BCSA/BCSA3"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA3"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_ss3_C1"

VISIUM_PATHS="${VISIUM_ROOT}/V19T26-031_C1/outs/filtered_feature_bc_matrix"
VISIUM_LABELS="C1"

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${SCRIPT_DIR}/run_numbat_joint.R" \
    --ss3_plates "BCSA3_1,BCSA3_2" \
    --ss3_data_root "${NAS_ROOT}" \
    --ss3_layout "zumis_plate" \
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

echo "Done. Results: ${OUT_DIR}/"
ls -lh "${OUT_DIR}/" 2>/dev/null
