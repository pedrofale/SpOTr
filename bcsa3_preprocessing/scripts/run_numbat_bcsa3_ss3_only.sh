#!/bin/bash
# BCSA3 SS3-only Numbat run (both plates BCSA3_1 + BCSA3_2, HCA Epithelial ref).
# Output: /mnt/data/pedro/BCSA/BCSA3/numbat_out_ss3
#
# Usage: bash run_numbat_bcsa3_ss3_only.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

NAS_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA3"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_ss3"

mkdir -p "${OUT_DIR}"
${RSCRIPT} "${SCRIPT_DIR}/run_numbat_joint.R" \
    --ss3_plates "BCSA3_1,BCSA3_2" \
    --ss3_data_root "${NAS_ROOT}" \
    --ss3_layout "zumis_plate" \
    --allele_dir "${OUTPUT_ROOT}/numbat_preprocessing" \
    --output "${OUT_DIR}" \
    --genome "hg38" \
    --ncores "${NCORES}" \
    --gamma 20 \
    --init_k 3 \
    --max_iter 2 \
    --min_cells 10 \
    --max_entropy 0.5 \
    --call_clonal_loh TRUE \
    --reference "Epithelial" 2>&1 | tee "${OUT_DIR}/numbat_run.log"

echo "Done. Results: ${OUT_DIR}/"
ls -lh "${OUT_DIR}/" 2>/dev/null