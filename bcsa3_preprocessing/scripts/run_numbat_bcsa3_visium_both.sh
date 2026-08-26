#!/bin/bash
# BCSA3 Visium-only Numbat run, BOTH sections together (B1 + D1, shared clones).
# Output: /mnt/data/pedro/BCSA/BCSA3/numbat_out_visium_both
#
# Usage: bash run_numbat_bcsa3_visium_both.sh [ncores]

set -euo pipefail

NCORES=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSCRIPT="/home/pedro/miniconda3/envs/cnv/bin/Rscript"

VISIUM_ROOT="/mnt/data/pedro/BCSA/BCSA3"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA3"
OUT_DIR="${OUTPUT_ROOT}/numbat_out_visium_both"

VISIUM_PATHS="${VISIUM_ROOT}/V19T26-031_B1/outs/filtered_feature_bc_matrix,${VISIUM_ROOT}/V19T26-031_D1/outs/filtered_feature_bc_matrix"
VISIUM_LABELS="B1,D1"

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

echo "Done. Results: ${OUT_DIR}/"
ls -lh "${OUT_DIR}/" 2>/dev/null