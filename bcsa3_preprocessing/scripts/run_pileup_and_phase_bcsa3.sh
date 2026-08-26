#!/bin/bash
# Numbat preprocessing for BCSA3: pileup_and_phase on
#   - 2 SS3 plates (BCSA3_1, BCSA3_2)   [zUMIs layout, cellTAG BC]
#   - 2 Visium sections (B1, D1)         [10x layout,   cellTAG CB]
#
# Each step is skipped if its allele_counts.tsv.gz already exists, so the
# script is restartable. Set ONLY_SS3=1 / ONLY_VISIUM=1 to run a subset.
#
# Usage: bash run_pileup_and_phase_bcsa3.sh [ncores]

set -euo pipefail

NCORES=${1:-8}

CNV_ENV="/home/pedro/miniconda3/envs/cnv"
export PATH="${CNV_ENV}/bin:${PATH}"
RSCRIPT="${CNV_ENV}/bin/Rscript"

NUMBAT_PATH="/home/pedro/tools/numbat"
PILEUP_SCRIPT="${NUMBAT_PATH}/numbat/inst/bin/pileup_and_phase.R"
GMAP="/home/pedro/tools/Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz"
SNPVCF="${NUMBAT_PATH}/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz"
PANELDIR="${NUMBAT_PATH}/1000G_hg38"
EAGLE="/home/pedro/tools/Eagle_v2.4.1/eagle"

NAS_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
VISIUM_ROOT="/mnt/data/pedro/BCSA/BCSA3"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA3/numbat_preprocessing"
mkdir -p "${OUTPUT_ROOT}"

run_pileup () {
    local label="$1" bam="$2" barcodes="$3" celltag="$4"
    local outdir="${OUTPUT_ROOT}/${label}"
    local allele="${outdir}/${label}_allele_counts.tsv.gz"
    if [ -f "${allele}" ]; then
        echo "== Skipping ${label}: allele counts already exist (${allele})"
        return 0
    fi
    echo "=============================================="
    echo "Processing ${label}  (cellTAG ${celltag})"
    echo "  BAM: ${bam}"
    echo "  Barcodes: ${barcodes} ($(wc -l < "${barcodes}") cells)"
    echo "=============================================="
    mkdir -p "${outdir}"
    ${RSCRIPT} "${PILEUP_SCRIPT}" \
        --label "${label}" \
        --samples "${label}" \
        --bams "${bam}" \
        --barcodes "${barcodes}" \
        --outdir "${outdir}" \
        --gmap "${GMAP}" \
        --snpvcf "${SNPVCF}" \
        --paneldir "${PANELDIR}" \
        --ncores "${NCORES}" \
        --eagle "${EAGLE}" \
        --cellTAG "${celltag}" 2>&1 | tee "${outdir}/pileup_and_phase.log"
    echo "Done: ${label}"
}

# ── SS3 plates ──────────────────────────────────────────────────────────
if [ "${ONLY_VISIUM:-0}" != "1" ]; then
    for plate in BCSA3_1 BCSA3_2; do
        BAM="${NAS_ROOT}/${plate}/${plate}.filtered.Aligned.GeneTagged.UBcorrected.sorted.bam"
        KEPT="${NAS_ROOT}/${plate}/zUMIs_output/${plate}kept_barcodes_binned.txt"
        OUTDIR="${OUTPUT_ROOT}/${plate}"
        mkdir -p "${OUTDIR}"
        BARCODES_CLEAN="${OUTDIR}/barcodes.tsv"
        if [ ! -f "${OUTDIR}/${plate}_allele_counts.tsv.gz" ]; then
            [ -f "${BAM}" ]  || { echo "ERROR: missing BAM ${BAM}"; exit 1; }
            [ -f "${KEPT}" ] || { echo "ERROR: missing barcodes ${KEPT}"; exit 1; }
            tail -n +2 "${KEPT}" | cut -d',' -f1 > "${BARCODES_CLEAN}"
        fi
        run_pileup "${plate}" "${BAM}" "${BARCODES_CLEAN}" "BC"
    done
fi

# ── Visium sections ─────────────────────────────────────────────────────
if [ "${ONLY_SS3:-0}" != "1" ]; then
    # label -> cellranger section dir name
    declare -A VIS_DIR=( [B1]="V19T26-031_B1" [D1]="V19T26-031_D1" [C1]="V19T26-031_C1" )
    for label in B1 D1 C1; do
        SECTION="${VIS_DIR[$label]}"
        BAM="${VISIUM_ROOT}/${SECTION}/outs/possorted_genome_bam.bam"
        BARCODES_GZ="${VISIUM_ROOT}/${SECTION}/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"
        OUTDIR="${OUTPUT_ROOT}/${label}"
        mkdir -p "${OUTDIR}"
        BARCODES_CLEAN="${OUTDIR}/barcodes.tsv"
        if [ ! -f "${OUTDIR}/${label}_allele_counts.tsv.gz" ]; then
            [ -f "${BAM}" ] || { echo "ERROR: missing Visium BAM ${BAM}"; exit 1; }
            zcat "${BARCODES_GZ}" > "${BARCODES_CLEAN}"
        fi
        run_pileup "${label}" "${BAM}" "${BARCODES_CLEAN}" "CB"
    done
fi

echo ""
echo "All BCSA3 preprocessing complete. Allele count files:"
for label in BCSA3_1 BCSA3_2 B1 D1 C1; do
    ls -lh "${OUTPUT_ROOT}/${label}/${label}_allele_counts.tsv.gz" 2>/dev/null \
        || echo "  ${label}: not found (check logs)"
done
