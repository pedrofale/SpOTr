#!/bin/bash
# Numbat preprocessing: pileup and phase for BCSA6 SS3 + Visium section
#
# Usage: bash run_pileup_and_phase.sh [ncores]
#
# Set VISIUM_LABEL to choose section (default: TumA1)
#   VISIUM_LABEL=TumD1 bash run_pileup_and_phase.sh 8

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro"
SS3_ROOT="${DATA_ROOT}/SS3"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA6/numbat_preprocessing"
BCSA6_CELLS="${OUTPUT_ROOT}/../BCSA56_bcsa6_cells.txt"

# ── BCSA6 SS3 (single plate) ────────────────────────────────────────────
# Skippable: set SKIP_SS3=1 (or rely on existing allele counts) to process
# only a new Visium section without re-running the large SS3 BAM pileup.
SS3_SAMPLE="BCSA56"
SS3_ALLELE="${OUTPUT_ROOT}/${SS3_SAMPLE}/${SS3_SAMPLE}_allele_counts.tsv.gz"

if [ "${SKIP_SS3:-0}" = "1" ] || [ -f "${SS3_ALLELE}" ]; then
    echo "=============================================="
    echo "Skipping SS3 pileup (${SS3_SAMPLE}); allele counts present or SKIP_SS3=1"
    echo "  ${SS3_ALLELE}"
    echo "=============================================="
else
    bash "${SCRIPT_DIR}/make_bcsa6_ss3_cell_list.sh"
    if [ ! -f "${BCSA6_CELLS}" ]; then
        echo "ERROR: BCSA6 cell list not found: ${BCSA6_CELLS}"
        exit 1
    fi

    echo "=============================================="
    echo "Processing SS3: ${SS3_SAMPLE}"
    echo "=============================================="

    BAM="${SS3_ROOT}/${SS3_SAMPLE}.filtered.Aligned.GeneTagged.UBcorrected.sorted.bam"
    OUTDIR="${OUTPUT_ROOT}/${SS3_SAMPLE}"

    if [ ! -f "${BAM}" ]; then
        echo "ERROR: BAM file not found: ${BAM}"
        exit 1
    fi

    BARCODES_CLEAN="${OUTDIR}/barcodes.tsv"
    mkdir -p "${OUTDIR}"
    cp "${BCSA6_CELLS}" "${BARCODES_CLEAN}"

    echo "BAM: ${BAM}"
    echo "Barcodes (BCSA6 only): ${BARCODES_CLEAN} ($(wc -l < "${BARCODES_CLEAN}") cells)"
    echo "Output: ${OUTDIR}"

    ${RSCRIPT} "${PILEUP_SCRIPT}" \
        --label "${SS3_SAMPLE}" \
        --samples "${SS3_SAMPLE}" \
        --bams "${BAM}" \
        --barcodes "${BARCODES_CLEAN}" \
        --outdir "${OUTDIR}" \
        --gmap "${GMAP}" \
        --snpvcf "${SNPVCF}" \
        --paneldir "${PANELDIR}" \
        --ncores "${NCORES}" \
        --eagle "${EAGLE}" \
        --cellTAG "BC" 2>&1 | tee "${OUTDIR}/pileup_and_phase.log"

    echo "Done: ${SS3_SAMPLE}"
fi

# ── Visium section ──────────────────────────────────────────────────────
VISIUM_LABEL="${VISIUM_LABEL:-TumA1}"
VISIUM_DIR="${DATA_ROOT}/visium/BCSA6_${VISIUM_LABEL}_visium_CR/outs"
VISIUM_BAM="${VISIUM_DIR}/possorted_genome_bam.bam"
VISIUM_BARCODES_GZ="${VISIUM_DIR}/filtered_feature_bc_matrix/barcodes.tsv.gz"

echo "=============================================="
echo "Processing Visium: ${VISIUM_LABEL}"
echo "=============================================="

if [ ! -f "${VISIUM_BAM}" ]; then
    echo "ERROR: Visium BAM not found: ${VISIUM_BAM}"
    exit 1
fi

OUTDIR="${OUTPUT_ROOT}/${VISIUM_LABEL}"
mkdir -p "${OUTDIR}"

BARCODES_CLEAN="${OUTDIR}/barcodes.tsv"
zcat "${VISIUM_BARCODES_GZ}" > "${BARCODES_CLEAN}"

echo "BAM: ${VISIUM_BAM}"
echo "Barcodes: ${BARCODES_CLEAN} ($(wc -l < "${BARCODES_CLEAN}") spots)"
echo "Output: ${OUTDIR}"

${RSCRIPT} "${PILEUP_SCRIPT}" \
    --label "${VISIUM_LABEL}" \
    --samples "${VISIUM_LABEL}" \
    --bams "${VISIUM_BAM}" \
    --barcodes "${BARCODES_CLEAN}" \
    --outdir "${OUTDIR}" \
    --gmap "${GMAP}" \
    --snpvcf "${SNPVCF}" \
    --paneldir "${PANELDIR}" \
    --ncores "${NCORES}" \
    --eagle "${EAGLE}" \
    --cellTAG "CB" 2>&1 | tee "${OUTDIR}/pileup_and_phase.log"

echo "Done: ${VISIUM_LABEL}"

echo ""
echo "All preprocessing complete."
echo "Allele count files:"
ls -lh "${OUTPUT_ROOT}/${SS3_SAMPLE}/${SS3_SAMPLE}_allele_counts.tsv.gz" 2>/dev/null || echo "  ${SS3_SAMPLE}: not found (check logs)"
ls -lh "${OUTPUT_ROOT}/${VISIUM_LABEL}/${VISIUM_LABEL}_allele_counts.tsv.gz" 2>/dev/null || echo "  ${VISIUM_LABEL}: not found (check logs)"
