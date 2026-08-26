#!/bin/bash
# Numbat preprocessing: pileup and phase for BCSA2 SS3 + Visium section
#
# Runs pileup_and_phase.R on each BAM to generate allele count
# dataframes needed for Numbat's joint CNV calling.
#
# Usage: bash run_pileup_and_phase.sh [ncores]

set -euo pipefail

NCORES=${1:-8}

# Conda env with Numbat, cellsnp-lite, samtools, tabix, etc.
CNV_ENV="/home/pedro/miniconda3/envs/cnv"
export PATH="${CNV_ENV}/bin:${PATH}"
RSCRIPT="${CNV_ENV}/bin/Rscript"

# Numbat tools
NUMBAT_PATH="/home/pedro/tools/numbat"
PILEUP_SCRIPT="${NUMBAT_PATH}/numbat/inst/bin/pileup_and_phase.R"
GMAP="/home/pedro/tools/Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz"
SNPVCF="${NUMBAT_PATH}/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz"
PANELDIR="${NUMBAT_PATH}/1000G_hg38"
EAGLE="/home/pedro/tools/Eagle_v2.4.1/eagle"

# Data paths (BCSA2 SS3 + Visium live under the same share folder)
DATA_ROOT="/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro"
BCSA2_ROOT="${DATA_ROOT}/BCSA2"
OUTPUT_ROOT="/mnt/data/pedro/BCSA/BCSA2/numbat_preprocessing"
mkdir -p "${OUTPUT_ROOT}"

# ── BCSA2 SS3 (single plate) ────────────────────────────────────────────
# Skippable: set SKIP_SS3=1 (or rely on existing allele counts) to process
# only a new Visium section without re-running the large SS3 BAM pileup.
SS3_SAMPLE="BCSA2Plate2"
SS3_ALLELE="${OUTPUT_ROOT}/${SS3_SAMPLE}/${SS3_SAMPLE}_allele_counts.tsv.gz"

if [ "${SKIP_SS3:-0}" = "1" ] || [ -f "${SS3_ALLELE}" ]; then
    echo "=============================================="
    echo "Skipping SS3 pileup (${SS3_SAMPLE}); allele counts present or SKIP_SS3=1"
    echo "  ${SS3_ALLELE}"
    echo "=============================================="
else
    echo "=============================================="
    echo "Processing SS3: ${SS3_SAMPLE}"
    echo "=============================================="

    BAM="${BCSA2_ROOT}/Smartseq3-${SS3_SAMPLE}.filtered.Aligned.GeneTagged.UBcorrected.sorted.bam"
    BARCODES="${BCSA2_ROOT}/Smartseq3-${SS3_SAMPLE}kept_barcodes_binned.txt"
    OUTDIR="${OUTPUT_ROOT}/${SS3_SAMPLE}"

    if [ ! -f "${BAM}" ]; then
        echo "ERROR: BAM file not found: ${BAM}"
        exit 1
    fi
    if [ ! -f "${BARCODES}" ]; then
        echo "ERROR: Barcodes file not found: ${BARCODES}"
        exit 1
    fi

    # Extract barcode column (XC, skip header)
    BARCODES_CLEAN="${OUTDIR}/barcodes.tsv"
    mkdir -p "${OUTDIR}"
    tail -n +2 "${BARCODES}" | cut -d',' -f1 > "${BARCODES_CLEAN}"

    echo "BAM: ${BAM}"
    echo "Barcodes: ${BARCODES_CLEAN} ($(wc -l < "${BARCODES_CLEAN}") cells)"
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
VISIUM_LABEL="${VISIUM_LABEL:-TumC2}"
VISIUM_DIR="${BCSA2_ROOT}/BCSA2_${VISIUM_LABEL}_visium"
REPO_DATA="${REPO_DATA:-/home/pedro/projects/spotr/phylospatial_mapping/data}"
VISIUM_BAM_LOCAL="${REPO_DATA}/BCSA2_${VISIUM_LABEL}_visium_bam.bam"
VISIUM_BAM="${VISIUM_BAM:-${VISIUM_BAM_LOCAL}}"
if [ ! -f "${VISIUM_BAM}" ]; then
    VISIUM_BAM="${VISIUM_DIR}/possorted_genome_bam.bam"
fi
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

# Decompress Visium barcodes (10x format, no header)
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

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "All preprocessing complete."
echo "Allele count files:"
ls -lh "${OUTPUT_ROOT}/${SS3_SAMPLE}/${SS3_SAMPLE}_allele_counts.tsv.gz" 2>/dev/null || echo "  ${SS3_SAMPLE}: not found (check logs)"
ls -lh "${OUTPUT_ROOT}/${VISIUM_LABEL}/${VISIUM_LABEL}_allele_counts.tsv.gz" 2>/dev/null || echo "  ${VISIUM_LABEL}: not found (check logs)"
