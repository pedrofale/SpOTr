# The authors' annotations of HT260C1, from their Seurat objects (read without Seurat, through the meta.data slot):
#   the nuclei's cell types (HT260C1-Th1K1.rds)                       -> HT260C1_sn_meta.tsv
#   the spots' tumour purity (HT260C1-Th1K1Fc2U1Z1Bs1-SeuratObj.rds)    -> HT260C1-Th1K1Fc2U1Z1Bs1_st_meta.tsv
# Usage: Rscript metadata.R <folder with the downloaded files>
args <- commandArgs(trailingOnly = TRUE); d <- args[1]
meta <- function(rds) suppressWarnings(readRDS(file.path(d, rds))@meta.data)
sn <- meta("HT260C1-Th1K1.rds"); sn$barcode <- rownames(sn)
write.table(sn, file.path(d, "HT260C1_sn_meta.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
st <- meta("HT260C1-Th1K1Fc2U1Z1Bs1-SeuratObj.rds"); st$barcode <- rownames(st)
write.table(st, file.path(d, "HT260C1-Th1K1Fc2U1Z1Bs1_st_meta.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
