suppressMessages(library(Matrix))
sp <- readRDS("Tumor_Immune_Evasion_eTRACER/SpatailTranscriptomics/spatial_RNA_count_LE.rds")
cat("spatial LE class:", class(sp), "\n")
if (inherits(sp,"list")) { cat("names:", names(sp), "\n") } else {
  cat("dim:", dim(sp), "\n"); cat("rownames[1:3]:", head(rownames(sp),3), "\n")
  cat("colnames[1:3]:", head(colnames(sp),3), "\n")
}
