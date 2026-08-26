suppressMessages(library(Matrix))
sp <- readRDS("Tumor_Immune_Evasion_eTRACER/SpatailTranscriptomics/spatial_RNA_count_LE.rds")
cat("attributes:", paste(names(attributes(sp)), collapse=", "), "\n\n")
as <- attr(sp, "assays")
cat("assays:", paste(names(as), collapse=", "), "\n")
a1 <- as[[1]]
cat("assay1 attrs:", paste(names(attributes(a1)), collapse=", "), "\n")
for (s in c("counts","data","layers")) {
  v <- attr(a1, s)
  if (!is.null(v)) {
    if (inherits(v,"list")) cat("  ", s, "-> list:", paste(names(v),collapse=","), "\n")
    else cat("  ", s, "dim:", paste(dim(v),collapse=" x "), " class:", class(v)[1], "\n")
  }
}
md <- attr(sp, "meta.data")
if (!is.null(md)) { cat("\nmeta.data dim:", paste(dim(md),collapse=" x "), "\n"); cat("cols:", paste(head(colnames(md),15),collapse=", "), "\n"); print(head(md[,1:min(6,ncol(md))],3)) }
