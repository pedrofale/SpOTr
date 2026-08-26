suppressMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE)
inf <- args[1]; out <- args[2]
sp <- readRDS(inf)
a1 <- attr(sp, "assays")[[1]]
lay <- attr(a1, "layers")
cnt <- lay[["counts"]]
getnames <- function(o) {
  d <- attr(o, "dimnames"); if (!is.null(d) && !is.null(d[[1]])) return(d[[1]])
  rn <- rownames(o); if (!is.null(rn)) return(rn)
  as.character(o)
}
feats <- getnames(attr(a1, "features"))
cells <- getnames(attr(a1, "cells"))
cat("counts dim:", paste(dim(cnt), collapse=" x "), " | features:", length(feats), " cells:", length(cells), "\n")
if (is.null(rownames(cnt)) && length(feats)==nrow(cnt)) rownames(cnt) <- feats
if (is.null(colnames(cnt)) && length(cells)==ncol(cnt)) colnames(cnt) <- cells
dir.create(out, showWarnings=FALSE, recursive=TRUE)
writeMM(cnt, file.path(out, "matrix.mtx"))
write.table(rownames(cnt), file.path(out,"features.tsv"), quote=FALSE, row.names=FALSE, col.names=FALSE)
write.table(colnames(cnt), file.path(out,"barcodes.tsv"), quote=FALSE, row.names=FALSE, col.names=FALSE)
md <- attr(sp, "meta.data")
if (!is.null(md)) write.csv(md, file.path(out,"metadata.csv"))
cat("wrote", out, "\n")
