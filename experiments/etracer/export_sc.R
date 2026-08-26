suppressMessages(library(Matrix))
sc <- readRDS("Tumor_Immune_Evasion_eTRACER/scRNA/sc_UMI_count.rds")
cls <- attr(sc, "class")
cat("class:", paste(cls, collapse=","), "\n")
if (any(cls %in% c("Seurat"))) {
  a1 <- attr(sc, "assays")[[1]]
  cnt <- attr(a1, "layers")[["counts"]]
  if (is.null(cnt)) cnt <- attr(a1, "counts")
  gn <- function(o){ d<-attr(o,"dimnames"); if(!is.null(d)&&!is.null(d[[1]])) return(d[[1]]); rn<-rownames(o); if(!is.null(rn)) return(rn); as.character(o) }
  f <- gn(attr(a1,"features")); cc <- gn(attr(a1,"cells"))
  if (is.null(rownames(cnt)) && length(f)==nrow(cnt)) rownames(cnt) <- f
  if (is.null(colnames(cnt)) && length(cc)==ncol(cnt)) colnames(cnt) <- cc
  md <- attr(sc,"meta.data"); if(!is.null(md)) write.csv(md, "export/sc_metadata.csv")
} else { cnt <- sc }
cat("dim:", paste(dim(cnt), collapse=" x "), "\n")
cat("rownames[1:3]:", paste(head(rownames(cnt),3), collapse=", "), "\n")
cat("colnames[1:3]:", paste(head(colnames(cnt),3), collapse=", "), "\n")
dir.create("export/sc", showWarnings=FALSE, recursive=TRUE)
writeMM(cnt, "export/sc/matrix.mtx")
write.table(rownames(cnt), "export/sc/features.tsv", quote=FALSE, row.names=FALSE, col.names=FALSE)
write.table(colnames(cnt), "export/sc/barcodes.tsv", quote=FALSE, row.names=FALSE, col.names=FALSE)
cat("done\n")
