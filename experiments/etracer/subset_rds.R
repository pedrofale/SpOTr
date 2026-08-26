suppressMessages(library(Matrix))
gn <- function(o){ d<-attr(o,"dimnames"); if(!is.null(d)&&!is.null(d[[1]])) return(d[[1]]); rn<-rownames(o); if(!is.null(rn)) return(rn); as.character(o) }
grab <- function(path){
  x <- readRDS(path); a1 <- attr(x,"assays")[[1]]
  cnt <- attr(a1,"layers")[["counts"]]; if (is.null(cnt)) cnt <- attr(a1,"counts")
  f <- gn(attr(a1,"features")); cc <- gn(attr(a1,"cells"))
  if (is.null(rownames(cnt)) && length(f)==nrow(cnt)) rownames(cnt) <- f
  if (is.null(colnames(cnt)) && length(cc)==ncol(cnt)) colnames(cnt) <- cc
  cnt
}
dump <- function(m, out){
  dir.create(out, showWarnings=FALSE, recursive=TRUE)
  writeMM(m, file.path(out,"matrix.mtx"))
  write.table(rownames(m), file.path(out,"features.tsv"), quote=FALSE, row.names=FALSE, col.names=FALSE)
  write.table(colnames(m), file.path(out,"barcodes.tsv"), quote=FALSE, row.names=FALSE, col.names=FALSE)
  cat("  wrote", out, paste(dim(m),collapse=" x "), "\n")
}
ids <- readLines("sel_cell_ids.txt")
sc <- grab("Tumor_Immune_Evasion_eTRACER/scRNA/sc_UMI_count.rds")
keep <- intersect(ids, colnames(sc)); cat("sc cells kept:", length(keep), "of", length(ids), "\n")
dump(sc[, keep, drop=FALSE], "export/sc_sub")
rm(sc); gc()
bids <- readLines("sel_bin_ids.txt")
sp <- grab("Tumor_Immune_Evasion_eTRACER/SpatailTranscriptomics/spatial_RNA_count_LE.rds")
kb <- intersect(bids, colnames(sp)); cat("bins kept:", length(kb), "of", length(bids), "\n")
dump(sp[, kb, drop=FALSE], "export/spatial_sub")
