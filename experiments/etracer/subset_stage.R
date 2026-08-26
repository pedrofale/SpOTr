suppressMessages(library(Matrix))
a <- commandArgs(trailingOnly=TRUE); stage <- a[1]
gn <- function(o){ d<-attr(o,"dimnames"); if(!is.null(d)&&!is.null(d[[1]])) return(d[[1]]); rn<-rownames(o); if(!is.null(rn)) return(rn); as.character(o) }
grab <- function(path){
  x <- readRDS(path); a1 <- attr(x,"assays")[[1]]
  cnt <- attr(a1,"layers")[["counts"]]; if (is.null(cnt)) cnt <- attr(a1,"counts")
  f <- gn(attr(a1,"features")); cc <- gn(attr(a1,"cells"))
  if (is.null(rownames(cnt)) && length(f)==nrow(cnt)) rownames(cnt) <- f
  if (is.null(colnames(cnt)) && length(cc)==ncol(cnt)) colnames(cnt) <- cc
  cnt
}
dump <- function(m,out){ dir.create(out,showWarnings=FALSE,recursive=TRUE)
  writeMM(m, file.path(out,"matrix.mtx"))
  write.table(rownames(m), file.path(out,"features.tsv"), quote=FALSE,row.names=FALSE,col.names=FALSE)
  write.table(colnames(m), file.path(out,"barcodes.tsv"), quote=FALSE,row.names=FALSE,col.names=FALSE)
  cat("  ",out,paste(dim(m),collapse=" x "),"\n") }
ids <- readLines(paste0("sel_cell_ids_",stage,".txt"))
sc <- grab("Tumor_Immune_Evasion_eTRACER/scRNA/sc_UMI_count.rds")
dump(sc[, intersect(ids, colnames(sc)), drop=FALSE], paste0("export/sc_sub_",stage)); rm(sc); gc()
bids <- readLines(paste0("sel_bin_ids_",stage,".txt"))
sp <- grab(paste0("Tumor_Immune_Evasion_eTRACER/SpatailTranscriptomics/spatial_RNA_count_",stage,".rds"))
dump(sp[, intersect(bids, colnames(sp)), drop=FALSE], paste0("export/spatial_sub_",stage))
