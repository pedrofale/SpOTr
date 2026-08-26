#!/usr/bin/env Rscript
# Convert a zUMIs dgecounts.rds into a genes x cells CSV (umicount/inex/all),
# matching the format the result notebooks expect (sc.AnnData(umi_counts.T)).
#
# Usage:
#   Rscript convert_ss3_dgecounts.R --rds <dgecounts.rds> --out <out.csv>

suppressPackageStartupMessages({
    library(optparse)
    library(Matrix)
})

opt <- parse_args(OptionParser(option_list = list(
    make_option("--rds", type = "character"),
    make_option("--out", type = "character")
)))
stopifnot(!is.null(opt$rds), !is.null(opt$out))

cat("Reading:", opt$rds, "\n")
dge <- readRDS(opt$rds)
mat <- dge[["umicount"]][["inex"]][["all"]]
cat("Matrix:", nrow(mat), "genes x", ncol(mat), "cells\n")

dense <- as.matrix(mat)
write.csv(dense, opt$out)
cat("Wrote:", opt$out, "\n")
