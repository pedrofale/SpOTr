library(numbat)
library(Matrix)
library(optparse)

option_list <- list(
    make_option("--ss3_plates", type = "character", default = NA,
                help = "Comma-separated SS3 sample labels (e.g. BCSA2Plate2); omit for Visium-only"),
    make_option("--ss3_data_root", type = "character",
                default = "/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro/BCSA2",
                help = "SS3 data directory (layout set by --ss3_layout)"),
    make_option("--ss3_layout", type = "character", default = "bcsa2",
                help = "Count matrix layout: bcsa2 (Smartseq3-<plate>.dgecounts.rds) or zumis (zUMIs_output/expression/<plate>.dgecounts.rds)"),
    make_option("--ss3_cells", type = "character", default = NA,
                help = "Optional file with one SS3 cell barcode per line (without plate prefix)"),
    make_option("--ref_cells", type = "character", default = NA,
                help = "Optional TSV (columns: cell, group) of normal cells used to build an internal expression reference via aggregate_counts; overrides --reference/ref_hca. 'cell' must match final count matrix names (with plate/visium prefix)."),
    make_option("--visium_path", type = "character", default = NA,
                help = "Path to Visium filtered_feature_bc_matrix directory"),
    make_option("--visium_label", type = "character", default = NA,
                help = "Label for the Visium section (e.g. TumC2)"),
    make_option("--allele_dir", type = "character",
                default = "/mnt/data/pedro/BCSA/BCSA2/numbat_preprocessing",
                help = "Directory with pileup_and_phase output (one subfolder per sample)"),
    make_option("--output", type = "character",
                default = "/mnt/data/pedro/BCSA/BCSA2/numbat_out",
                help = "Output directory for Numbat results"),
    make_option("--genome", type = "character", default = "hg38"),
    make_option("--ncores", type = "integer", default = 8),
    make_option("--gamma", type = "double", default = 20),
    make_option("--init_k", type = "integer", default = 3),
    make_option("--max_iter", type = "integer", default = 2),
    make_option("--min_cells", type = "integer", default = 10),
    make_option("--min_LLR", type = "integer", default = 5),
    make_option("--max_entropy", type = "double", default = 0.5,
                help = "Use 0.8 for Visium or joint SS3+Visium runs (Numbat recommendation)"),
    make_option("--tau", type = "double", default = 0.3),
    make_option("--t", type = "double", default = 1e-5),
    make_option("--call_clonal_loh", type = "logical", default = TRUE),
    make_option("--reference", type = "character", default = "Epithelial",
                help = "HCA reference cell type to use (column in ref_hca)"),
    make_option("--biomart", type = "character",
                default = "/home/pedro/projects/cancer/clonal_drug_response/pipeline/ss3/resources/biomart_gene_names.txt",
                help = "Biomart table mapping Ensembl IDs to gene symbols")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Harmonize phasing and REF/ALT orientation across SS3 and Visium pileups
harmonize_allele_df <- function(df) {
    flip <- df$GT == "1|0"
    if (any(flip)) {
        ref_flip <- df$REF[flip]
        df$REF[flip] <- df$ALT[flip]
        df$ALT[flip] <- ref_flip
        if ("AD" %in% names(df) && "DP" %in% names(df)) {
            df$AD[flip] <- df$DP[flip] - df$AD[flip]
        }
        df$GT[flip] <- "0|1"
    }

    canon <- df[df$GT != "", c("snp_id", "REF", "ALT")]
    canon <- canon[!duplicated(canon$snp_id), ]
    colnames(canon) <- c("snp_id", "REF_c", "ALT_c")
    df <- merge(df, canon, by = "snp_id", all.x = TRUE)

    swap <- df$REF == df$ALT_c & df$ALT == df$REF_c
    if (any(swap)) {
        ref_swap <- df$REF[swap]
        df$REF[swap] <- df$ALT[swap]
        df$ALT[swap] <- ref_swap
        if ("AD" %in% names(df) && "DP" %in% names(df)) {
            df$AD[swap] <- df$DP[swap] - df$AD[swap]
        }
    }
    df[, !colnames(df) %in% c("REF_c", "ALT_c")]
}

aggregate_gene_symbols <- function(mat) {
    if (!anyDuplicated(rownames(mat))) return(mat)
    gene_names <- rownames(mat)
    uniq <- unique(gene_names)
    idx <- match(gene_names, uniq)
    S <- sparseMatrix(
        i = seq_along(idx), j = idx, x = 1,
        dims = c(length(idx), length(uniq)),
        dimnames = list(NULL, uniq)
    )
    t(S) %*% mat
}

map_ensembl_to_symbols <- function(mat, biomart_path) {
    bm <- read.delim(biomart_path)
    ens_col <- "Gene.stable.ID"
    sym_col <- "Gene.name"
    mapping <- unique(bm[, c(ens_col, sym_col)])
    colnames(mapping) <- c("ensg", "symbol")
    mapping <- mapping[mapping$symbol != "", ]

    ensg <- sub("\\..*", "", rownames(mat))
    symbols <- mapping$symbol[match(ensg, mapping$ensg)]
    keep <- !is.na(symbols) & symbols != ""
    mat <- mat[keep, , drop = FALSE]
    rownames(mat) <- symbols[keep]
    aggregate_gene_symbols(mat)
}

ss3_cell_keep <- NULL
if (!is.na(opt$ss3_cells)) {
    ss3_cell_keep <- readLines(opt$ss3_cells)
    ss3_cell_keep <- ss3_cell_keep[nzchar(ss3_cell_keep)]
    cat("SS3 cell filter:", length(ss3_cell_keep), "barcodes from", opt$ss3_cells, "\n")
}

ss3_rds_path <- function(root, plate, layout) {
    if (layout == "bcsa2") {
        file.path(root, paste0("Smartseq3-", plate, ".dgecounts.rds"))
    } else if (layout == "zumis") {
        file.path(root, "zUMIs_output", "expression", paste0(plate, ".dgecounts.rds"))
    } else if (layout == "zumis_plate") {
        # one zUMIs output folder per plate: <root>/<plate>/zUMIs_output/expression/<plate>.dgecounts.rds
        file.path(root, plate, "zUMIs_output", "expression", paste0(plate, ".dgecounts.rds"))
    } else {
        stop("Unknown ss3_layout: ", layout, " (use bcsa2, zumis, or zumis_plate)")
    }
}

count_mat <- NULL
ss3_plates <- character(0)

# ── 1. Load SS3 count matrices (optional) ───────────────────────────────
if (!is.na(opt$ss3_plates)) {
    ss3_plates <- strsplit(opt$ss3_plates, ",")[[1]]
    cat("Loading SS3 plates:", ss3_plates, "\n")

    ss3_mats <- list()
    for (plate in ss3_plates) {
        rds_path <- ss3_rds_path(opt$ss3_data_root, plate, opt$ss3_layout)
        cat("  Reading:", rds_path, "\n")
        dgecounts <- readRDS(rds_path)
        mat <- dgecounts[["umicount"]][["inex"]][["all"]]
        if (!is.null(ss3_cell_keep)) {
            keep <- colnames(mat) %in% ss3_cell_keep
            mat <- mat[, keep, drop = FALSE]
            cat("    filtered to", ncol(mat), "cells (from", length(ss3_cell_keep), "requested)\n")
        }
        colnames(mat) <- paste0(plate, "_", colnames(mat))
        ss3_mats[[plate]] <- mat
        cat("    ", ncol(mat), "cells,", nrow(mat), "genes\n")
    }

    all_genes <- Reduce(union, lapply(ss3_mats, rownames))
    for (plate in names(ss3_mats)) {
        missing <- setdiff(all_genes, rownames(ss3_mats[[plate]]))
        if (length(missing) > 0) {
            filler <- sparseMatrix(i = integer(0), j = integer(0),
                                   dims = c(length(missing), ncol(ss3_mats[[plate]])),
                                   dimnames = list(missing, colnames(ss3_mats[[plate]])))
            ss3_mats[[plate]] <- rbind(ss3_mats[[plate]], filler)
        }
        ss3_mats[[plate]] <- ss3_mats[[plate]][all_genes, , drop = FALSE]
    }
    count_mat <- do.call(cbind, ss3_mats)
    cat("Combined SS3:", ncol(count_mat), "cells,", nrow(count_mat), "genes\n")
}

# ── 2. Load Visium count matrices (optional; one or more sections) ───────
# --visium_path / --visium_label accept comma-separated lists so that several
# Visium sections from the same tumor can be modeled together (shared clones).
visium_paths <- character(0)
visium_labels <- character(0)
if (!is.na(opt$visium_path)) {
    visium_paths <- trimws(strsplit(opt$visium_path, ",")[[1]])
    if (is.na(opt$visium_label)) {
        stop("--visium_label is required when --visium_path is set")
    }
    visium_labels <- trimws(strsplit(opt$visium_label, ",")[[1]])
    if (length(visium_paths) != length(visium_labels)) {
        stop("--visium_path and --visium_label must have the same number of comma-separated entries")
    }
}

cbind_genes <- function(a, b) {
    if (is.null(a)) return(b)
    all_genes <- union(rownames(a), rownames(b))
    miss_a <- setdiff(all_genes, rownames(a))
    if (length(miss_a) > 0) {
        a <- rbind(a, sparseMatrix(i = integer(0), j = integer(0),
                   dims = c(length(miss_a), ncol(a)),
                   dimnames = list(miss_a, colnames(a))))
    }
    miss_b <- setdiff(all_genes, rownames(b))
    if (length(miss_b) > 0) {
        b <- rbind(b, sparseMatrix(i = integer(0), j = integer(0),
                   dims = c(length(miss_b), ncol(b)),
                   dimnames = list(miss_b, colnames(b))))
    }
    cbind(a[all_genes, , drop = FALSE], b[all_genes, , drop = FALSE])
}

for (i in seq_along(visium_paths)) {
    vpath <- visium_paths[i]
    vlabel <- visium_labels[i]
    cat("Loading Visium section:", vlabel, "\n")
    cat("  Path:", vpath, "\n")

    barcodes <- read.table(gzfile(file.path(vpath, "barcodes.tsv.gz")),
                           stringsAsFactors = FALSE)$V1
    features <- read.table(gzfile(file.path(vpath, "features.tsv.gz")),
                           stringsAsFactors = FALSE)
    mat_vis <- readMM(gzfile(file.path(vpath, "matrix.mtx.gz")))

    rownames(mat_vis) <- features$V1  # ENSG IDs
    colnames(mat_vis) <- paste0(vlabel, "_", barcodes)
    cat("    ", ncol(mat_vis), "spots,", nrow(mat_vis), "genes\n")

    count_mat <- cbind_genes(count_mat, mat_vis)
}

if (is.null(count_mat)) {
    stop("Provide --ss3_plates and/or --visium_path")
}

cat("Mapping Ensembl IDs to gene symbols...\n")
count_mat <- map_ensembl_to_symbols(count_mat, opt$biomart)
cat("\nFinal count matrix:", nrow(count_mat), "genes x", ncol(count_mat), "cells/spots\n")
cat("Cells with >0 counts:", sum(colSums(count_mat) > 0), "\n")

# ── 3. Load and combine allele data ─────────────────────────────────────
df_allele_list <- list()

# SS3 allele counts
for (plate in ss3_plates) {
    allele_file <- file.path(opt$allele_dir, plate,
                             paste0(plate, "_allele_counts.tsv.gz"))
    cat("Loading SS3 allele data:", allele_file, "\n")
    df <- read.table(allele_file, sep = "\t", header = TRUE)
    if (!is.null(ss3_cell_keep)) {
        df <- df[df$cell %in% ss3_cell_keep, , drop = FALSE]
    }
    df$cell <- paste0(plate, "_", df$cell)
    df_allele_list[[plate]] <- df
    cat("  ", nrow(df), "allele records,", length(unique(df$cell)), "cells\n")
}

# Visium allele counts (one or more sections)
for (vlabel in visium_labels) {
    visium_allele_file <- file.path(opt$allele_dir, vlabel,
                                    paste0(vlabel, "_allele_counts.tsv.gz"))
    cat("Loading Visium allele data:", visium_allele_file, "\n")
    df_vis <- read.table(visium_allele_file, sep = "\t", header = TRUE)
    df_vis$cell <- paste0(vlabel, "_", df_vis$cell)
    df_allele_list[[vlabel]] <- df_vis
    cat("  ", nrow(df_vis), "allele records,", length(unique(df_vis$cell)), "spots\n")
}

df_allele <- do.call(rbind, df_allele_list)
df_allele <- harmonize_allele_df(df_allele)
cat("Combined allele data:", nrow(df_allele), "records,",
    length(unique(df_allele$cell)), "cells/spots\n\n")

# ── 3b. Build expression reference ──────────────────────────────────────
# Default: external HCA reference (one cell type column).
# If --ref_cells is given: build an internal reference from matched normal
# cells via aggregate_counts (Numbat-recommended when normals are available).
lambdas_ref <- ref_hca[, opt$reference]
if (!is.na(opt$ref_cells)) {
    ref_annot <- read.table(opt$ref_cells, header = TRUE, sep = "\t",
                            stringsAsFactors = FALSE)
    if (!all(c("cell", "group") %in% colnames(ref_annot))) {
        stop("--ref_cells file must have columns: cell, group")
    }
    ref_annot <- ref_annot[ref_annot$cell %in% colnames(count_mat), , drop = FALSE]
    if (nrow(ref_annot) == 0) {
        stop("No --ref_cells matched count matrix column names")
    }
    cat("Internal expression reference:", nrow(ref_annot), "cells\n")
    print(table(ref_annot$group))
    ref_annot$group <- as.factor(ref_annot$group)
    lambdas_ref <- aggregate_counts(count_mat[, ref_annot$cell, drop = FALSE], ref_annot)
    cat("Built internal reference:", ncol(as.matrix(lambdas_ref)), "group(s)\n")
}

# ── 4. Run Numbat ───────────────────────────────────────────────────────
dir.create(opt$output, recursive = TRUE, showWarnings = FALSE)

cat("Running Numbat...\n")
cat("  genome:", opt$genome, "\n")
cat("  init_k:", opt$init_k, "\n")
cat("  gamma:", opt$gamma, "\n")
cat("  ncores:", opt$ncores, "\n")
cat("  output:", opt$output, "\n\n")

count_mat_dense <- as.matrix(count_mat)

out <- tryCatch(
    run_numbat(
        count_mat_dense,
        lambdas_ref,
        df_allele,
        genome = opt$genome,
        t = opt$t,
        gamma = opt$gamma,
        ncores = opt$ncores,
        plot = TRUE,
        out_dir = opt$output,
        min_cells = opt$min_cells,
        min_LLR = opt$min_LLR,
        max_entropy = opt$max_entropy,
        tau = opt$tau,
        init_k = opt$init_k,
        max_iter = opt$max_iter,
        call_clonal_loh = opt$call_clonal_loh
    ),
    error = function(e) {
        # Numbat writes results incrementally; a late failure in the final
        # plotting step (e.g. ggsave "Dimensions exceed 50 inches" with many
        # clones/segments) should not discard the completed analysis.
        clone_files <- list.files(opt$output, pattern = "^clone_post_.*\\.tsv$")
        if (length(clone_files) > 0) {
            cat("\nWARNING: run_numbat raised an error after results were written:\n  ",
                conditionMessage(e), "\n",
                "Continuing: clone assignments exist (", paste(clone_files, collapse = ", "),
                ").\n", sep = "")
            return(NULL)
        }
        stop(e)
    }
)

# ── 5. Save cell metadata with source annotation ───────────────────────
cell_meta <- data.frame(
    cell = colnames(count_mat),
    source = "unknown",
    has_allele = colnames(count_mat) %in% unique(df_allele$cell)
)

if (length(ss3_plates) > 0) {
    cell_meta$source[grepl(paste0("^(", paste(ss3_plates, collapse = "|"), ")_"),
                           cell_meta$cell)] <- "SS3"
}

for (vlabel in visium_labels) {
    cell_meta$source[grepl(paste0("^", vlabel, "_"), cell_meta$cell)] <-
        paste0("Visium_", vlabel)
}

write.csv(cell_meta, file.path(opt$output, "cell_metadata.csv"), row.names = FALSE)
cat("\nDone. Results saved to:", opt$output, "\n")
