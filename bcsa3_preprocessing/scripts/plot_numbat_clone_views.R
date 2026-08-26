#!/usr/bin/env Rscript
# Save Numbat clone-specific visualisations:
#   - mutation history graph (events on phylogeny branches)
#   - per-clone CNV profile heatmap
#
# Usage:
#   Rscript plot_numbat_clone_views.R --out_dir <numbat_out_dir> [--i 2]

suppressPackageStartupMessages({
    library(optparse)
    library(numbat)
    library(ggplot2)
})

opt <- parse_args(OptionParser(option_list = list(
    make_option("--out_dir", type = "character"),
    make_option("--i", type = "integer", default = 2)
)))

stopifnot(!is.null(opt$out_dir))
nb <- Numbat$new(out_dir = opt$out_dir, i = opt$i)

mut_path <- file.path(opt$out_dir, sprintf("mut_history_%d.png", opt$i))
prof_path <- file.path(opt$out_dir, sprintf("clone_profile_%d.png", opt$i))

cat("Saving mutation history ->", mut_path, "\n")
mut_ok <- tryCatch({
    ggsave(mut_path, nb$plot_mut_history(), width = 11, height = 5, dpi = 200)
    TRUE
}, error = function(e) {
    # Large/branching trees can hit a ggraph fontsize NA error coming from the
    # clone-size node labels; retry without them (keeps the edge mutation labels).
    cat("  default failed (", conditionMessage(e), "); retrying without clone-size labels\n", sep = "")
    tryCatch({
        ggsave(mut_path, nb$plot_mut_history(show_clone_size = FALSE),
               width = 11, height = 5, dpi = 200)
        TRUE
    }, error = function(e2) {
        cat("  mutation history failed: ", conditionMessage(e2), "\n", sep = "")
        FALSE
    })
})

cat("Saving clone profile ->", prof_path, "\n")
prof_ok <- tryCatch({
    ggsave(prof_path, nb$plot_clone_profile(), width = 11, height = 6, dpi = 200, limitsize = FALSE)
    TRUE
}, error = function(e) {
    cat("  clone profile failed: ", conditionMessage(e), "\n", sep = "")
    FALSE
})

cat("Done. mut_history:", mut_ok, " clone_profile:", prof_ok, "\n")
