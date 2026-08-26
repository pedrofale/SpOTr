#!/usr/bin/env Rscript
# Re-render the Numbat phylo-heatmap (cell x genomic-position CNV states) with an
# extra annotation bar for the Visium section (derived from the cell-name prefix,
# e.g. TumC2_/TumE2_), in addition to the default "Genotype" (clone) bar.
#
# Usage:
#   Rscript plot_panel_with_section.R --out_dir <numbat_out_dir> [--i 2] \
#       [--outfile panel_section_2.png] [--width 13] [--height 9]

suppressPackageStartupMessages({
    library(optparse)
    library(numbat)
    library(ggplot2)
    library(dplyr)
})

opt <- parse_args(OptionParser(option_list = list(
    make_option("--out_dir", type = "character"),
    make_option("--i", type = "integer", default = 2),
    make_option("--outfile", type = "character", default = NULL),
    make_option("--width", type = "double", default = 13),
    make_option("--height", type = "double", default = 9),
    make_option("--dpi", type = "double", default = 200)
)))

stopifnot(!is.null(opt$out_dir))
outfile <- if (is.null(opt$outfile)) sprintf("panel_section_%d.png", opt$i) else opt$outfile
outpath <- file.path(opt$out_dir, outfile)

cat("Loading Numbat object from:", opt$out_dir, "(i =", opt$i, ")\n")
nb <- Numbat$new(out_dir = opt$out_dir, i = opt$i)

# Section = prefix before the first underscore of each cell barcode
cells <- nb$clone_post$cell
section <- sub("_.*$", "", cells)
annot <- data.frame(cell = cells, Section = factor(section))
cat("Sections found:\n"); print(table(annot$Section))

p <- nb$plot_phylo_heatmap(
    annot = annot,
    annot_title = "Section"
)

ggsave(outpath, p, width = opt$width, height = opt$height, dpi = opt$dpi, limitsize = FALSE)
cat("Wrote:", outpath, "\n")
