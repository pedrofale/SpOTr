# Export a Numbat clone tree (mut_graph) as an edge list with mutation counts.
# FIX vs experiments/bcsa3/export_clone_tree_any.R: mut_graph edge labels are
# "<parent GT>-><new events>"; the old script split the WHOLE label on "," and so
# counted parent events as branch length (e.g. "6b,9a->3a" scored 2 instead of 1).
# usage: Rscript export_clone_tree_fixed.R <numbat_out_dir> <out_csv>
suppressMessages({library(igraph)})
a <- commandArgs(trailingOnly=TRUE)
mg <- readRDS(file.path(a[1], "mut_graph_2.rds"))
el <- as_edgelist(mg, names=FALSE)
lab <- ifelse(is.na(E(mg)$label), "", E(mg)$label)
new_part <- sub("^.*->", "", lab)                       # keep only the child side
n_events <- sapply(strsplit(new_part, ","), function(x) max(1, length(x[x != ""])))
edf <- data.frame(from_clone = V(mg)$clone[el[,1]],
                  to_clone   = V(mg)$clone[el[,2]],
                  n_events   = n_events,
                  label      = lab, stringsAsFactors=FALSE)
write.csv(edf[,c("from_clone","to_clone","n_events")], a[2], row.names=FALSE)
print(edf)
cat("wrote", a[2], "-", nrow(edf), "edges\n")
