# Export a Numbat clone tree (mut_graph) as an edge list with mutation counts.
# usage: Rscript export_clone_tree_any.R <numbat_out_dir> <out_csv>
suppressMessages({library(igraph)})
a <- commandArgs(trailingOnly=TRUE)
mg <- readRDS(file.path(a[1], "mut_graph_2.rds"))
vdf <- data.frame(idx=as.integer(V(mg)), clone=V(mg)$clone, stringsAsFactors=FALSE)
el <- as_edgelist(mg, names=FALSE)
edf <- data.frame(from_idx=el[,1], to_idx=el[,2], label=E(mg)$label, stringsAsFactors=FALSE)
edf$from_clone <- vdf$clone[match(edf$from_idx, vdf$idx)]
edf$to_clone   <- vdf$clone[match(edf$to_idx,   vdf$idx)]
edf$n_events <- sapply(strsplit(ifelse(is.na(edf$label), "", edf$label), ","),
                       function(x) max(1, length(x[x != ""])))
write.csv(edf[,c("from_clone","to_clone","n_events")], a[2], row.names=FALSE)
cat("wrote", a[2], "-", nrow(edf), "edges\n")
