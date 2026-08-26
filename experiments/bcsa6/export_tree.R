# Export Numbat clone tree (mut_graph_2) with full event labels + clone GT sets.
suppressMessages({library(igraph)})
a <- commandArgs(trailingOnly=TRUE)
nd <- a[1]; outdir <- a[2]
mg <- readRDS(file.path(nd, "mut_graph_2.rds"))
vdf <- data.frame(idx=as.integer(V(mg)), clone=V(mg)$clone,
                  gt=if(!is.null(V(mg)$GT)) V(mg)$GT else NA, stringsAsFactors=FALSE)
el <- as_edgelist(mg, names=FALSE)
edf <- data.frame(from_idx=el[,1], to_idx=el[,2], label=E(mg)$label,
                  length=if(!is.null(E(mg)$length)) E(mg)$length else NA,
                  stringsAsFactors=FALSE)
edf$from_clone <- vdf$clone[match(edf$from_idx, vdf$idx)]
edf$to_clone   <- vdf$clone[match(edf$to_idx,   vdf$idx)]
edf$n_events <- sapply(strsplit(ifelse(is.na(edf$label), "", edf$label), ","),
                       function(x) max(1, length(x[x != ""])))
write.csv(edf[,c("from_clone","to_clone","label","n_events","length")],
          file.path(outdir,"bcsa6_clone_edges_full.csv"), row.names=FALSE)
write.csv(vdf, file.path(outdir,"bcsa6_clone_nodes.csv"), row.names=FALSE)
cat("edges:", nrow(edf), " nodes:", nrow(vdf), "\n")
print(head(vdf, 25))
