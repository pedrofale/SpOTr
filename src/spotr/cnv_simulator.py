from typing import Optional, List, Iterable

import numpy as np
import pandas as pd
import itertools
import ete3
from scipy.linalg import cholesky

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.DataSimulator import DataSimulator


class CNVSimulator(DataSimulator):
    """
    Simulate CNV data from the lineage tree and spatial coordinates.

    Args:
        trait_covariances: trait-trait covariance matrix 
        spatial_covariances: Pixel-pixel covariance matrix
        random_seed: A seed for reproducibility
    """

    def __init__(
        self,
        n_genes: int,
        n_regions: int,
        n_regions_per_division: int,
        random_seed: Optional[int] = None,
    ):
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.n_regions_per_division = n_regions_per_division
        self.random_seed = random_seed
        self.gene_names = ["G" + str(i) for i in range(self.n_genes)]
        # Assign genes evenly to regions
        genes_per_region = self.n_genes // self.n_regions
        self.region_gene_map = {
            region: [
                i
                for i in range(region * genes_per_region, (region + 1) * genes_per_region)
            ]
            for region in range(self.n_regions)
        }

    def sample_cnv(self, tree):
        cnvs = {tree.root: np.zeros(self.n_genes)}
        for parent, child in tree.depth_first_traverse_edges(source=tree.root):
            parent_cnvs = cnvs[parent]
            branch_length = tree.get_branch_length(parent, child)
            # Sample regions to be affected
            affected_regions = np.random.choice(self.n_regions, size=min(int(self.n_regions_per_division * branch_length), self.n_regions), replace=False)
            # If CNV of any region is 0, ignore that region
            affected_regions = affected_regions[parent_cnvs[affected_regions] != 0]
            # Sample CNV events for the affected regions
            region_events = np.random.choice(2, size=len(affected_regions), p=[0.5, 0.5])
            # Map region events to gene events
            gene_events = np.zeros(self.n_genes)
            for region in affected_regions:
                gene_events[self.region_gene_map[region]] = region_events[region]
            cnvs[child] = parent_cnvs + gene_events
        return pd.DataFrame(list(cnvs.values()), index=list(cnvs.keys()), columns=self.gene_names)

    def overlay_data(self, tree: CassiopeiaTree):
        if self.random_seed:
            np.random.seed(self.random_seed)

        cnvs = self.sample_cnv(tree)
        
        # Set cell meta
        cell_meta = (
            tree.cell_meta.copy()
            if tree.cell_meta is not None
            else pd.DataFrame(index=tree.leaves)
        )

        columns = self.gene_names
        cell_meta[columns] = np.nan
        for leaf in tree.leaves:
            cell_meta.loc[leaf, columns] = cnvs.loc[leaf]

        tree.cell_meta = cell_meta