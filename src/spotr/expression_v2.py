"""Expression simulation with a WORKING tree/external blend.

`expression_simulator.ExpressionSimulator.sample_combined_expression` accepts an `alpha`
argument documented in `utils.simulate_expression` as "0: fully external factors, 1: fully
tree" -- but never uses it:

    clone_expression = brownian_motion_expression
    noise            = spatial_expression + self.sample_expression_noise(...)
    expression       = clone_expression + noise        # alpha never appears

So `expression_modes` ('tree' alpha=1.0, 'mix' alpha=0.2, 'external' alpha=0.0) all produce
byte-identical output, and expression is ALWAYS fully tree-determined. Consequence: the GW
tree term is always redundant with the expression term, the best global alpha in the OT is
always 0, and the regime SpOTr is designed for -- expression cannot resolve the clones, the
tree must supply that information -- is unreachable in simulation.

This module implements the blend as documented. The external component is the tree-driven
activation matrix PERMUTED ACROSS CELLS: identical marginal distribution and scale, zero
tree structure. So alpha moves only the amount of lineage information in expression, without
changing anything else.

`spotr.expression_simulator` is not modified; import from here to opt in.
"""

import numpy as np
import pandas as pd

from .expression_simulator import ExpressionSimulator


class ExpressionSimulatorV2(ExpressionSimulator):
    """ExpressionSimulator with `alpha` actually applied."""

    def sample_combined_expression(self, tree, alpha=0.5, clades=None, rates=None,
                                   noise_std=0.5, rng=None):
        rng = np.random.default_rng() if rng is None else rng

        bm_act = self.sample_brownian_motion(clades=clades, rates=rates)
        bm_prop = np.exp(bm_act) / np.sum(np.exp(bm_act), axis=1).values[:, None]

        # External counterpart: same values, shuffled across cells -> same distribution,
        # no phylogenetic structure. Blending against this varies ONLY how much lineage
        # information expression carries.
        perm = rng.permutation(bm_prop.shape[0])
        ext_prop = pd.DataFrame(np.asarray(bm_prop)[perm],
                                index=bm_prop.index, columns=bm_prop.columns)

        alpha = float(np.clip(alpha, 0.0, 1.0))
        mixed_prop = alpha * bm_prop + (1.0 - alpha) * ext_prop
        clone_expression = mixed_prop.dot(self.trait_signatures)

        spatial_expression, spatial_activations = self.sample_spatial_effects(tree)
        noise = spatial_expression + self.sample_expression_noise(noise_std=noise_std)
        expression = clone_expression + noise

        cnvs = None
        if self.use_cnvs:
            cnvs = self.sample_cnv(tree, clades=clades, rates=rates)
            baseline = self.sample_baseline_expression()
            clone_expression = clone_expression * baseline * cnvs / 2
            expression = clone_expression + noise
            expression[np.where(cnvs == 0)[0]] = 0.0

        return expression, mixed_prop, spatial_activations, cnvs
