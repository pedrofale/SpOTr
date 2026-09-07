import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, str(Path("..").resolve() / "petracer"))
from check_signature import signature
import sim_branching
print()
for sq in [None, 0.30, 0.50]:
    for bs in [8.0, 40.0]:
        s_ = sim_branching.simulate(0, bl_shape=bs, sat_q=sq)
        signature(s_["C_tree"], s_["xy_sc"], s_["branch"], s_["clade"],
                  f"sat={sq} bl_shape={bs:g}")
