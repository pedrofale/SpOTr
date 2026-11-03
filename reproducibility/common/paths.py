"""Where the reproducibility scripts read and write, and how they reach SpOTr and cell2location.

    DATA      the downloaded and prepared data sets      $SPOTR_DATA     (default reproducibility/data)
    RESULTS   fits, maps and scores                       $SPOTR_RESULTS  (default reproducibility/results)
    FIGURES   the paper's figures                         $SPOTR_FIGURES  (default reproducibility/figures)

SpOTr is imported from the installed package, or from this repository's src/ when it is not installed.  cell2location runs in its own
environment: $SPOTR_C2L_PYTHON names that environment's Python; without it, `conda run -n c2l python`.
"""
import os
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
REPO = REPRO.parent
DATA = Path(os.environ.get("SPOTR_DATA", REPRO / "data"))
RESULTS = Path(os.environ.get("SPOTR_RESULTS", REPRO / "results"))
FIGURES = Path(os.environ.get("SPOTR_FIGURES", REPRO / "figures"))

try:
    import spotr  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO / "src"))


def out(*parts):
    """a path under RESULTS, its folder created"""
    p = RESULTS.joinpath(*parts); p.parent.mkdir(parents=True, exist_ok=True); return p


def figure(name):
    """a path under FIGURES, the folder created"""
    FIGURES.mkdir(parents=True, exist_ok=True); return FIGURES / name


def c2l_python():
    """the command that runs Python in the cell2location environment"""
    exe = os.environ.get("SPOTR_C2L_PYTHON")
    return [exe] if exe else ["conda", "run", "--no-capture-output", "-n", "c2l", "python"]
