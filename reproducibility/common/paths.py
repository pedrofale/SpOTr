"""Where the reproducibility scripts read and write, and how they reach SpOTr and cell2location.

    DATA      the downloaded and prepared data sets      $SPOTR_DATA     (default reproducibility/data)
    RESULTS   maps, proportions and scores               $SPOTR_RESULTS  (default reproducibility/results)
    FIGURES   the figures of the paper                   $SPOTR_FIGURES  (default reproducibility/figures)

The scripts import SpOTr from the installed package, or from src/ in this repository when it is not installed. cell2location runs in its own
environment, whose Python is given by $SPOTR_C2L_PYTHON (by default `conda run -n c2l python`).
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
    """Return a path under RESULTS, creating its folder."""
    p = RESULTS.joinpath(*parts); p.parent.mkdir(parents=True, exist_ok=True); return p


def figure(name):
    """Return a path under FIGURES, creating its folder."""
    FIGURES.mkdir(parents=True, exist_ok=True); return FIGURES / name


def c2l_python():
    """Return the command that runs Python in the cell2location environment."""
    exe = os.environ.get("SPOTR_C2L_PYTHON")
    return [exe] if exe else ["conda", "run", "--no-capture-output", "-n", "c2l", "python"]
