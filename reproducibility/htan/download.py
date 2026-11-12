"""Download the files of HT260C1 from the Synapse project of the HTAN data portal to $SPOTR_DATA/htan/raw, with a free Synapse account and its
credentials in ~/.synapseConfig. Requires synapseclient (pip install synapseclient).

Usage: python download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import paths  # noqa: E402

FILES = {
    "syn52176159": "HT260C1-Th1K1-matrix.mtx.gz", "syn52175994": "HT260C1-Th1K1-barcodes.tsv.gz", "syn52175932": "HT260C1-Th1K1-features.tsv.gz",
    "syn53214665": "HT260C1-Th1K1.rds", "syn61962293": "HT260C1-Th1K1XBmn1-1_arm_level.tsv",
    "syn61462534": "HT260C1-Th1K1Fc2U1Z1Bs1-matrix.mtx.gz", "syn61462025": "HT260C1-Th1K1Fc2U1Z1Bs1-barcodes.tsv.gz",
    "syn61462308": "HT260C1-Th1K1Fc2U1Z1Bs1-features.tsv.gz", "syn61462223": "HT260C1-Th1K1Fc2U1Z1Bs1-scalefactors_json.json",
    "syn61462376": "HT260C1-Th1K1Fc2U1Z1Bs1-tissue_positions_list.csv", "syn61462767": "HT260C1-Th1K1Fc2U1Z1Bs1-SeuratObj.rds",
    "syn61465334": "B1-HT260C1-Th1K1Fc2U1.tif",
}


def main():
    import synapseclient
    raw = paths.DATA / "htan" / "raw"; raw.mkdir(parents=True, exist_ok=True); syn = synapseclient.login()
    for sid, name in FILES.items():
        if not (raw / name).exists():
            syn.get(sid, downloadLocation=str(raw)); print(f"{sid} {name}", flush=True)


if __name__ == "__main__":
    main()
