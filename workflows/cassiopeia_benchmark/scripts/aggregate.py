"""All scores and readouts in one long table (run in the spotr environment).

Usage: python aggregate.py <worlds_table.json> <results dir> <out.csv>
Columns: study, knob, value, world, seed, method, metric, score.
"""
import sys, json, glob
from pathlib import Path
import pandas as pd


def main(table, res, out):
    rows = []
    for study, knob, value, wid in json.load(open(table)):
        for d in sorted(glob.glob(f"{res}/worlds/{wid}/seed*")):
            seed = int(Path(d).name[4:])
            if Path(d, "scores.json").exists():
                for method, sc in json.load(open(Path(d, "scores.json"))).items():
                    rows += [dict(study=study, knob=knob, value=value, world=wid, seed=seed, method=method, metric=m, score=v) for m, v in sc.items()]
            if Path(d, "readout.json").exists():
                rows += [dict(study=study, knob=knob, value=value, world=wid, seed=seed, method="readout", metric=m, score=v) for m, v in json.load(open(Path(d, "readout.json"))).items()]
    pd.DataFrame(rows).to_csv(out, index=False); print(f"{len(rows)} rows")


if __name__ == "__main__":
    main(*sys.argv[1:4])
