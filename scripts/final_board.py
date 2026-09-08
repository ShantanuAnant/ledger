"""The honest board row: lever 1 (checkpoint) x lever 2 (readout), on TEST.

    PYTHONPATH=. .venv/bin/python scripts/final_board.py

Both selections are made on VALIDATION and the number quoted is the TEST score
of what val chose. That ordering is the entire point: a per-task max over test
runs is test-set selection and is not submittable, which is why the older
best-per-task 73.72 could never be published.

  lever 1  which CHECKPOINT scores each task   (scripts/lever1_best.json)
  lever 2  which READOUT that checkpoint uses  (derived vs probe@1024)

The published LEDGER row is validation numbers filed under a header that reads
"on the official test split" -- `board.py --split` defaults to "val". So the
comparison printed here is our test against a board column that is, for our
row only, val. Both are shown and neither is quietly swapped for the other.
"""
from __future__ import annotations
import glob, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from leaderboard import LB, fetch, parse, render          # noqa: E402

CLS = {"user-churn", "item-churn", "user-visits", "user-clicks", "user-repeat",
       "user-ignore", "driver-dnf", "driver-top3", "user-engagement",
       "user-badge", "study-outcome"}


def probe(path):
    """task-header -> (zero_label, probe) from an icl_readout jsonl."""
    out = {}
    for f in glob.glob(path):
        for ln in open(f):
            if not ln.strip():
                continue
            d = json.loads(ln)
            r = d.get("results", d)
            k = f"{d['dataset']} {d['task']}"
            if "roc_auc" in r:
                out[k] = (r["zero_label_roc_auc"] * 100, r["roc_auc"] * 100)
            elif "nmae" in r:
                out[k] = (r["zero_label_nmae"], r["nmae"])
    return out


def main():
    if not LB.exists():
        fetch()
    boards = parse()
    v, t = probe("runs/lever2_val.jsonl"), probe("runs/lever2_test.jsonl")
    print(f"lever2 coverage: val {len(v)}, test {len(t)}\n")

    for board, higher in (("Classification", True), ("Regression", False)):
        want = [x.strip() for x in boards[board]["head"][4:]]
        want = [w.replace("  ", " ") for w in want]
        cols = {}
        for label, pick in (
                ("derived readout (zero labels)", lambda z, p: z),
                ("probe@1024 (ICL-cluster budget)", lambda z, p: p),
                ("val-selected readout", None)):
            col = {}
            for k in want:
                if k not in t:
                    continue
                tz, tp = t[k]
                if pick is None:
                    if k not in v:
                        col[k] = tz
                        continue
                    vz, vp = v[k]
                    better = (vp > vz) if higher else (vp < vz)
                    col[k] = tp if better else tz
                else:
                    col[k] = pick(tz, tp)
            cols[label] = col
        for label, col in cols.items():
            if not col:
                continue
            print(f"\n{'=' * 78}\n{board} -- {label} (TEST)\n{'=' * 78}")
            render(board, boards[board], col, higher)


if __name__ == "__main__":
    main()
