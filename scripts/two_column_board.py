"""Place BOTH readout columns on the leaderboard, from one test-split run.

    PYTHONPATH=. .venv/bin/python scripts/two_column_board.py

`icl_readout --split test` reports two numbers per task from the SAME weights:
`zero_label_*` (the derived composition, no labels at all) and the probe fitted
on 1,024 TRAIN rows. Both are reported, because the board's "zero-shot" regime
already contains both kinds -- RDB-PFN, TabPFN-2.5 and TabICL all say "ICL,
1,024-example context" and KumoRFM-2 is "in-context". The probe column is
therefore the like-for-like comparison against that cluster, and the zero-label
column is a STRICTLY STRONGER claim than any entry on the board makes. Reporting
only the second, which is what we have been doing, quietly penalises us.

Coverage is checked rather than assumed, and a partial mean is compared only
against other methods' means over the SAME subset -- HANDOFF 10.5c, where a
truncated sweep silently dropped three datasets and the mean was taken over
whatever survived.
"""
from __future__ import annotations
import glob, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from leaderboard import LB, fetch, parse, render          # noqa: E402


def val_pick(vc, tc, higher):
    """Per task, use the readout that won on VAL; report its TEST score.

    The selection rule is fixed in advance and never looks at test, so this is
    legitimate in the way a per-task max over test runs is not -- the thing
    that makes the existing best-per-task 73.72 unpublishable.

    It matters because the probe is NOT uniformly safe: it is worth +19.77 on
    driver-dnf and -8.87 on user-clicks, and val calls the sign right in both
    cases. Always-probe scores 71.74 on test, always-zero-label 69.96, and
    choosing per task on val scores 72.74.
    """
    out = {}
    for k, (tz, tp) in tc.items():
        if k not in vc:
            out[k] = tz
            continue
        vz, vp = vc[k]
        better_probe = (vp > vz) if higher else (vp < vz)
        out[k] = tp if better_probe else tz
    return out


def ours(pat="runs/icltest_*.jsonl"):
    """task-header -> (zero_label, probe), keyed as the published header is."""
    cls, reg = {}, {}
    for f in sorted(glob.glob(pat)):
        for ln in open(f):
            if not ln.strip():
                continue
            d = json.loads(ln)
            r = d.get("results", d)
            k = f"{d['dataset']} {d['task']}"
            if "roc_auc" in r:
                cls[k] = (r["zero_label_roc_auc"] * 100, r["roc_auc"] * 100)
            elif "nmae" in r:
                reg[k] = (r["zero_label_nmae"], r["nmae"])
    return cls, reg


if __name__ == "__main__":
    if not LB.exists():
        fetch()
    boards = parse()
    cls, reg = ours()
    vcls, vreg = ours("runs/icl_b200_*.jsonl")      # the val run
    print(f"coverage: classification {len(cls)}/12, regression {len(reg)}/9\n")
    for board, data, vdata, higher in (
            ("Classification", cls, vcls, True),
            ("Regression", reg, vreg, False)):
        cols = [("zero-label (no labels at all)",
                 {k: v[0] for k, v in data.items()}),
                ("probe@1024 (same budget as the ICL cluster)",
                 {k: v[1] for k, v in data.items()}),
                ("val-selected readout (per task, chosen on val)",
                 val_pick(vdata, data, higher))]
        for label, col in cols:
            print(f"\n{'=' * 78}\n{board} -- {label}\n{'=' * 78}")
            render(board, boards[board], col, higher)
