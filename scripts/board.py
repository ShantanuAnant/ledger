"""Score every registered entity task from its dataset's WindowHead
checkpoint, then place the result on the RelBench leaderboard.

    PYTHONPATH=. .venv/bin/python scripts/board.py --split val

Reads `scripts/board_ckpts.json` (dataset -> checkpoint path) or takes
`--ckpt dataset=path` repeatedly. Each task runs in its own subprocess so one
failure costs one cell, not the sweep.

The board comparison is honest about coverage: a mean over a SUBSET of tasks
is only comparable to other methods' means over the SAME subset, so that is
what is computed, and the missing tasks are named.
"""
from __future__ import annotations

import argparse
import html as H
import json
import pathlib
import re
import subprocess
import sys
import tempfile

from ledger import win_readout as wr

LB = pathlib.Path(__file__).resolve().parent.parent / ".scratch" / "lb.html"


def leaderboard():
    """-> {"Classification": (task_names, [(mean, method, regime, values)])}"""
    if not LB.exists():
        return {}
    h = LB.read_text()
    out = {}
    for s in re.split(r"<h2>", h)[1:]:
        name = s.split("</h2>")[0]
        tbl = s.split("<table", 1)[1].split("</table>")[0]
        rows = []
        for r in re.findall(r"<tr>(.*?)</tr>", tbl, re.S):
            c = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)
            rows.append([H.unescape(re.sub("<[^>]+>", " ", x)).strip()
                         for x in c])
        tasks = [x.replace("  ", " ") for x in rows[0][4:]]
        methods = [(r[1], r[2], [float(v) for v in r[4:]]) for r in rows[1:]]
        out[name] = (tasks, methods)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                    help="dataset=path, repeatable")
    ap.add_argument("--ckpts-json", default="scripts/board_ckpts.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--corpus-at", default="rowmax")
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = {}
    p = pathlib.Path(args.ckpts_json)
    if p.exists():
        ck.update(json.loads(p.read_text()))
    for kv in args.ckpt:
        d, _, v = kv.partition("=")
        ck[d] = v

    results = {}
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as jf:
        for (ds, task) in wr.QUERIES:
            # A task's checkpoint is chosen by its ENTITY, not its dataset:
            # rel-stack post-votes is on `posts` while user-badge and
            # user-engagement are on `users`, and a model trained on one side
            # has never seen the other side's sequences.
            path = ck.get(f"{ds}/{task}") or ck.get(ds)
            if not path:
                print(f"skip {ds}/{task}: no checkpoint", flush=True)
                continue
            cmd = [sys.executable, "scripts/eval_window.py",
                   "--ckpt", path, "--dataset", ds, "--task", task,
                   "--split", args.split, "--corpus-at", args.corpus_at,
                   "--rows", str(args.rows), "--json", jf.name]
            if args.split == "test":
                cmd.append("--rebuild-at-test")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stderr.strip().splitlines() or ["?"])[-1]
                print(f"FAIL {ds}/{task}: {tail}", flush=True)
                continue
            for line in r.stdout.splitlines():
                if line.startswith(("  LEDGER", "rate:", "cold ")):
                    print(f"  {ds}/{task:20s} {line.strip()}", flush=True)
        jf.seek(0)
        for line in jf:
            if not line.strip():
                continue
            rec = json.loads(line)
            m = rec["results"]["LEDGER window-head"]
            key = f"{rec['dataset']} {rec['task']}"
            results[key] = (m["roc_auc"] * 100 if "nmae" not in m
                            else m["nmae"])

    if args.out:
        # STAMP THE SPLIT. Without it this file is a bare {task: number} map
        # with no record of which split produced it, and `leaderboard.py`
        # files it under a header reading "on the official test split". That
        # is exactly what happened: the published LEDGER row is VALIDATION
        # numbers, and it took a decimal-exact match (driver-dnf 68.49 vs
        # 68.49, study-adverse 0.1683 vs 0.1683) to notice. The reader now
        # refuses to print a val row as a test row.
        pathlib.Path(args.out).write_text(json.dumps(
            {"__split__": args.split, "__corpus_at__": args.corpus_at,
             **results}, indent=1))

    lb = leaderboard()
    for board, higher in (("Classification", True), ("Regression", False)):
        if board not in lb:
            continue
        tasks, methods = lb[board]
        have = [t for t in tasks if t in results]
        if not have:
            continue
        idx = [tasks.index(t) for t in have]
        print(f"\n{'=' * 76}\n{board}: {len(have)} of {len(tasks)} tasks scored")
        miss = [t for t in tasks if t not in results]
        if miss:
            print("  NOT scored: " + ", ".join(miss))
        print(f"\n  {'task':34s} {'LEDGER':>9s} {'best pub':>9s}")
        for t in have:
            best = (max if higher else min)(m[2][tasks.index(t)]
                                            for m in methods)
            f = "{:9.2f}" if higher else "{:9.4f}"
            print(f"  {t:34s} " + f.format(results[t]) + " " + f.format(best))
        ours = sum(results[t] for t in have) / len(have)
        rank = [(sum(m[2][i] for i in idx) / len(idx), m[0], m[1])
                for m in methods]
        rank.append((ours, f"*** LEDGER window head ({args.split}) ***",
                     "zero-shot"))
        rank.sort(key=lambda x: x[0], reverse=higher)
        print(f"\n  standing on those {len(have)} tasks "
              f"({'AUROC, higher' if higher else 'NMAE, lower'} is better):")
        for i, (v, n, reg) in enumerate(rank, 1):
            mark = "  <<<" if "LEDGER" in n else ""
            f = "{:6.2f}" if higher else "{:6.4f}"
            print(f"  {i:3d}. " + f.format(v) + f"  {n[:50]:50s} {reg}{mark}")


if __name__ == "__main__":
    main()
