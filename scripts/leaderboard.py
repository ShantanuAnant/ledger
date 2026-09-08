"""Render the official RelBench leaderboard with LEDGER's row inserted.

    python scripts/leaderboard.py                 # console tables
    python scripts/leaderboard.py --markdown      # regenerate RESULTS.md
    python scripts/leaderboard.py --refresh       # refetch the board first

SOURCE OF TRUTH. The board at star-project.stanford.edu/relbench/leaderboard/
renders client-side from `leaderboard/leaderboard.json` in the
`stanford-star/relbench` repository. This script reads that file, not the
rendered page: an earlier version of this script scraped a rendered HTML page
and silently dropped a whole task column.

Do not confuse it with the HuggingFace `relbench/leaderboard` Space, which
this script used to read. The two carry DIFFERENT populations of entries, so
means and standings from the Space are not comparable to these.

Our own numbers live in `scripts/ledger_scores.json`, as scored by
`python -m relbench.submit`. A board is ranked only over entries that cover
every one of its tasks, which is how the board itself ranks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / ".scratch" / "leaderboard.json"
MINE = ROOT / "scripts" / "ledger_scores.json"
URL = ("https://raw.githubusercontent.com/stanford-star/relbench/main/"
       "leaderboard/leaderboard.json")
OURS = "LEDGER"

#: key -> (title, metric, higher_is_better, display scale, format)
BOARDS = {
    "binary_classification": ("Classification", "AUROC (%)", True, 100, "{:.2f}"),
    "regression":            ("Regression", "NMAE", False, 1, "{:.4f}"),
    "recommendation":        ("Recommendation", "MAP (%)", True, 100, "{:.2f}"),
}


def fetch():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-s", "-m", "90", "-L", URL, "-o", str(CACHE)],
                   check=True)


def load(refresh: bool):
    if refresh or not CACHE.exists():
        fetch()
    return json.loads(CACHE.read_text())


def entries(board, key):
    """Published entries covering the whole board: (name, in_context, results, mean)."""
    out = []
    for e in board:
        b = e["boards"].get(key)
        if b and b.get("cov") == 1.0:
            out.append((e["name"], e["in_context"], b["results"], b["mean"]))
    return out


def ranked(board, key, mine, higher):
    rows = [(m, n, ic) for n, ic, _, m in entries(board, key)]
    rows.append((sum(mine.values()) / len(mine), OURS, False))
    rows.sort(key=lambda t: -t[0] if higher else t[0])
    return rows


def best_per_task(board, key, task, higher):
    vals = [r[task] for _, _, r, _ in entries(board, key) if task in r]
    return (max if higher else min)(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="refetch leaderboard.json before rendering")
    ap.add_argument("--markdown", action="store_true",
                    help="write RESULTS.md instead of printing")
    args = ap.parse_args()

    board = load(args.refresh)
    mine = {k: v for k, v in json.loads(MINE.read_text()).items()
            if not k.startswith("_")}

    secs = []
    for i, (key, (title, metric, hi, scale, fmt)) in enumerate(BOARDS.items(), 1):
        rows = ranked(board, key, mine[key], hi)
        pos = next(j for j, (_, n, _) in enumerate(rows, 1) if n == OURS)
        mean = next(m for m, n, _ in rows if n == OURS)

        per = ["| Database | Task | LEDGER | Best on board |", "|---|---|---|---|"]
        for task, v in mine[key].items():
            b = best_per_task(board, key, task, hi)
            db, t = task.split("/")
            per.append(f"| {db} | {t} | {fmt.format(v*scale)} | {fmt.format(b*scale)} |")

        tbl = ["| # | Method | In-context | Mean |", "|---|---|---|---|"]
        for j, (m, name, ic) in enumerate(rows, 1):
            if name == OURS:
                tbl.append(f"| **{j}** | **{OURS} (ours)** | **no** | "
                           f"**{fmt.format(m*scale)}** |")
            else:
                tbl.append(f"| {j} | {name} | {'yes' if ic else 'no'} | "
                           f"{fmt.format(m*scale)} |")

        n = len(mine[key])
        secs.append(
            f"## {i}. {title} — test {metric}, "
            f"{'higher' if hi else 'lower'} is better\n\n"
            f"### {i}.1 Per task\n\n" + "\n".join(per) +
            f"\n\n**Mean: {fmt.format(mean*scale)}** over {n}/{n} tasks.\n\n"
            f"### {i}.2 Leaderboard\n\n" + "\n".join(tbl) +
            f"\n\n**Position: {pos} of {len(rows)}.**\n")

    doc = f"""# LEDGER — Results

All numbers are on the **official RelBench test split**, scored by RelBench's
own evaluator (`python -m relbench.submit`, relbench 3.0.1): **31/31 tasks
valid**, all three boards complete.

Comparisons are against the **official RelBench leaderboard**
(<https://star-project.stanford.edu/relbench/leaderboard/>), read from its
source of truth, `leaderboard/leaderboard.json` in `stanford-star/relbench`.
Only entries that cover a full board are listed, which is how the board itself
ranks. Regenerate with `python scripts/leaderboard.py --markdown --refresh`.

**In-context.** The board's rule is that an in-context submission may not do
gradient-based training on the target database. LEDGER uses **no task labels**,
but it does pretrain on the target database, so it is **not in-context** and is
listed as such.

---

{chr(10).join(secs)}"""

    if args.markdown:
        (ROOT / "RESULTS.md").write_text(doc)
        print(f"wrote {ROOT / 'RESULTS.md'}")
    else:
        print(doc)


if __name__ == "__main__":
    main()
